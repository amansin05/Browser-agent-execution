"""Reasoner — picks exactly ONE action per turn against a fresh page snapshot."""

import json

from groq import BadRequestError

from browser_agent.agent.prompts import REASONER_SYSTEM
from browser_agent.config import MODEL
from browser_agent.log import get_logger

log = get_logger(__name__)


def _explore_guidance(subgoal) -> str:
    """Render the explore_spec into a guidance block for the reasoner. Without this the reasoner
    only sees the subgoal's one-line goal and is blind to WHICH sources to gather from — so on a
    leftover/unrelated tab it tries to satisfy 'find retailers' on whatever page is open instead of
    searching for them (the Amazon 'Levi's 501 retailers' failure). Returns '' for non-explore
    subgoals so their prompt is unchanged."""
    spec = subgoal.get("explore_spec") or {}
    if not spec:
        return ""
    bits = []
    sources = spec.get("sources") or []
    if sources:
        bits.append(
            f"Gather candidates from these sources: {', '.join(map(str, sources))}. If the page "
            f"you are on is NOT one of them, navigate to one first — if you don't have its URL, do "
            f"a web search (e.g. google.com) for the user's need, then open a result. Do NOT try "
            f"to satisfy this on an unrelated page.")
    # Reaching the listings page efficiently is the hard part on retail sites: the reasoner tends to
    # click around the homepage/menus and get stuck instead of going straight to the grid. Steer it
    # to the fastest route — a direct category/search URL, or the on-page search box — and away from
    # re-clicking the same nav element (the "buy a phone" homepage-click failure).
    bits.append(
        "To REACH the listings fast: use `navigate` to go DIRECTLY to a category or search-results "
        "URL (e.g. https://site/mobiles or https://site/search?q=<your query>), OR `input_text` on "
        "the search box's [index] with submit:true. Prefer this over clicking through menus, and "
        "never repeat the same click — if a click does nothing, `navigate` to a URL instead.")
    must = spec.get("must_have") or []
    if must:
        bits.append(f"Each candidate must meet: {', '.join(map(str, must))}.")
    target = spec.get("target_count")
    if target:
        bits.append(f"Aim for about {target} candidates.")
    # The biggest explore failure mode is never finishing: the reasoner keeps applying filters /
    # clicking around a results page and never signals done, so the candidate extractor (which runs
    # AFTER completion) never fires. Tell it to stop and complete as soon as a list is visible.
    bits.append("As SOON as the page shows a list of results/listings matching the goal, call "
                "subgoal_complete — do NOT keep applying filters, sorting, or clicking; the "
                "candidates are read off the page automatically once you complete.")
    return "### Exploration guidance\n" + " ".join(bits) + "\n" if bits else ""


async def reasoner_decide(groq, reasoner_tools, subgoal, snapshot_text, recent_actions, model=MODEL):
    """One reasoner turn -> (thought, action_name, action_args). Recovers from the Scout
    tool_use_failed wobble by retrying with a correction (bounded). The reasoner is text-only: when
    a page's DOM snapshot is too sparse, the orchestrator appends an `### Extracted readable content`
    block (Readability.js / trafilatura) to `snapshot_text` for it to read — there is no image path."""
    recent = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent_actions[-5:])) or "  (none yet)"
    user = (f"### Current subgoal\n{subgoal['goal']}\n"
            f"Success when: {subgoal['success_condition']}\n"
            f"{_explore_guidance(subgoal)}\n"
            f"### Live page\n{snapshot_text[:9000]}\n\n"
            f"### Recent actions\n{recent}\n\n"
            f"Choose exactly ONE action now.")
    messages = [{"role": "system", "content": REASONER_SYSTEM}, {"role": "user", "content": user}]
    for _attempt in range(4):
        try:
            resp = await groq.chat.completions.create(
                model=model, temperature=0.0, max_completion_tokens=700,
                tools=reasoner_tools, tool_choice="required", messages=messages,
            )
        except BadRequestError as e:
            detail = (getattr(e, "body", None) or {}).get("error", {}).get("message", str(e))
            log.warning("reasoner tool call rejected (attempt %d/4): %s", _attempt + 1, detail[:160])
            messages.append({"role": "user", "content":
                             f"Your tool call was rejected: {detail}\nRe-issue ONE call with only "
                             "needed params and correct JSON types (booleans/numbers unquoted)."})
            continue
        msg = resp.choices[0].message
        if not msg.tool_calls:
            # No action chosen — nudge once by treating as escalate.
            return (msg.content or "", "escalate", {"reason": "model returned no action"})
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            log.warning("reasoner returned non-JSON tool args; treating as empty: %.120r",
                        tc.function.arguments)
            args = {}
        return (msg.content or "", tc.function.name, args)
    log.warning("reasoner gave up after repeated malformed tool calls")
    return ("", "escalate", {"reason": "repeated malformed tool calls"})
