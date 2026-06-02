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
    # The biggest explore failure mode is never finishing: the reasoner keeps scrolling / filtering
    # a results page and never signals done, so the candidate extractor (which runs AFTER completion)
    # never fires — and every scroll is wasted because extraction reads the WHOLE page (including
    # off-screen items), not just what's in view. Tell it to complete the instant a list is visible.
    bits.append("As SOON as the page shows a list of results/listings matching the goal, call "
                "subgoal_complete IMMEDIATELY — do NOT scroll, filter, sort, or click to 'gather' "
                "more. Scrolling does NOT help: the candidates (including off-screen ones) are read "
                "off the full page automatically the moment you complete. One scroll at most if the "
                "list isn't visible yet, then complete.")
    return "### Exploration guidance\n" + " ".join(bits) + "\n" if bits else ""


async def reasoner_decide(groq, reasoner_tools, subgoal, snapshot_text, recent_actions,
                          last_thought="", plan_context="", model=MODEL):
    """One reasoner turn -> (thought, actions) where `actions` is an ordered list of (name, args).
    The model MAY emit several tool calls in one turn (e.g. fill a few fields then submit); the
    orchestrator executes them in order behind a page-change guard. `thought` is the model's
    free-text note (its evaluation of the last action + what it's doing now), echoed back next turn
    via `last_thought` for continuity. Recovers from the Scout tool_use_failed wobble by retrying
    with a correction (bounded). Text-only: when the page is too sparse, the orchestrator appends an
    `### Extracted readable content` block to `snapshot_text` — there is no image path."""
    recent = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent_actions[-6:])) or "  (none yet)"
    note = f"### Your previous note\n{last_thought.strip()}\n\n" if last_thought.strip() else ""
    progress = f"### Plan progress\n{plan_context.strip()}\n\n" if plan_context.strip() else ""
    user = (f"{progress}### Current subgoal\n{subgoal['goal']}\n"
            f"Success when: {subgoal['success_condition']}\n"
            f"{_explore_guidance(subgoal)}\n"
            f"### Live page\n{snapshot_text[:9000]}\n\n"
            f"{note}### Recent actions\n{recent}\n\n"
            f"First, briefly evaluate whether your previous action worked and note what to remember. "
            f"Then choose the next action. You MAY issue SEVERAL actions in this turn ONLY when they "
            f"are safe to chain on the SAME page (e.g. fill multiple fields, then submit) — the page "
            f"view refreshes after any navigation/click, so never queue actions past one of those.")
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
                             f"Your tool call was rejected: {detail}\nRe-issue your call(s) with only "
                             "needed params and correct JSON types (booleans/numbers unquoted)."})
            continue
        msg = resp.choices[0].message
        if not msg.tool_calls:
            # No action chosen — nudge once by treating as escalate.
            return (msg.content or "", [("escalate", {"reason": "model returned no action"})])
        actions: list[tuple[str, dict]] = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                log.warning("reasoner returned non-JSON tool args; treating as empty: %.120r",
                            tc.function.arguments)
                args = {}
            actions.append((tc.function.name, args))
        return (msg.content or "", actions)
    log.warning("reasoner gave up after repeated malformed tool calls")
    return ("", [("escalate", {"reason": "repeated malformed tool calls"})])
