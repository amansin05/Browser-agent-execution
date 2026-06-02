"""Planner — decomposes a goal into ordered subgoals (and re-plans on escalation)."""

from browser_agent.agent.prompts import ENHANCED_PLANNER
from browser_agent.config import PLANNER_MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


def _normalize(sg: dict, idx: int) -> dict:
    """Fill subgoal defaults so older/simpler plans (no type/tier) still work, and derive the
    tier <-> needs_approval relationship both ways."""
    sg_type = sg.get("type", "act")
    tier = sg.get("tier")
    if tier not in ("auto", "confirm", "secured"):
        tier = "secured" if sg.get("needs_approval") else "auto"
    needs_approval = bool(sg.get("needs_approval", tier in ("confirm", "secured")))
    out = {
        "id": sg.get("id", idx),
        "type": sg_type,
        "goal": sg["goal"],
        "success_condition": sg.get("success_condition", ""),
        "tier": tier,
        "needs_approval": needs_approval,
    }
    for key in ("explore_spec", "scoring", "options"):
        if sg.get(key) is not None:
            out[key] = sg[key]
    if sg_type == "ask_user":
        out["allow_free_text"] = bool(sg.get("allow_free_text", True))
        out["multi_select"] = bool(sg.get("multi_select", False))
        out["question"] = sg.get("question", sg["goal"])
    return out


async def plan_subgoals(groq, goal, capabilities, *, prior_plan=None, failed_id=None,
                        reason=None, observations=None, preamble="", model=PLANNER_MODEL) -> list[dict]:
    system = ENHANCED_PLANNER.format(capabilities=", ".join(capabilities))
    # `preamble` carries session memory (rolling history + scratchpad) so a goal like
    # "now do X to it" can resolve against what earlier tasks did and left on screen.
    head = (preamble + "\n\n") if preamble else ""
    if prior_plan is None:
        user = f"{head}User goal: {goal}"
    else:
        user = (f"{head}User goal: {goal}\n"
                f"The plan is failing at subgoal {failed_id}.\n"
                f"Reason from the reasoner: {reason!r}\n"
                f"Observations so far: {observations or '(none)'}\n"
                f"Revise the remaining subgoals (keep finished ones conceptually done). Same JSON schema.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # PLANNER_MODEL is a THINKING model: its <think> reasoning shares the completion budget with the
    # JSON, so a tight cap truncates the JSON mid-structure (-> JSONDecodeError). Give it ample room
    # AND a couple of retries — a rambly goal can blow a smaller budget (the "track my package"
    # eval failure), so the first attempt gets the most headroom.
    attempts = 3
    last_err: Exception | None = None
    for _attempt in range(attempts):
        resp = await groq.chat.completions.create(
            model=model, temperature=0.0, max_completion_tokens=8192, messages=messages,
        )
        content = resp.choices[0].message.content
        try:
            data = extract_json(content)
            subgoals = data["subgoals"] if isinstance(data, dict) else data
            return [_normalize(sg, idx) for idx, sg in enumerate(subgoals, 1)]
        except (ValueError, KeyError, TypeError) as e:
            # The usual culprit is the THINKING phase eating the whole budget so no JSON is emitted.
            # Retry with qwen3's `/no_think` soft-switch: it skips reasoning and emits the JSON
            # directly (an empty <think></think> + the object), which can't truncate. The first
            # attempt keeps full reasoning for plan quality; the retries trade it for reliability.
            log.warning("planner JSON parse failed (attempt %d/%d): %r", _attempt + 1, attempts, e)
            last_err = e
            messages.append({"role": "assistant", "content": content or ""})
            messages.append({"role": "user", "content":
                             "/no_think That could not be parsed as the required JSON (the reasoning "
                             "likely ran long and truncated it). Skip all reasoning and reply with "
                             "ONLY the JSON object for the schema — no <think>, no prose, no code fence."})
    raise last_err
