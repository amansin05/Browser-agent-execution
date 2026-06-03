"""Planner — decomposes a goal into ordered subgoals (and re-plans on escalation)."""

from browser_agent.agent.prompts import ENHANCED_PLANNER
from browser_agent.config import PLANNER_MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


# Strong commit phrases (an act goal containing one is a purchase/booking commitment). Used with the
# tier/needs_approval signal to find the TERMINAL steps a re-plan must not silently drop (fix 6).
_COMMIT_PHRASES = ("cart", "bag", "basket", "checkout", "check out", "place order",
                   "place the order", "proceed to pay", "proceed to checkout", "make payment",
                   "payment", "complete the purchase", "complete purchase", "confirm booking",
                   "confirm the order", "pay now")


def _is_commit(sg: dict) -> bool:
    """Is this subgoal a committing (purchase/booking) step? A confirm/secured-tier act, or an act
    whose goal names a strong commit phrase. Discovery steps ('find books to buy') are NOT commits —
    only act steps qualify, so 'buy'/'book' inside an explore goal don't false-positive."""
    if sg.get("type", "act") != "act":
        return False
    if sg.get("tier") in ("confirm", "secured") or sg.get("needs_approval"):
        return True
    goal = (sg.get("goal") or "").lower()
    return any(p in goal for p in _COMMIT_PHRASES)


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
    # Fix 6: a re-plan must not silently drop the purchase/booking TAIL (the bug where a 4-step
    # "…→ add to cart" plan collapsed to find→show). Capture the prior plan's committing steps so we
    # can both ASK the planner to keep them and re-append them if it drops them anyway.
    prior_terminals = [sg for sg in (prior_plan or []) if _is_commit(sg)]
    if prior_plan is None:
        user = f"{head}User goal: {goal}"
    else:
        carry = ""
        if prior_terminals:
            tail = "; ".join(t.get("goal", "") for t in prior_terminals)
            carry = ("\nThe user's goal COMMITS (a purchase/booking). The revised plan must still END "
                     f"with the committing step(s): {tail}. Preserve them as the final subgoals — "
                     "revise only the discovery/earlier steps; never collapse to just find/show.")
        user = (f"{head}User goal: {goal}\n"
                f"The plan is failing at subgoal {failed_id}.\n"
                f"Reason from the reasoner: {reason!r}\n"
                f"Observations so far: {observations or '(none)'}\n"
                f"Revise the remaining subgoals (keep finished ones conceptually done). "
                f"Same JSON schema.{carry}")
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
            result = [_normalize(sg, idx) for idx, sg in enumerate(subgoals, 1)]
            # Guard: if the re-plan dropped the purchase tail, re-append it so intent isn't lost.
            if prior_terminals and not any(_is_commit(sg) for sg in result):
                base = len(result)
                for k, t in enumerate(prior_terminals, 1):
                    result.append(_normalize({**t, "id": base + k}, base + k))
                log.info("re-plan dropped the commit tail; re-appended %d terminal subgoal(s)",
                         len(prior_terminals))
            return result
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
