"""The two-tier orchestration: per-subgoal reasoner loop + the plan -> verify -> re-plan flow.

Both `run_subgoal` and `orchestrate` operate on an already-open MCP session (the browser is
opened/closed by the caller — AgentSession), so a session can run many tasks against the same
tabs.
"""

import asyncio
import json

from browser_agent.agent.extract import extract_candidates, synthesize_options
from browser_agent.agent.grounding import web_grounding
from browser_agent.agent.planner import plan_subgoals
from browser_agent.agent.reader import readable_text
from browser_agent.agent.reasoner import reasoner_decide
from browser_agent.agent.scoring import is_near_tie, score_candidates
from browser_agent.agent.verifier import verify_success
from browser_agent.config import COMPOSITION_MODEL, MODEL, PLANNER_MODEL
from browser_agent.log import get_logger
from browser_agent.services.mcp_client import (
    full_snapshot, list_tabs_state, newest_new_tab, observe, select_tab, tool_result_to_text,
)
from browser_agent.utils.domains import domain_allowed, domain_of
from browser_agent.utils.io import maybe_await
from browser_agent.utils.text import page_looks_blocked, snapshot_is_sufficient

log = get_logger(__name__)


def _log_plan(label: str, plan: list[dict]) -> None:
    log.info("[%s] %s", label, " | ".join(f"{sg['id']}. {sg['goal']}" for sg in plan))

# A confirm/secured subgoal that gets no human answer within this window is ABORTED (never
# auto-approved) — the safe default from B5.
APPROVAL_TIMEOUT = 300.0

# Sentinel an ask_user answer carries when the user clicked "No preference — explore all".
# The UI (AskModal) submits this exact string; we then steer the re-plan to broaden, not narrow.
SKIP_SENTINEL = "__no_preference__"

# Stop the planner from pestering the user with one clarification after another (each forces a full
# re-plan). After this many answered clarifications, further ask_user subgoals are skipped and we
# explore broadly instead.
MAX_CLARIFICATIONS = 2

# A standoff guard: the reasoner declares subgoal_complete, the verifier refuses, the reasoner
# immediately re-declares — spamming until the step budget drains. After this many refused
# completions in a subgoal, hand back to the planner instead.
MAX_COMPLETE_REJECTS = 3


async def _approve_with_timeout(approve, prompt: str, timeout: float = APPROVAL_TIMEOUT) -> bool:
    """Await an approval, but treat no-answer-in-time as a denial. Sync (CLI) callbacks aren't
    timed (a human is at the terminal); async (web) ones are."""
    value = approve(prompt)
    if asyncio.iscoroutine(value):
        try:
            return bool(await asyncio.wait_for(value, timeout))
        except asyncio.TimeoutError:
            return False
    return bool(value)


async def run_subgoal(groq, session, reasoner_tools, subgoal, *, allowlist, approve, ask,
                      observations, max_steps, model=MODEL, read_content=True, emit=None) -> tuple[str, str]:
    """Drive one subgoal. Returns (status, detail) where status in
    {'complete','escalate','exhausted','denied'}. `emit(event_dict)` (optional) streams progress
    to a UI; prints are kept for the CLI. When `read_content` is set and a page's snapshot is too
    sparse to reason over, the content extractor (Readability.js -> trafilatura) turns the live page
    into clean text appended to the observation — this replaced the old screenshot/vision fallback."""
    emit = emit or (lambda e: None)
    recent_actions: list[str] = []
    last_sig = None
    repeat = 0
    last_action = None
    name_repeat = 0
    complete_rejects = 0
    loop_nudges = 0
    working_tab = None  # the tab the agent drives; we keep focus pinned here across new-tab popups

    for step in range(1, max_steps + 1):
        # Keep the agent and the user on the SAME tab. `working_tab` follows the tab a click opened
        # (set after the action below). Before observing, make sure that tab is the active one — so
        # the snapshot (and any in-page prompt overlay) target the tab you're looking at, not the
        # opener. The first step establishes the working tab.
        cur, idxs = await list_tabs_state(session)
        if working_tab is None:
            working_tab = cur
        elif cur is not None and cur != working_tab and working_tab in idxs:
            await select_tab(session, working_tab)

        snapshot_text, url = await observe(session)
        if read_content and not snapshot_is_sufficient(snapshot_text):
            # The accessibility snapshot is too sparse to reason over (JS-heavy app, canvas, opaque
            # content blob). Turn the LIVE page into clean text and append it to the observation so
            # the reasoner has something to work with — instead of a screenshot (vision removed).
            extra = await readable_text(session)
            if extra:
                log.debug("step %d: DOM snapshot sparse — appended %d chars of extracted content",
                          step, len(extra))
                emit({"type": "extract", "step": step, "chars": len(extra)})
                snapshot_text = f"{snapshot_text}\n\n### Extracted readable content\n{extra[:6000]}"
        thought, action, args = await reasoner_decide(
            groq, reasoner_tools, subgoal, snapshot_text, recent_actions, model=model)
        log.info("step %d: %s(%s) — %s", step, action, json.dumps(args)[:120], thought[:100])
        emit({"type": "step", "step": step, "thought": thought, "action": action, "args": args})

        # --- synthetic control actions ---
        if action == "subgoal_complete":
            ok, reason = await verify_success(groq, snapshot_text, subgoal["success_condition"], model=model)
            log.info("verifier: satisfied=%s (%s)", ok, reason[:80])
            emit({"type": "verifier", "satisfied": ok, "reason": reason})
            if ok:
                return "complete", reason
            complete_rejects += 1
            if complete_rejects >= MAX_COMPLETE_REJECTS:
                # Standoff: stop letting the reasoner re-declare done forever. Hand back to the
                # planner — for an explore subgoal, orchestrate still salvages whatever is on the page.
                return "escalate", f"verifier refused completion {complete_rejects}x: {reason}"
            recent_actions.append(f"subgoal_complete -> REJECTED by verifier: {reason}")
            continue
        if action == "escalate":
            return "escalate", args.get("reason", "escalated")
        if action == "ask_human":
            answer = await maybe_await(ask(args.get("question", "(no question)")))
            recent_actions.append(f"ask_human -> {answer!r}")
            emit({"type": "ask_answer", "answer": answer})
            continue

        # --- real browser action: enforce domain allowlist ---
        target_url = args.get("url", url)
        if not domain_allowed(target_url, allowlist):
            msg = f"blocked by allowlist: {domain_of(target_url)!r} not in {sorted(allowlist)}"
            log.warning("%s", msg)
            recent_actions.append(f"{action} -> {msg}")
            emit({"type": "action_result", "action": action, "blocked": True, "outcome": msg})
            continue

        # --- loop detection ---
        # Tight: the EXACT same (action, args) three turns running. Loose: the same action NAME
        # four turns running even as args wobble (e.g. retrying the same failing click/wait with a
        # tweaked target/text) — the tight check alone missed this and let the budget drain.
        sig = (action, json.dumps(args, sort_keys=True))
        repeat = repeat + 1 if sig == last_sig else 0
        last_sig = sig
        name_repeat = name_repeat + 1 if action == last_action else 0
        last_action = action
        if repeat >= 2 or name_repeat >= 3:
            # First loop in this subgoal: don't throw the whole explore away — NUDGE the reasoner to
            # change tactics (go straight to a listings URL / use search) and give it another go. On
            # retail homepages the reasoner often gets stuck re-clicking a menu before it ever reaches
            # the product grid; cutting out immediately wasted the attempt (the "buy a phone" run).
            # Escalate only on a SECOND loop, so a genuinely stuck agent still hands back.
            if loop_nudges < 1:
                loop_nudges += 1
                recent_actions.append(
                    f"LOOP: you repeated {action} with no progress. STOP repeating it. Reach the "
                    f"content a DIFFERENT way — navigate DIRECTLY to a category or search-results URL "
                    f"(e.g. /search?q=… or a category path), or type your query into the page's search "
                    f"box and submit. Do not click the same element again.")
                repeat = name_repeat = 0
                last_sig = last_action = None
                emit({"type": "loop_nudge", "step": step, "action": action})
                continue
            return "escalate", f"loop detected: repeated {action}"

        # --- execute via MCP ---
        try:
            before_idxs = idxs
            result = await session.call_tool(action, args)
            rtext = tool_result_to_text(result)
            # If the action opened a new tab (e.g. a result that opens in its own tab), FOLLOW it:
            # adopt it as the working tab and make it active, so the agent works on the same tab the
            # user is now looking at instead of being stranded on the opener.
            cur2, idxs2 = await list_tabs_state(session)
            adopt = newest_new_tab(before_idxs, idxs2)
            if adopt is not None:
                working_tab = adopt
                if cur2 != adopt:
                    await select_tab(session, adopt)
                rtext += "\n(note: a new tab opened — followed it)"
        except Exception as e:
            rtext = f"ERROR calling {action}: {e!r}"
            log.warning("step %d: %s raised: %r", step, action, e)
        outcome = rtext[:150].replace("\n", " ")
        log.debug("step %d result: %s", step, outcome)
        recent_actions.append(f"{action}({json.dumps(args)[:60]}) -> {outcome}")
        emit({"type": "action_result", "action": action, "outcome": outcome})
        # Cross-subgoal memory: keep evaluate/extract-style readouts.
        if action == "browser_evaluate":
            observations.append(rtext[:300])

    return "exhausted", "step budget exhausted"


async def orchestrate(groq, session, reasoner_tools, capabilities, goal, *, allowlist,
                      approve, ask, max_steps, max_replans, read_content, ground=True, emit=None,
                      preamble="", observations=None, model=MODEL) -> str:
    """Plan the goal into subgoals, run each as a verified reasoner loop, re-plan on escalation.
    `preamble` (session memory) is fed to the planner; `observations` (if provided) accumulates
    cross-subgoal readouts so the caller can persist them to the scratchpad. When `ground` is set,
    a quick live web search of the goal runs FIRST and its top results are prepended to the planner
    preamble, so the planner anchors on real, goal-appropriate sources (see agent/grounding.py)."""
    emit = emit or (lambda e: None)
    observations = observations if observations is not None else []

    working_goal = goal  # grows as the user clarifies via ask_user, so the rest re-plans around it

    # Ground the planner in real search results BEFORE planning. Computed once and reused across all
    # plan/re-plan calls (we don't re-search on every re-plan). Best-effort: "" if it can't run.
    grounding = await web_grounding(session, working_goal) if ground else ""
    if grounding:
        emit({"type": "grounding", "text": grounding})
    plan_preamble = f"{grounding}\n\n{preamble}".strip() if grounding else preamble

    plan = await plan_subgoals(groq, working_goal, capabilities, preamble=plan_preamble, model=PLANNER_MODEL)
    _log_plan("plan", plan)
    emit({"type": "plan", "subgoals": plan})

    state: dict = {"candidates": [], "selected": None, "answers": [], "failed_sources": []}
    replans = 0
    clarifications = 0
    i = 0
    while i < len(plan):
        sg = plan[i]
        sg_type = sg.get("type", "act")
        tier = sg.get("tier", "auto")
        log.info("subgoal %s (%s/%s): %s", sg["id"], sg_type, tier, sg["goal"])
        emit({"type": "subgoal_start", "id": sg["id"], "goal": sg["goal"], "kind": sg_type,
              "tier": tier, "success_condition": sg["success_condition"],
              "needs_approval": sg["needs_approval"]})

        # --- HITL: confirm/secured subgoals pause for approval BEFORE any side effect ---
        if tier in ("confirm", "secured") or sg["needs_approval"]:
            prompt = f"[{tier}] Subgoal {sg['id']}: {sg['goal']}"
            if not await _approve_with_timeout(approve, prompt):
                emit({"type": "subgoal_end", "id": sg["id"], "status": "denied", "detail": "declined/timeout"})
                return "Stopped: user declined approval."

        # --- ask_user: disambiguate (single/multi choice + "other"), then re-plan around the
        #     answer so the choice actually steers the remaining work ---
        if sg_type == "ask_user":
            question = sg.get("question", sg["goal"])
            # Cap clarifications: a weak planner tends to ask one preference at a time, re-planning
            # after each. Once we've asked enough, stop asking — skip this subgoal and let the rest
            # of the plan (explore/rank/present) run, exploring broadly.
            if clarifications >= MAX_CLARIFICATIONS:
                log.info("subgoal %s ask_user skipped — clarification cap (%d) reached; exploring broadly",
                         sg["id"], MAX_CLARIFICATIONS)
                observations.append(f"skipped further clarification '{question}' (cap reached)")
                emit({"type": "subgoal_end", "id": sg["id"], "status": "skipped",
                      "detail": "clarification cap reached"})
                i += 1
                continue
            clarifications += 1
            answer = await maybe_await(ask(question, options=sg.get("options"),
                                           allow_free_text=sg.get("allow_free_text", True),
                                           multi_select=sg.get("multi_select", False)))
            # "No preference / skip": don't narrow — tell the planner to explore broadly instead.
            if (answer or "").strip() in (SKIP_SENTINEL, ""):
                answer = "(no preference — explore all options)"
                clarification = (f"(User has NO preference about — {question}. Do NOT narrow on "
                                 f"this; explore broadly across all reasonable options and compare them.)")
            else:
                clarification = f"(User clarified — {question}: {answer})"
            state["answers"].append(answer)
            observations.append(f"user answered '{question}': {answer}")
            emit({"type": "subgoal_end", "id": sg["id"], "status": "answered", "detail": str(answer)})
            working_goal = f"{working_goal}\n{clarification}"
            plan = await plan_subgoals(groq, working_goal, capabilities, preamble=plan_preamble,
                                       observations="; ".join(observations[-5:]), model=PLANNER_MODEL)
            _log_plan(f"plan re-shaped around: {answer}", plan)
            emit({"type": "replan", "n": replans, "reason": f"clarified: {answer}", "subgoals": plan})
            i = 0
            continue

        # --- exploit: deterministic scoring over gathered candidates (no LLM) ---
        if sg_type == "exploit":
            cands = state["candidates"]
            weights = (sg.get("scoring") or {}).get("weights") or {}
            ranked = score_candidates(cands, weights)
            state["selected"] = ranked[0] if ranked else None
            top3 = ranked[:3]
            detail = (f"selected {state['selected']}" if state["selected"] else "no candidates to score")
            log.info("subgoal %s exploit -> %s", sg["id"], detail)
            emit({"type": "exploit", "id": sg["id"], "selected": state["selected"],
                  "top": top3, "near_tie": is_near_tie(ranked)})
            emit({"type": "subgoal_end", "id": sg["id"], "status": "complete", "detail": detail})
            observations.append(detail)
            i += 1
            continue

        # --- present: synthesize gathered candidates into a structured markdown shortlist (no LLM
        #     browsing). Its markdown becomes the run's final answer shown to the user. ---
        if sg_type == "present":
            md = await synthesize_options(groq, working_goal, state["candidates"],
                                          state.get("selected"), model=COMPOSITION_MODEL)
            state["summary"] = md
            log.info("subgoal %s present -> %d options synthesized", sg["id"], len(state["candidates"]))
            emit({"type": "present", "id": sg["id"], "markdown": md})
            emit({"type": "subgoal_end", "id": sg["id"], "status": "complete",
                  "detail": f"presented {len(state['candidates'])} options"})
            observations.append("presented a shortlist of options to the user")
            i += 1
            continue

        # --- explore: reasoner gathers, then we extract a structured candidate list ---
        # Explore is the heaviest subgoal kind — it has to search, route to one or more sources,
        # and read enough to gather candidates — so give it more headroom than an act subgoal.
        sg_steps = max_steps + 6 if sg_type == "explore" else max_steps
        status, detail = await run_subgoal(
            groq, session, reasoner_tools, sg, allowlist=allowlist, approve=approve, ask=ask,
            observations=observations, max_steps=sg_steps, model=model, read_content=read_content, emit=emit)

        if sg_type == "explore" and status != "denied":
            # Extract whatever listings are on the current page — even when the reasoner escalated
            # or exhausted. In practice it almost always still landed on a results page (it just
            # never called subgoal_complete), so SALVAGING those candidates beats throwing the whole
            # attempt away and re-planning from scratch (the failure mode that exhausted the budget
            # while real listings sat on the page).
            # Use the FULL (untruncated) snapshot for extraction: products sit past the 12k cap on
            # heavy retail pages, so observe()'s capped text yields 0 candidates (Amazon failure).
            snapshot_text, page_url = await full_snapshot(session)
            cands = await extract_candidates(groq, snapshot_text, sg.get("explore_spec"), model=model)
            if not cands and read_content and not page_looks_blocked(snapshot_text):
                # Heavy retail SPAs (Amazon/Flipkart/Reliance) render a product GRID that the a11y
                # snapshot captures poorly — the reasoner reaches the listings page but extraction
                # reads 0 from the noisy/truncated tree, so every source "fails" and the re-plan
                # budget drains (the "i want to buy a phone" run). Give the content extractor
                # (Readability -> trafilatura) a second pass over the live page text and re-extract.
                extra = await readable_text(session)
                if extra:
                    cands = await extract_candidates(groq, extra, sg.get("explore_spec"), model=model)
                    if cands:
                        log.info("subgoal %s explore -> %d candidates salvaged via content extractor",
                                 sg["id"], len(cands))
            if cands:
                state["candidates"].extend(cands)  # accumulate across multiple explore subgoals
                salvaged = " [salvaged]" if status != "complete" else ""
                log.info("subgoal %s explore -> +%d candidates (%d total)%s",
                         sg["id"], len(cands), len(state["candidates"]), salvaged)
                emit({"type": "candidates", "id": sg["id"], "count": len(cands),
                      "total": len(state["candidates"]), "candidates": cands[:10]})
                observations.append(f"gathered {len(cands)} candidates ({len(state['candidates'])} total)")
                status, detail = "complete", f"gathered {len(cands)} candidates"  # proceed, don't re-plan
            else:
                # Nothing extracted here. Work out WHY so the re-plan can steer to a DIFFERENT
                # source instead of retrying the same dead end: record this source as failed and,
                # if it is a bot-check / captcha wall, say so. (A blocked or empty page often still
                # has refs, so it looked "sufficient" to the reasoner — but there is nothing to read.)
                src = domain_of(page_url)
                blocked = page_looks_blocked(snapshot_text)
                if src and src not in state["failed_sources"]:
                    state["failed_sources"].append(src)
                wall = f"looks like a {blocked}" if blocked else "no listings on the page"
                avoid = (f" Already tried (avoid these): {', '.join(state['failed_sources'])}."
                         if state["failed_sources"] else "")
                log.info("subgoal %s explore -> 0 candidates from %s%s",
                         sg["id"], src or "page", f" [{blocked}]" if blocked else "")
                emit({"type": "candidates", "id": sg["id"], "count": 0,
                      "total": len(state["candidates"]), "blocked": bool(blocked), "source": src})
                if state["candidates"]:
                    # Earlier explores already gathered options — one dead source is not fatal.
                    # Proceed to rank/present what we have instead of burning re-plans.
                    status, detail = "complete", (
                        f"no new candidates from {src or 'this page'} ({wall}); proceeding with "
                        f"{len(state['candidates'])} already gathered")
                else:
                    # No candidates anywhere yet — force a re-plan that routes to another source.
                    status, detail = "escalate", (
                        f"source {src or 'page'!r} yielded no candidates ({wall}). Route the explore "
                        f"step to a DIFFERENT retailer/source.{avoid}")
                observations.append(detail)

        log.info("subgoal %s -> %s: %s", sg["id"], status, detail)
        emit({"type": "subgoal_end", "id": sg["id"], "status": status, "detail": detail})

        if status == "complete":
            i += 1
            continue
        if status == "denied":
            return "Stopped: user declined."

        if replans >= max_replans:
            return f"Failed: exhausted re-plan budget at subgoal {sg['id']} ({detail})."
        replans += 1
        log.info("replan %d/%d: %s", replans, max_replans, detail)
        plan = await plan_subgoals(groq, working_goal, capabilities, prior_plan=plan,
                                   failed_id=sg["id"], reason=detail,
                                   observations="; ".join(observations[-5:]),
                                   preamble=preamble, model=PLANNER_MODEL)
        _log_plan("plan revised", plan)
        emit({"type": "replan", "n": replans, "reason": detail, "subgoals": plan})
        i = 0

    # A present subgoal's markdown is the user-facing answer; otherwise a plain completion.
    return state.get("summary") or "Done."
