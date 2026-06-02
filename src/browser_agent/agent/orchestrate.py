"""The two-tier orchestration: per-subgoal reasoner loop + the plan -> verify -> re-plan flow.

Both `run_subgoal` and `orchestrate` operate on an already-open MCP session (the browser is
opened/closed by the caller — AgentSession), so a session can run many tasks against the same
tabs.
"""

import asyncio
import json

from browser_agent.agent.dom_index import execute_action, index_dom, is_terminating
from browser_agent.agent.extract import extract_candidates, synthesize_options
from browser_agent.agent.gather import gather_parallel
from browser_agent.agent.grounding import web_grounding
from browser_agent.agent.planner import plan_subgoals
from browser_agent.agent.reader import readable_text
from browser_agent.agent.reasoner import reasoner_decide
from browser_agent.agent.scoring import is_near_tie, score_candidates
from browser_agent.agent.verifier import verify_success
from browser_agent.config import COMPOSITION_MODEL, MODEL, PLANNER_MODEL
from browser_agent.log import get_logger
from browser_agent.services.mcp_client import (
    full_snapshot, list_tabs_state, newest_new_tab, observe, select_tab,
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

# Below this many interactive elements, the indexed-DOM view alone is too thin to reason over (a
# content blob, an article, a canvas) — append the cleaned page text (Readability -> trafilatura ->
# innerText) so the reasoner has something to read, the same salvage the a11y path used.
MIN_INTERACTIVE_FOR_TEXT = 3

# The reasoner may emit several actions in one turn; execute at most this many (the page-change
# guard usually stops a batch sooner). Mirrors browser-use's max_actions_per_step.
MAX_ACTIONS_PER_STEP = 5

# Page-stagnation guard: if the page fingerprint is unchanged this many steps running DESPITE the
# agent taking actions, those actions are having no effect (dead element, JS not firing) — nudge
# once, then escalate. Complements loop detection (repeated identical actions); this catches
# DIFFERENT actions that still change nothing. Mirrors browser-use's consecutive_stagnant_pages.
STAGNATION_LIMIT = 3

# Once the subgoal has burned this fraction of its step budget, warn the reasoner to land it or
# escalate rather than fritter the tail away (browser-use's 75% budget nudge).
BUDGET_WARN_FRACTION = 0.75


def _page_fingerprint(dom, url: str, snapshot_text: str):
    """A cheap identity of the current page used to detect stagnation. From the indexed view when we
    have one (url + scroll position + the set of interactive elements), else the a11y snapshot."""
    if dom is not None and dom.elements:
        return ("dom", dom.url, dom.scroll_y, tuple(sorted(str(e.key()) for e in dom.elements)))
    return ("a11y", url, hash(snapshot_text))


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
                      observations, max_steps, model=MODEL, read_content=True, emit=None,
                      plan_context="") -> tuple[str, str]:
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
    previous_keys: set = set()  # interactive-element keys from the last step -> mark NEW ones (*)
    dom = None                  # the current indexed-DOM view (None when we fell back to a11y)
    last_thought = ""           # the reasoner's note from the prior turn, echoed back for continuity
    last_fp = None              # page fingerprint from the prior step (stagnation detection)
    stagnant = 0                # consecutive steps the page hasn't changed despite acting
    stagnation_nudges = 0
    budget_warned = False
    acted = False               # did the PRIOR step execute a page action? (gates stagnation)

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

        # PERCEPTION: build the indexed-DOM view (numbered interactive elements + NEW marks) as the
        # reasoner's primary observation — stable addresses instead of a11y refs. Fall back to the
        # accessibility snapshot only if the in-page build fails or finds nothing to act on.
        dom = await index_dom(session)
        if dom is not None and dom.elements:
            snapshot_text = dom.render(previous_keys)
            url = dom.url or url
            previous_keys = dom.keys
        else:
            snapshot_text, url = await observe(session)
            previous_keys = set()

        # Append cleaned page text when there's little to act on (a content page / opaque app): the
        # reasoner needs something to read. Gate on the indexed view when we have one, else on the
        # legacy a11y sufficiency check — so a normal interactive page never triggers it.
        if read_content and (
                (dom is not None and len(dom.elements) < MIN_INTERACTIVE_FOR_TEXT)
                or (dom is None or not dom.elements) and not snapshot_is_sufficient(snapshot_text)):
            extra = await readable_text(session)
            if extra:
                log.debug("step %d: page thin to act on — appended %d chars of extracted content",
                          step, len(extra))
                emit({"type": "extract", "step": step, "chars": len(extra)})
                snapshot_text = f"{snapshot_text}\n\n### Extracted readable content\n{extra[:6000]}"

        # --- stagnation guard: did the page change since last step despite us acting? ---
        # `acted` still holds whether the PRIOR step ran a page action; use it before resetting.
        fp = _page_fingerprint(dom, url, snapshot_text)
        if fp != last_fp:
            stagnant = 0
        elif step > 1 and acted:
            stagnant += 1
        last_fp = fp
        if stagnant >= STAGNATION_LIMIT:
            if stagnation_nudges < 1:
                stagnation_nudges += 1
                stagnant = 0
                recent_actions.append(
                    "STAGNANT: the page has NOT changed despite your last few actions — they are "
                    "having no effect. Do NOT repeat them. Try a DIFFERENT element, navigate "
                    "directly to a URL, or scroll; if you cannot make progress, escalate.")
                emit({"type": "stagnation_nudge", "step": step})
            else:
                return "escalate", "page stagnant: actions had no visible effect"

        # --- step-budget warning: near the end, push the reasoner to land it or escalate ---
        if not budget_warned and step >= max(2, int(max_steps * BUDGET_WARN_FRACTION)):
            budget_warned = True
            recent_actions.append(
                f"BUDGET: you have used {step}/{max_steps} steps. If the success condition is "
                f"satisfied, call subgoal_complete now; otherwise make your single most decisive "
                f"move (navigate directly to the target), or escalate if you are stuck.")
            emit({"type": "budget_warning", "step": step, "max_steps": max_steps})

        acted = False  # reset for THIS step; set True below only if a page action runs
        thought, actions = await reasoner_decide(
            groq, reasoner_tools, subgoal, snapshot_text, recent_actions, last_thought,
            plan_context=plan_context, model=model)
        last_thought = thought
        actions = actions[:MAX_ACTIONS_PER_STEP]
        if not actions:                       # reasoner_decide always returns at least one
            return "escalate", "model returned no action"
        first_action = actions[0][0]
        log.info("step %d: %s%s — %s", step, first_action,
                 f" (+{len(actions) - 1} more)" if len(actions) > 1 else "", thought[:100])
        emit({"type": "step", "step": step, "thought": thought,
              "action": first_action, "args": actions[0][1],
              "actions": [{"action": n, "args": a} for n, a in actions]})

        # --- a LEADING control action is the whole decision (any trailing actions are ignored) ---
        # Handled before loop detection so a legitimately repeated complete/ask isn't mistaken for a
        # navigation loop (the complete-reject standoff guard handles repeated completes instead).
        if first_action in ("subgoal_complete", "escalate", "ask_human"):
            _, args = actions[0]
            if first_action == "escalate":
                return "escalate", args.get("reason", "escalated")
            if first_action == "ask_human":
                answer = await maybe_await(ask(args.get("question", "(no question)")))
                recent_actions.append(f"ask_human -> {answer!r}")
                emit({"type": "ask_answer", "answer": answer})
                continue
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

        # --- loop detection on the action BATCH (the leading action drives the looser name check) ---
        # Tight: the EXACT same batch two turns running. Loose: the same leading action NAME three
        # turns running even as args wobble (retrying the same failing click with a tweaked target).
        sig = json.dumps([[n, a] for n, a in actions], sort_keys=True)
        repeat = repeat + 1 if sig == last_sig else 0
        last_sig = sig
        name_repeat = name_repeat + 1 if first_action == last_action else 0
        last_action = first_action
        if repeat >= 2 or name_repeat >= 3:
            # First loop in this subgoal: don't throw the whole explore away — NUDGE the reasoner to
            # change tactics (go straight to a listings URL / use search) and give it another go. On
            # retail homepages the reasoner often gets stuck re-clicking a menu before it ever reaches
            # the product grid; cutting out immediately wasted the attempt (the "buy a phone" run).
            # Escalate only on a SECOND loop, so a genuinely stuck agent still hands back.
            if loop_nudges < 1:
                loop_nudges += 1
                recent_actions.append(
                    f"LOOP: you repeated {first_action} with no progress. STOP repeating it. Reach the "
                    f"content a DIFFERENT way — navigate DIRECTLY to a category or search-results URL "
                    f"(e.g. /search?q=… or a category path), or type your query into the page's search "
                    f"box and submit. Do not repeat the same action.")
                repeat = name_repeat = 0
                last_sig = last_action = None
                emit({"type": "loop_nudge", "step": step, "action": first_action})
                continue
            return "escalate", f"loop detected: repeated {first_action}"

        # --- execute the batch in order, behind the page-change guard (see dom_index) ---
        before_idxs = idxs
        for action, args in actions:
            # A control action reached mid-batch (e.g. [scroll, subgoal_complete]) ends the turn.
            if action == "escalate":
                return "escalate", args.get("reason", "escalated")
            if action == "ask_human":
                answer = await maybe_await(ask(args.get("question", "(no question)")))
                recent_actions.append(f"ask_human -> {answer!r}")
                emit({"type": "ask_answer", "answer": answer})
                break
            if action == "subgoal_complete":
                ok, reason = await verify_success(groq, snapshot_text, subgoal["success_condition"], model=model)
                emit({"type": "verifier", "satisfied": ok, "reason": reason})
                if ok:
                    return "complete", reason
                complete_rejects += 1
                if complete_rejects >= MAX_COMPLETE_REJECTS:
                    return "escalate", f"verifier refused completion {complete_rejects}x: {reason}"
                recent_actions.append(f"subgoal_complete -> REJECTED by verifier: {reason}")
                break

            # enforce the domain allowlist (only navigate / a link click can change origin)
            target_url = None
            if action == "navigate":
                target_url = args.get("url", "")
            elif action == "click_element" and dom is not None:
                try:
                    el = dom.selector_map.get(int(args.get("index")))
                except (TypeError, ValueError):
                    el = None
                if el and el.href.startswith(("http://", "https://")):
                    target_url = el.href
            if target_url and not domain_allowed(target_url, allowlist):
                msg = f"blocked by allowlist: {domain_of(target_url)!r} not in {sorted(allowlist)}"
                log.warning("%s", msg)
                recent_actions.append(f"{action} -> {msg}")
                emit({"type": "action_result", "action": action, "blocked": True, "outcome": msg})
                break  # a blocked navigation ends the batch

            res = await execute_action(session, action, args)
            acted = True  # a page action ran -> next step's stagnation check is meaningful
            rtext = res["outcome"]
            # FOLLOW a new tab the action opened (adopt + activate) so the agent works on the tab the
            # user is now looking at instead of being stranded on the opener.
            cur2, idxs2 = await list_tabs_state(session)
            adopt = newest_new_tab(before_idxs, idxs2)
            new_tab_opened = adopt is not None
            if new_tab_opened:
                working_tab = adopt
                if cur2 != adopt:
                    await select_tab(session, adopt)
                rtext += " (note: a new tab opened — followed it)"
            before_idxs = idxs2
            outcome = rtext[:150].replace("\n", " ")
            log.debug("step %d result: %s(%s) -> %s", step, action, json.dumps(args)[:60], outcome)
            recent_actions.append(f"{action}({json.dumps(args)[:60]}) -> {outcome}")
            emit({"type": "action_result", "action": action, "outcome": outcome, "ok": res["ok"]})

            # PAGE GUARD: stop after any action that may have changed the page (navigation, click,
            # submit) or that opened a new tab — re-perceive next step rather than act on a stale view.
            if new_tab_opened or is_terminating(action, args):
                break

    return "exhausted", "step budget exhausted"


async def orchestrate(groq, session, reasoner_tools, capabilities, goal, *, allowlist,
                      approve, ask, max_steps, max_replans, read_content, ground=True, emit=None,
                      preamble="", observations=None, model=MODEL,
                      parallel=False, browser="chrome") -> str:
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

        # --- PARALLEL EXPLORE: dispatch a run of consecutive explore subgoals concurrently, each on
        #     its own headless session, instead of one-at-a-time on the live browser (agent/gather).
        #     gather_parallel emits its own per-source subgoal_start/candidates/subgoal_end, so we
        #     skip the normal single-subgoal emit + loop below for these.
        if parallel and sg_type == "explore":
            run = [sg]
            j = i + 1
            while j < len(plan) and plan[j].get("type", "act") == "explore":
                run.append(plan[j])
                j += 1
            if len(run) >= 2:
                log.info("dispatching %d explore subgoals in parallel", len(run))
                results = await gather_parallel(
                    groq, reasoner_tools, run, allowlist=allowlist, max_steps=max_steps + 6,
                    model=model, read_content=read_content, browser=browser, emit=emit)
                for sg2 in run:
                    cands = results.get(sg2.get("id"), [])
                    if cands:
                        state["candidates"].extend(cands)
                        observations.append(
                            f"gathered {len(cands)} candidates from source (parallel)")
                log.info("parallel explore -> %d candidates total", len(state["candidates"]))
                i = j
                continue
            # a lone explore -> fall through to the normal sequential path below

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
        prior_goals = [p["goal"] for p in plan[:i]]
        plan_context = (f"Subgoal {i + 1} of {len(plan)} in the overall plan."
                        + (f" Earlier subgoals already handled: {'; '.join(prior_goals[-4:])}."
                           if prior_goals else ""))
        status, detail = await run_subgoal(
            groq, session, reasoner_tools, sg, allowlist=allowlist, approve=approve, ask=ask,
            observations=observations, max_steps=sg_steps, model=model, read_content=read_content,
            emit=emit, plan_context=plan_context)

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
