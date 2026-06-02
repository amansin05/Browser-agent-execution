r"""
ORCHESTRATOR eval set — 50 deterministic control-flow scenarios (no network, no LLM).

orchestrate() is the state machine that drives plan -> per-subgoal run -> verify -> re-plan, with
explore salvage, candidate accumulation, deterministic exploit scoring, present synthesis,
clarification capping, HITL approval gating, blocked-source steering, and web-grounding injection.
We exercise each branch by faking its collaborators (plan_subgoals / run_subgoal / observe /
extract_candidates / synthesize_options / web_grounding) and asserting the result + emitted events.
The exploit scorer (agent/scoring.py) runs FOR REAL — those scenarios check real ranking.

Run:
    .\.venv\Scripts\python.exe scripts\eval_orchestrator.py
"""

import asyncio
import json
import pathlib
import sys

import browser_agent.agent.orchestrate as orch

OUT = pathlib.Path(__file__).parent / "eval_results" / "orchestrator.json"


# ----------------------------------------------------------------- subgoal builders
def act(i=1, goal="do x", tier="auto", needs=False, sc="done"):
    return {"id": i, "type": "act", "goal": goal, "success_condition": sc, "tier": tier, "needs_approval": needs}


def explore(i=1, sources=("amazon",), goal="gather options", sc="a list is visible"):
    return {"id": i, "type": "explore", "goal": goal, "success_condition": sc, "tier": "auto",
            "needs_approval": False, "explore_spec": {"sources": list(sources), "target_count": 5}}


def present(i=2):
    return {"id": i, "type": "present", "goal": "show shortlist", "success_condition": "shown",
            "tier": "auto", "needs_approval": False}


def exploit(i=2, weights=None):
    return {"id": i, "type": "exploit", "goal": "rank", "success_condition": "ranked", "tier": "auto",
            "needs_approval": False, "scoring": {"weights": weights or {}}}


def ask_user(i=1, q="What's your budget?", opts=None, multi=False):
    return {"id": i, "type": "ask_user", "goal": q, "question": q, "success_condition": "",
            "tier": "auto", "needs_approval": False, "options": opts or [], "allow_free_text": True,
            "multi_select": multi}


# ----------------------------------------------------------------- fake factories
def make_plan(plans, cap):
    # `plans` is either a single plan (list of subgoal dicts) or a list of plans (list of lists).
    # A plan whose first element is a dict is a single plan (repeated for every re-plan call);
    # otherwise it's a sequence of plans consumed one per call.
    seq = [plans] if (plans and isinstance(plans[0], dict)) else list(plans)

    async def fake(groq, goal, caps, *, prior_plan=None, failed_id=None, reason=None,
                   observations=None, preamble="", model=None):
        cap["plan_calls"].append({"goal": goal, "preamble": preamble, "failed_id": failed_id, "reason": reason})
        return seq[0] if len(seq) == 1 else seq.pop(0)
    return fake


def make_run(spec, cap):
    if callable(spec):
        async def fake(groq, session, tools, sg, **kw):
            cap["run_calls"].append(sg); return spec(sg)
        return fake
    seq = list(spec)

    async def fake(groq, session, tools, sg, **kw):
        cap["run_calls"].append(sg)
        return seq[0] if len(seq) == 1 else seq.pop(0)
    return fake


def make_observe(spec):
    async def fake(session):
        return spec if isinstance(spec, tuple) else ("snap", "http://x/")
    return fake


def make_extract(spec):
    if callable(spec):
        async def fake(groq, snap, sp, *, model=None):
            return spec(sp)
        return fake
    seq = list(spec) if isinstance(spec, list) and spec and isinstance(spec[0], list) else None

    async def fake(groq, snap, sp, *, model=None):
        if seq is not None:
            return seq[0] if len(seq) == 1 else seq.pop(0)
        return spec
    return fake


def make_synth(text, cap):
    async def fake(groq, goal, cands, selected=None, *, model=None):
        cap["synth_cands"] = cands; cap["synth_selected"] = selected
        return text
    return fake


def make_grounding(text, cap):
    async def fake(session, goal, **kw):
        cap["grounding_called"] = True
        return text
    return fake


def make_readable(text, cap):
    async def fake(session):
        cap["readable_called"] = True
        return text
    return fake


def evt_types(events):
    return [e["type"] for e in events]


def find(events, t):
    return [e for e in events if e["type"] == t]


# ----------------------------------------------------------------- scenarios
# Each scenario: name + the fakes it needs + a check(result, events, cap) -> (ok, note).
CANDS_PR = [{"name": "A", "price": 100, "rating": 4.0, "source": "x"},
            {"name": "B", "price": 50, "rating": 4.8, "source": "y"},
            {"name": "C", "price": 200, "rating": 4.5, "source": "z"}]


def S(name, **kw):
    kw["name"] = name
    return kw


SCENARIOS = [
    # ---------- basic completion ----------
    S("single act completes", plans=[act(1)], run=[("complete", "ok")],
      check=lambda r, e, c: (r == "Done.", r)),
    S("three acts complete", plans=[[act(1), act(2), act(3)]], run=[("complete", "ok")],
      check=lambda r, e, c: (r == "Done." and len(c["run_calls"]) == 3, f"{r} runs={len(c['run_calls'])}")),
    S("escalate then replan completes", plans=[[act(1)], [act(1)]],
      run=[("escalate", "stuck"), ("complete", "ok")], max_replans=2,
      check=lambda r, e, c: (r == "Done." and len(find(e, "replan")) == 1, f"{r} replans={len(find(e,'replan'))}")),
    S("exhausted then replan completes", plans=[[act(1)], [act(1)]],
      run=[("exhausted", "budget"), ("complete", "ok")], max_replans=2,
      check=lambda r, e, c: (r == "Done.", r)),
    S("replan budget exhausted -> Failed", plans=[[act(1)]], run=[("escalate", "loop")], max_replans=1,
      check=lambda r, e, c: (r.startswith("Failed"), r)),
    S("max_replans=0 fails immediately", plans=[[act(1)]], run=[("escalate", "x")], max_replans=0,
      check=lambda r, e, c: (r.startswith("Failed") and len(find(e, "replan")) == 0, r)),
    S("mixed act+act long plan", plans=[[act(i, goal=f"step {i}") for i in range(1, 6)]],
      run=[("complete", "ok")], check=lambda r, e, c: (r == "Done." and len(c["run_calls"]) == 5, r)),
    S("denied status -> stopped", plans=[[act(1)]], run=[("denied", "user")],
      check=lambda r, e, c: (r == "Stopped: user declined.", r)),

    # ---------- explore + present ----------
    S("explore complete + present", plans=[[explore(1), present(2)]], run=[("complete", "ok")],
      extract=[[{"name": "P", "source": "amazon"}]], synth="## Shortlist",
      check=lambda r, e, c: (r == "## Shortlist" and c.get("synth_cands"), r)),
    S("explore escalates but salvaged", plans=[[explore(1), present(2)]], run=[("escalate", "loop")],
      extract=[[{"name": "P", "source": "amazon"}]], synth="## Salvaged",
      check=lambda r, e, c: (r == "## Salvaged", r)),
    S("explore exhausted but salvaged", plans=[[explore(1), present(2)]], run=[("exhausted", "budget")],
      extract=[[{"name": "P"}]], synth="## OK", check=lambda r, e, c: (r == "## OK", r)),
    S("explore complete no present -> Done", plans=[[explore(1)]], run=[("complete", "ok")],
      extract=[[{"name": "P"}]], check=lambda r, e, c: (r == "Done.", r)),
    S("explore nothing, none anywhere -> replan steer", plans=[[explore(1, sources=("amazon",))], [explore(1)]],
      run=[("escalate", "loop")], observe=("Enter the characters you see below", "https://www.amazon.in/s"),
      extract=[[]], max_replans=1,
      check=lambda r, e, c: (any("amazon.in" in (pc["reason"] or "") and "DIFFERENT" in (pc["reason"] or "")
                                 for pc in c["plan_calls"] if pc["failed_id"]), str([pc["reason"] for pc in c["plan_calls"]])[:80])),
    S("explore blocked names bot-check", plans=[[explore(1)], [explore(1)]], run=[("escalate", "x")],
      observe=("captcha please verify you are a human", "https://amazon.in/s"), extract=[[]], max_replans=1,
      check=lambda r, e, c: (any("bot-check" in (pc["reason"] or "") for pc in c["plan_calls"] if pc["failed_id"]),
                             "no bot-check reason")),
    S("two explores accumulate candidates", plans=[[explore(1), explore(2), present(3)]],
      run=[("complete", "ok")], extract=[[{"name": "A"}], [{"name": "B"}]], synth="## Two",
      check=lambda r, e, c: (len(c.get("synth_cands", [])) == 2, f"cands={len(c.get('synth_cands', []))}")),
    S("explore nothing but earlier candidates proceed", plans=[[explore(1), explore(2), present(3)]],
      run=[("complete", "ok")], extract=[[{"name": "A"}], []], observe=("no listings here", "http://b/"),
      synth="## Proceed", check=lambda r, e, c: (r == "## Proceed", r)),
    S("present with zero candidates honest", plans=[[present(1)]], run=[("complete", "ok")], synth="REAL",
      check=lambda r, e, c: ("couldn't gather" in r, r[:50])),

    # ---------- exploit (REAL scorer) ----------
    S("exploit picks cheapest+best", plans=[[explore(1), exploit(2, {"price": 0.5, "rating": 0.5})]],
      run=[("complete", "ok")], extract=[CANDS_PR],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"]["name"] == "B", str(find(e, "exploit")[0]["selected"]))),
    S("exploit price-only cheapest", plans=[[explore(1), exploit(2, {"price": 1.0})]],
      run=[("complete", "ok")], extract=[CANDS_PR],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"]["name"] == "B", "price-only")),
    S("exploit rating-only best", plans=[[explore(1), exploit(2, {"rating": 1.0})]],
      run=[("complete", "ok")], extract=[CANDS_PR],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"]["name"] == "B", "rating-only")),
    S("exploit empty candidates -> None", plans=[[exploit(1, {"price": 1.0})]], run=[("complete", "ok")],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"] is None, "expected None")),
    S("exploit near-tie flagged", plans=[[explore(1), exploit(2, {"x": 1.0})]], run=[("complete", "ok")],
      extract=[[{"name": "A", "x": 10}, {"name": "B", "x": 10}]],
      check=lambda r, e, c: (find(e, "exploit")[0]["near_tie"] is True, "expected near_tie")),
    S("exploit currency strings", plans=[[explore(1), exploit(2, {"price": 1.0})]], run=[("complete", "ok")],
      extract=[[{"name": "A", "price": "Rs 3,499"}, {"name": "B", "price": "Rs 2,899"}]],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"]["name"] == "B", "currency parse")),
    S("exploit single candidate", plans=[[explore(1), exploit(2, {"price": 1.0})]], run=[("complete", "ok")],
      extract=[[{"name": "Solo", "price": 99}]],
      check=lambda r, e, c: (find(e, "exploit")[0]["selected"]["name"] == "Solo", "single")),
    S("exploit then present uses selected", plans=[[explore(1), exploit(2, {"price": 1.0}), present(3)]],
      run=[("complete", "ok")], extract=[CANDS_PR], synth="## Ranked",
      check=lambda r, e, c: (r == "## Ranked" and c.get("synth_selected", {}).get("name") == "B", "selected passed")),

    # ---------- clarification (ask_user) ----------
    S("ask answered steers replan", plans=[[ask_user(1, "Budget?")], [act(1)]], run=[("complete", "ok")],
      ask="under 2000", check=lambda r, e, c: (any("under 2000" in pc["goal"] for pc in c["plan_calls"]),
                                                "answer not in goal")),
    S("ask skip sentinel broadens", plans=[[ask_user(1, "Brand?")], [act(1)]], run=[("complete", "ok")],
      ask=orch.SKIP_SENTINEL,
      check=lambda r, e, c: (any("no preference" in pc["goal"].lower() or "explore broadly" in pc["goal"].lower()
                                 for pc in c["plan_calls"]), "no broaden steer")),
    S("ask empty answer broadens", plans=[[ask_user(1, "Size?")], [act(1)]], run=[("complete", "ok")], ask="",
      check=lambda r, e, c: (any("no preference" in pc["goal"].lower() for pc in c["plan_calls"]), "empty not broadened")),
    S("clarification cap reached", plans=[[ask_user(1, "Budget?")]], run=[("complete", "ok")], ask="x",
      check=lambda r, e, c: (sum(1 for ev in e if ev.get("type") == "ask_answer") <= orch.MAX_CLARIFICATIONS
                             and r == "Done.", f"asks={sum(1 for ev in e if ev.get('type')=='ask_answer')} r={r}")),
    S("ask multi-select value", plans=[[ask_user(1, "Colours?", multi=True)], [act(1)]], run=[("complete", "ok")],
      ask="red, blue", check=lambda r, e, c: (any("red, blue" in pc["goal"] for pc in c["plan_calls"]), "multi val")),

    # ---------- HITL approval ----------
    S("confirm approved proceeds", plans=[[act(1, tier="confirm")]], run=[("complete", "ok")], approve=True,
      check=lambda r, e, c: (r == "Done.", r)),
    S("confirm denied stops", plans=[[act(1, tier="confirm")]], run=[("complete", "ok")], approve=False,
      check=lambda r, e, c: (r == "Stopped: user declined approval.", r)),
    S("secured denied stops", plans=[[act(1, tier="secured", goal="pay now")]], run=[("complete", "ok")],
      approve=False, check=lambda r, e, c: (r == "Stopped: user declined approval.", r)),
    S("secured approved proceeds", plans=[[act(1, tier="secured", goal="pay now")]], run=[("complete", "ok")],
      approve=True, check=lambda r, e, c: (r == "Done.", r)),
    S("needs_approval flag pauses", plans=[[act(1, tier="auto", needs=True)]], run=[("complete", "ok")],
      approve=False, check=lambda r, e, c: (r.startswith("Stopped"), r)),
    S("approval timeout denies", plans=[[act(1, tier="secured")]], run=[("complete", "ok")],
      approve="TIMEOUT", check=lambda r, e, c: (r.startswith("Stopped"), r)),
    S("auto step then confirm approved", plans=[[act(1), act(2, tier="confirm", goal="submit order")]],
      run=[("complete", "ok")], approve=True, check=lambda r, e, c: (r == "Done.", r)),
    S("confirm denied on second step", plans=[[act(1), act(2, tier="confirm")]], run=[("complete", "ok")],
      approve=lambda p: "Subgoal 2" not in p,
      check=lambda r, e, c: (r == "Stopped: user declined approval." and len(c["run_calls"]) == 1, r)),

    # ---------- web grounding injection ----------
    S("grounding injected into preamble", plans=[[act(1)]], run=[("complete", "ok")], ground=True,
      grounding="### Web grounding\n1. X — myntra.com",
      check=lambda r, e, c: (c.get("grounding_called") and "myntra.com" in c["plan_calls"][0]["preamble"], "no inject")),
    S("grounding off not called", plans=[[act(1)]], run=[("complete", "ok")], ground=False,
      grounding="SHOULD-NOT-RUN",
      check=lambda r, e, c: (not c.get("grounding_called") and "SHOULD-NOT-RUN" not in c["plan_calls"][0]["preamble"], "ran anyway")),
    S("grounding empty keeps preamble clean", plans=[[act(1)]], run=[("complete", "ok")], ground=True,
      grounding="", check=lambda r, e, c: (c["plan_calls"][0]["preamble"] == "", repr(c["plan_calls"][0]["preamble"][:30]))),
    S("grounding emits event", plans=[[act(1)]], run=[("complete", "ok")], ground=True,
      grounding="### Web grounding\n1. Y — ajio.com",
      check=lambda r, e, c: (len(find(e, "grounding")) == 1, "no grounding event")),

    # ---------- emitted-event contract ----------
    S("emits plan event", plans=[[act(1)]], run=[("complete", "ok")],
      check=lambda r, e, c: (len(find(e, "plan")) == 1, evt_types(e))),
    S("emits subgoal_start/end", plans=[[act(1)]], run=[("complete", "ok")],
      check=lambda r, e, c: (find(e, "subgoal_start") and find(e, "subgoal_end"), evt_types(e))),
    S("present emits present event", plans=[[explore(1), present(2)]], run=[("complete", "ok")],
      extract=[[{"name": "P"}]], synth="## P",
      check=lambda r, e, c: (len(find(e, "present")) == 1, evt_types(e))),
    S("candidates event on explore", plans=[[explore(1)]], run=[("complete", "ok")], extract=[[{"name": "P"}]],
      check=lambda r, e, c: (find(e, "candidates") and find(e, "candidates")[0]["count"] == 1, evt_types(e))),
    S("replan event carries reason", plans=[[act(1)], [act(1)]], run=[("escalate", "boom"), ("complete", "ok")],
      max_replans=2, check=lambda r, e, c: (find(e, "replan")[0]["reason"] == "boom", "reason missing")),

    # ---------- robustness ----------
    S("re-plan three times then succeed", plans=[[act(1)], [act(1)], [act(1)], [act(1)]],
      run=[("escalate", "a"), ("escalate", "b"), ("escalate", "c"), ("complete", "ok")], max_replans=3,
      check=lambda r, e, c: (r == "Done." and len(find(e, "replan")) == 3, f"{r}")),
    S("explore denied not extracted", plans=[[explore(1), present(2)]], run=[("denied", "user")],
      extract=[[{"name": "SHOULD_NOT_BE_USED"}]],
      check=lambda r, e, c: (r == "Stopped: user declined.", r)),
    S("long explore->exploit->present pipeline",
      plans=[[explore(1), explore(2), exploit(3, {"price": 1.0}), present(4)]], run=[("complete", "ok")],
      extract=[[{"name": "A", "price": 99}], [{"name": "B", "price": 50}]], synth="## Pipe",
      check=lambda r, e, c: (r == "## Pipe" and c.get("synth_selected", {}).get("name") == "B", "pipeline")),

    # ---------- content-extractor salvage on a heavy SPA (the "buy a phone" failure) ----------
    S("explore 0 on snapshot, salvaged via content extractor", read_content=True,
      plans=[[explore(1), present(2)]], run=[("complete", "ok")],
      extract=[[], [{"name": "Galaxy", "source": "flipkart"}]],   # snapshot:0, then readable:1
      readable="# Mobiles\nGalaxy M14 Rs 12999\nRedmi 13 Rs 11999", synth="## Salvaged",
      check=lambda r, e, c: (r == "## Salvaged" and c.get("readable_called"), f"{r} readable={c.get('readable_called')}")),
    S("explore 0 and readable also empty -> replan", read_content=True,
      plans=[[explore(1)], [explore(1)]], run=[("escalate", "loop")], extract=[[], []],
      readable="", observe=("just nav chrome no products", "https://shop/"), max_replans=1,
      check=lambda r, e, c: (r.startswith("Failed") and c.get("readable_called"), f"{r}")),
]


# Originals captured once, so each scenario starts from a clean slate (no fake leaks across runs).
_PATCHABLE = ["plan_subgoals", "run_subgoal", "observe", "full_snapshot", "extract_candidates",
              "synthesize_options", "web_grounding", "readable_text"]
_ORIG = {n: getattr(orch, n) for n in _PATCHABLE}
_ORIG_APPROVE_DEFAULTS = orch._approve_with_timeout.__defaults__


async def run_scenario(scn):
    for n in _PATCHABLE:                                  # reset everything to the real impl first
        setattr(orch, n, _ORIG[n])
    orch._approve_with_timeout.__defaults__ = _ORIG_APPROVE_DEFAULTS

    cap = {"plan_calls": [], "run_calls": [], "synth_cands": None, "synth_selected": None,
           "grounding_called": False, "readable_called": False}
    orch.plan_subgoals = make_plan(scn["plans"], cap)
    orch.run_subgoal = make_run(scn.get("run", [("complete", "ok")]), cap)
    orch.observe = make_observe(scn.get("observe"))
    orch.full_snapshot = make_observe(scn.get("observe"))  # explore extraction reads the full snapshot
    orch.extract_candidates = make_extract(scn.get("extract", []))
    if scn.get("synth") != "REAL":                        # "REAL" keeps the genuine synthesize_options
        orch.synthesize_options = make_synth(scn.get("synth", "## MD"), cap)
    orch.web_grounding = make_grounding(scn.get("grounding", ""), cap)
    orch.readable_text = make_readable(scn.get("readable"), cap)  # default None -> no content salvage

    approve_spec = scn.get("approve", True)
    if approve_spec == "TIMEOUT":
        # The timeout default is bound into _approve_with_timeout's signature; shrink it so the
        # never-resolving approval coroutine trips asyncio.TimeoutError -> deny, without a 5-min wait.
        orch._approve_with_timeout.__defaults__ = (0.05,)
        approve = lambda p: asyncio.sleep(5, result=True)
    elif callable(approve_spec):
        approve = approve_spec
    else:
        approve = lambda p: approve_spec
    ask_val = scn.get("ask", "")
    ask = lambda q, **kw: ask_val

    events = []
    result = await orch.orchestrate(
        None, None, [], [], scn.get("goal", "do something"), allowlist=set(), approve=approve,
        ask=ask, max_steps=scn.get("max_steps", 5), max_replans=scn.get("max_replans", 3),
        read_content=scn.get("read_content", False), ground=scn.get("ground", False),
        emit=events.append, observations=[])
    try:
        ok, note = scn["check"](result, events, cap)
    except Exception as e:
        ok, note = False, f"check raised {type(e).__name__}: {e}"
    return {"name": scn["name"], "ok": bool(ok), "note": str(note)[:90], "result": str(result)[:60]}


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"Orchestrator eval: {len(SCENARIOS)} scenarios (deterministic)\n")
    results = []
    for scn in SCENARIOS:                       # sequential: each rebinds orch.* module globals
        results.append(await run_scenario(scn))

    n = len(results)
    n_pass = sum(1 for r in results if r["ok"])
    for r in results:
        tag = "PASS" if r["ok"] else "FAIL"
        detail = "" if r["ok"] else f"  <-- {r['note']} | result={r['result']}"
        print(f"  [{tag}] {r['name'][:44]:<46}{detail}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rate": f"{n_pass}/{n}", "results": results}, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"ORCHESTRATOR SUCCESS RATE: {n_pass}/{n} = {100 * n_pass // n}%")
    print(f"Details -> {OUT}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
