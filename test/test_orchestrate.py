"""orchestrate() flow: clarification cap and salvage-extraction on a non-completing explore,
plus the role->model wiring (planner=qwen, composer=gpt-oss)."""

from types import SimpleNamespace as NS

import browser_agent.agent.orchestrate as orch
import browser_agent.config as cfg
from browser_agent.agent.extract import synthesize_options
from browser_agent.agent.planner import plan_subgoals


class RecordingGroq:
    """Captures the `model` of the last create() call and returns scripted content."""
    def __init__(self, content):
        self.model = None

        async def create(**kw):
            self.model = kw.get("model")
            return NS(choices=[NS(message=NS(content=content))])

        self.chat = NS(completions=NS(create=create))


def _explore():
    return {"id": 1, "type": "explore", "goal": "find phones", "success_condition": "5+",
            "tier": "auto", "needs_approval": False, "explore_spec": {"sources": ["amazon"]}}


def _present():
    return {"id": 2, "type": "present", "goal": "show", "success_condition": "shown",
            "tier": "auto", "needs_approval": False}


def _ask():
    return {"id": 1, "type": "ask_user", "goal": "clarify", "question": "Budget?",
            "success_condition": "", "tier": "auto", "needs_approval": False,
            "allow_free_text": True, "multi_select": False}


async def _orchestrate(**over):
    # ground=False: grounding needs a live browser; these tests run with session=None.
    kw = dict(allowlist=set(), approve=lambda p: True, ask=lambda *a, **k: "",
              max_steps=5, max_replans=3, read_content=False, ground=False)
    kw.update(over)
    return await orch.orchestrate(None, None, [], [], "buy a phone", **kw)


async def test_orchestrate_salvages_candidates_when_explore_escalates(monkeypatch):
    seen = {}

    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("escalate", "loop detected")          # reasoner never called subgoal_complete

    async def fake_observe(session):
        return ("a page full of phone listings", "http://amazon.in/s")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return [{"name": "Phone A", "source": "amazon"}]   # but listings ARE on the page

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        seen["cands"] = cands
        return "## Shortlist\n- Phone A"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate()
    # The explore escalated, but candidates on the page were salvaged -> present ran -> we get a
    # shortlist instead of "Failed: exhausted re-plan budget".
    assert result == "## Shortlist\n- Phone A"
    assert seen["cands"] == [{"name": "Phone A", "source": "amazon"}]


async def test_orchestrate_salvages_via_content_extractor(monkeypatch):
    # Heavy retail SPA: the a11y snapshot yields 0 candidates, but the content extractor
    # (readable_text) recovers the product text and a re-extract finds them. The run should proceed
    # to present instead of burning the re-plan budget (the "i want to buy a phone" failure).
    seen = {}
    calls = {"extract": 0}

    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")

    async def fake_observe(session):
        return ("nav chrome only, grid not in a11y tree", "http://flipkart.com/mobiles/pr")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        calls["extract"] += 1
        # first call (raw snapshot) -> nothing; second call (readable text) -> candidates
        return [] if calls["extract"] == 1 else [{"name": "Galaxy M14", "source": "flipkart"}]

    async def fake_readable(session):
        return "# Mobiles\nGalaxy M14 5G Rs 12,999\nRedmi 13C Rs 9,499"

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        seen["cands"] = cands
        return "## Shortlist\n- Galaxy M14"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)
    monkeypatch.setattr(orch, "readable_text", fake_readable)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate(read_content=True)
    assert result == "## Shortlist\n- Galaxy M14"
    assert seen["cands"] == [{"name": "Galaxy M14", "source": "flipkart"}]
    assert calls["extract"] == 2          # raw snapshot, then content-extractor re-pass


async def test_orchestrate_does_not_present_empty_when_explore_finds_nothing(monkeypatch):
    plans = {"n": 0}

    async def fake_plan(*a, **k):
        plans["n"] += 1
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")          # reasoner CLAIMS done...

    async def fake_observe(session):
        return ("a page with no listings", "http://x")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return []                          # ...but nothing extractable -> not really done

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)

    result = await _orchestrate(max_replans=1)
    # It must re-plan (not march on to present an empty shortlist), then fail honestly.
    assert plans["n"] >= 2 and result.startswith("Failed")


def test_page_looks_blocked_detects_walls_and_passes_real_pages():
    from browser_agent.utils.text import page_looks_blocked
    assert page_looks_blocked("Enter the characters you see below")
    assert page_looks_blocked("Our systems have detected unusual traffic from your network")
    assert page_looks_blocked("Please complete the CAPTCHA to continue")
    # Real listings and an empty string are NOT walls.
    assert page_looks_blocked("Galaxy M33 5G  4.2 stars  Rs 13999  Add to cart") is None
    assert page_looks_blocked("") is None


async def test_orchestrate_steers_replan_away_from_blocked_source(monkeypatch):
    captured = {}

    async def fake_plan(*a, **k):
        if k.get("failed_id") is not None:        # the re-plan call
            captured["reason"] = k.get("reason")
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("escalate", "loop detected")

    async def fake_observe(session):
        return ("Enter the characters you see below — we just need to make sure you're not a robot",
                "https://www.amazon.in/s?k=phones")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return []                                  # blocked page -> nothing extractable

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)

    await _orchestrate(max_replans=1)
    # The re-plan reason must name the dead source, flag it as a wall, and steer elsewhere.
    reason = captured["reason"]
    assert "amazon.in" in reason
    assert "DIFFERENT" in reason
    assert "bot-check" in reason
    assert "avoid these" in reason.lower()


async def test_orchestrate_caps_clarifications(monkeypatch):
    calls = {"ask": 0}

    async def fake_plan(*a, **k):
        return [_ask()]                                # planner keeps re-asking a clarification

    async def fake_run(*a, **k):
        return ("complete", "ok")

    def fake_ask(q, **k):
        calls["ask"] += 1
        return "Budget"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)

    result = await _orchestrate(ask=fake_ask)
    # Without the cap this loops forever (every re-plan re-asks). With it, we ask at most the cap
    # then skip and finish.
    assert calls["ask"] == orch.MAX_CLARIFICATIONS
    assert result == "Done."


async def test_planner_uses_planner_model():
    g = RecordingGroq('{"subgoals": [{"id": 1, "goal": "g", "success_condition": "s"}]}')
    await plan_subgoals(g, "buy a phone", ["browser_navigate"])
    assert g.model == cfg.PLANNER_MODEL


async def test_synthesize_uses_composition_model():
    g = RecordingGroq("## Shortlist")
    await synthesize_options(g, "buy a phone", [{"name": "P", "source": "x"}])
    assert g.model == cfg.COMPOSITION_MODEL
