"""Deterministic tests for parallel multi-source gathering (agent/gather.py).

No browser: we fake open_session (the headless worker session), run_subgoal (the per-source drive),
full_snapshot, and extract_candidates, then assert gather_parallel runs every source, aggregates
{id: candidates}, emits per-source events, and maps a failing source to [] without raising.
"""

import contextlib

import browser_agent.agent.gather as gather
import browser_agent.agent.orchestrate as orch


def _fake_open_session(record=None):
    @contextlib.asynccontextmanager
    async def fake(use_extension, browser, headless=False):
        if record is not None:
            record.append({"use_extension": use_extension, "browser": browser, "headless": headless})
        yield (object(), None)  # (session, params)
    return fake


async def _ok_run_subgoal(*a, **k):
    return ("complete", "ok")


async def _snap(_sess):
    return ("page snapshot", "http://source/")


def _patch_common(monkeypatch, *, extract, open_rec=None, readable=None):
    monkeypatch.setattr(gather, "open_session", _fake_open_session(open_rec))
    monkeypatch.setattr(orch, "run_subgoal", _ok_run_subgoal)   # lazy-imported inside the worker
    monkeypatch.setattr(gather, "full_snapshot", _snap)
    monkeypatch.setattr(gather, "extract_candidates", extract)
    monkeypatch.setattr(gather, "page_looks_blocked", lambda s: None)
    if readable is not None:
        monkeypatch.setattr(gather, "readable_text", readable)


async def test_gather_parallel_aggregates_per_source(monkeypatch):
    counter = {"n": 0}

    async def fake_extract(groq, snap, spec, *, model=None):
        counter["n"] += 1
        return [{"name": f"book{counter['n']}"}]

    open_rec = []
    _patch_common(monkeypatch, extract=fake_extract, open_rec=open_rec)

    subgoals = [{"id": 1, "goal": "amazon", "explore_spec": {}},
                {"id": 2, "goal": "thriftbooks", "explore_spec": {}}]
    events = []
    results = await gather.gather_parallel(None, [], subgoals, allowlist=set(), max_steps=5,
                                           browser="chrome", emit=events.append)

    assert set(results) == {1, 2}
    assert all(len(v) == 1 for v in results.values())          # each source contributed candidates
    # Every worker opened a HEADLESS, non-extension session.
    assert open_rec and all(r["headless"] and not r["use_extension"] for r in open_rec)
    # Per-source events for the UI.
    assert len([e for e in events if e["type"] == "subgoal_start"]) == 2
    assert len([e for e in events if e["type"] == "candidates"]) == 2
    assert len([e for e in events if e["type"] == "subgoal_end"]) == 2


async def test_gather_parallel_salvages_via_readable(monkeypatch):
    # First extract (snapshot) returns nothing; the readable-text re-extract recovers candidates.
    calls = {"n": 0}

    async def fake_extract(groq, snap, spec, *, model=None):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [{"name": "salvaged"}]

    async def fake_readable(_sess):
        return "Some product grid text"

    _patch_common(monkeypatch, extract=fake_extract, readable=fake_readable)
    results = await gather.gather_parallel(None, [], [{"id": 5, "goal": "x", "explore_spec": {}}],
                                           allowlist=set(), max_steps=5, emit=lambda e: None)
    assert results == {5: [{"name": "salvaged"}]}
    assert calls["n"] == 2                                      # snapshot, then readable re-extract


async def test_gather_parallel_failed_source_yields_empty(monkeypatch):
    async def boom_extract(groq, snap, spec, *, model=None):
        raise RuntimeError("extract boom")

    async def fake_readable(_sess):
        return None

    _patch_common(monkeypatch, extract=boom_extract, readable=fake_readable)
    events = []
    results = await gather.gather_parallel(None, [], [{"id": 7, "goal": "x", "explore_spec": {}}],
                                           allowlist=set(), max_steps=3, emit=events.append)
    assert results == {7: []}                                  # failure -> empty, never raises
    end = next(e for e in events if e["type"] == "subgoal_end")
    assert end["status"] == "escalate"
