"""Server logging: per-run JSONL event recorder + idempotent logger config."""

import json

from browser_agent.server.logging import EventRecorder, configure_logging


def test_event_recorder_writes_one_json_line_per_event(tmp_path):
    rec = EventRecorder("sess1", log_dir=tmp_path)
    rec.record({"type": "step", "step": 1, "action": "browser_navigate"})
    rec.record({"type": "run_finished", "result": "ok"})
    rec.close()

    lines = rec.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "step" and first["step"] == 1 and "ts" in first
    assert json.loads(lines[1])["type"] == "run_finished"


def test_event_recorder_tolerates_unserializable_values(tmp_path):
    rec = EventRecorder("sess2", log_dir=tmp_path)
    rec.record({"type": "x", "obj": object()})   # default=str keeps it from blowing up
    rec.close()
    lines = rec.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["type"] == "x"


def test_event_recorder_is_noop_without_writable_dir(tmp_path):
    # Pointing at a path that's a FILE (not a dir) can't be used as a log dir -> recorder degrades
    # to a no-op instead of raising.
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    rec = EventRecorder("sess3", log_dir=blocker)
    rec.record({"type": "step"})   # must not raise
    rec.close()
    assert rec.path is None or not rec.path.exists()


def test_trace_enabled_reads_log_level(monkeypatch):
    # The deep per-turn trace (fix 5a) is gated on BROWSER_AGENT_LOG_LEVEL ∈ {DEBUG, TRACE}; anything
    # else (the INFO default) keeps it off so a normal run's JSONL stays small.
    from browser_agent.log import trace_enabled
    monkeypatch.delenv("BROWSER_AGENT_LOG_LEVEL", raising=False)
    assert trace_enabled() is False                # default INFO -> off
    monkeypatch.setenv("BROWSER_AGENT_LOG_LEVEL", "DEBUG")
    assert trace_enabled() is True
    monkeypatch.setenv("BROWSER_AGENT_LOG_LEVEL", "trace")
    assert trace_enabled() is True                 # case-insensitive
    monkeypatch.setenv("BROWSER_AGENT_LOG_LEVEL", "WARNING")
    assert trace_enabled() is False


def test_configure_logging_idempotent():
    a = configure_logging()
    n = len(a.handlers)
    b = configure_logging()
    assert a is b and len(a.handlers) == n   # no duplicate handlers on repeat calls
    assert n >= 1
