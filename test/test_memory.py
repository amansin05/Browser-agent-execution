"""Deterministic tests for the session memory subsystem (SQLite, rolling context, scratchpad)."""

import pytest

from browser_agent.memory import MemorySession, MemoryStore
from browser_agent.memory.rolling import RollingContext
from browser_agent.memory.scratchpad import Scratchpad


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem"))
    yield s
    s.close()


# ----------------------------------------------------------------- store
def test_messages_roundtrip_and_order(store):
    sid = store.create_session()
    store.add_message(sid, "user", "first")
    store.add_message(sid, "assistant", "second")
    msgs = store.recent_messages(sid, 10)
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "first"), ("assistant", "second")]


def test_recent_messages_limit_keeps_newest(store):
    sid = store.create_session()
    for i in range(5):
        store.add_message(sid, "user", f"m{i}")
    msgs = store.recent_messages(sid, 2)
    assert [m["content"] for m in msgs] == ["m3", "m4"]  # newest two, chronological


def test_notes_roundtrip(store):
    sid = store.create_session()
    store.add_note(sid, "note A")
    store.add_note(sid, "note B")
    assert store.notes(sid, 10) == ["note A", "note B"]


def test_sessions_are_isolated(store):
    a = store.create_session()
    b = store.create_session()
    store.add_message(a, "user", "in-a")
    store.add_note(b, "in-b")
    assert store.recent_messages(b, 10) == []
    assert store.notes(a, 10) == []


def test_end_and_list_sessions(store):
    sid = store.create_session()
    store.end_session(sid)
    listed = {s["id"]: s for s in store.list_sessions()}
    assert sid in listed and listed[sid]["closed_at"] is not None


# ----------------------------------------------------------------- rolling
def test_rolling_window_turn_cap(store):
    sid = store.create_session()
    rc = RollingContext(store, sid, max_turns=3, max_chars=10_000)
    for i in range(6):
        rc.add("user", f"t{i}")
    assert [m["content"] for m in rc.window()] == ["t3", "t4", "t5"]


def test_rolling_window_char_budget(store):
    sid = store.create_session()
    rc = RollingContext(store, sid, max_turns=50, max_chars=12)
    rc.add("user", "aaaaa")   # 5
    rc.add("user", "bbbbb")   # 5
    rc.add("user", "ccccc")   # 5  -> 15 total > 12, oldest dropped
    contents = [m["content"] for m in rc.window()]
    assert contents == ["bbbbb", "ccccc"]


def test_rolling_text_format(store):
    sid = store.create_session()
    rc = RollingContext(store, sid)
    rc.add("user", "hi")
    rc.add("assistant", "yo")
    assert rc.text() == "user: hi\nassistant: yo"


# ----------------------------------------------------------------- scratchpad
def test_scratchpad_writes_and_skips_empty(store):
    sid = store.create_session()
    sp = Scratchpad(store, sid)
    sp.write("kept")
    sp.write("   ")     # blank -> ignored
    sp.write("")        # empty -> ignored
    assert sp.notes() == ["kept"]
    assert sp.text() == "- kept"


# ----------------------------------------------------------------- MemorySession
def test_memory_session_preamble_and_record(store):
    ms = MemorySession(store)
    ms.record_task("open example.com", "saw the heading")
    ms.note("dropdown options: Alpha, Beta")
    pre = ms.preamble()
    assert "Recent session history" in pre and "open example.com" in pre and "saw the heading" in pre
    assert "Scratchpad" in pre and "dropdown options" in pre


def test_memory_session_history_messages(store):
    ms = MemorySession(store)
    ms.record_task("task one", "result one")
    hist = ms.history_messages()
    assert {"role": "user", "content": "task one"} in hist
    assert {"role": "assistant", "content": "result one"} in hist


def test_memory_session_empty_preamble(store):
    ms = MemorySession(store)
    assert ms.preamble() == ""   # nothing recorded yet
