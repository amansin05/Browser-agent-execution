"""Tab-focus logic for AgentSession — picking which tab to surface a human prompt on, and the
fallback to the 'origin' tab (the one active before automation started) — plus the per-run
JSONL trace file that run_task writes."""

import json
from types import SimpleNamespace as NS

import browser_agent.agent.session as session_mod
from browser_agent.agent.session import AgentSession
from browser_agent.log import EventRecorder


def _sess(focus_url, origin_tab):
    """An AgentSession with just the attributes _find_chat_tab reads — no browser/Groq needed."""
    s = AgentSession.__new__(AgentSession)
    s.focus_url = focus_url
    s._origin_tab = origin_tab
    return s


def test_find_chat_tab_matches_focus_url():
    s = _sess("http://localhost:5173/", origin_tab=(9, "http://other/"))
    tabs = [(0, False, "chrome-extension://x/connect.html"),
            (2, True, "http://localhost:5173/#/run")]  # trailing route still matches
    assert s._find_chat_tab(tabs) == (2, "http://localhost:5173/#/run")


def test_find_chat_tab_falls_back_to_origin_by_url():
    # Chat URL isn't in the list, but the pre-automation origin tab is — return that.
    s = _sess("http://localhost:5173/", origin_tab=(3, "http://example.com/page"))
    tabs = [(0, False, "chrome-extension://x/connect.html"),
            (3, True, "http://example.com/page")]
    assert s._find_chat_tab(tabs) == (3, "http://example.com/page")


def test_find_chat_tab_falls_back_to_origin_by_index_when_url_moved():
    # The origin tab navigated since we recorded it, but its slot is still there — use the index.
    s = _sess("http://localhost:5173/", origin_tab=(3, "http://example.com/old"))
    tabs = [(3, True, "http://example.com/new")]
    assert s._find_chat_tab(tabs) == (3, "http://example.com/old")


def test_find_chat_tab_none_when_nothing_reachable():
    s = _sess("http://localhost:5173/", origin_tab=(9, "http://gone/"))
    tabs = [(0, True, "chrome-extension://x/connect.html")]
    assert s._find_chat_tab(tabs) is None


def test_find_chat_tab_no_origin_recorded():
    s = _sess("http://localhost:5173/", origin_tab=None)
    tabs = [(0, True, "chrome-extension://x/connect.html")]
    assert s._find_chat_tab(tabs) is None


def _bare_session():
    """An AgentSession with just what run_task needs (no real browser/Groq)."""
    s = AgentSession.__new__(AgentSession)
    s.session = object()        # non-None so run_task proceeds
    s.use_extension = False
    s.browser = "chrome"
    s.memory = None
    s.profile = None
    s.groq = None
    s.approve = lambda p: True
    s.ask = lambda *a, **k: ""
    s.focus_url = None
    s.reasoner_tools = []
    s.capabilities = []
    s.flat_tools = []
    s._origin_tab = None
    s._parked_tabs = []          # tab-parking state read at the top/bottom of run_task
    s._last_task = None
    return s


async def test_run_task_writes_trace_file(tmp_path, monkeypatch):
    # Redirect the trace recorder to a temp dir.
    monkeypatch.setattr(session_mod, "EventRecorder",
                        lambda sid: EventRecorder(sid, log_dir=tmp_path))

    async def fake_orchestrate(*a, emit=None, **k):
        emit({"type": "step", "step": 1, "action": "browser_navigate"})  # a mid-run event
        return "the answer"

    monkeypatch.setattr(session_mod, "orchestrate", fake_orchestrate)

    seen = []
    result = await _bare_session().run_task("buy a phone", emit=seen.append)

    assert result == "the answer"
    assert seen == [{"type": "step", "step": 1, "action": "browser_navigate"}]  # caller emit fired

    files = list(tmp_path.glob("run-cli-*.jsonl"))
    assert len(files) == 1
    events = [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines()]
    types = [e["type"] for e in events]
    assert types == ["run_started", "step", "run_finished"]    # boundaries + the forwarded event
    assert events[0]["task"] == "buy a phone"
    assert events[-1]["result"] == "the answer"
    assert all("ts" in e for e in events)                      # every line is timestamped


async def test_run_task_trace_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "EventRecorder",
                        lambda sid: EventRecorder(sid, log_dir=tmp_path))

    async def fake_orchestrate(*a, emit=None, **k):
        return "ok"

    monkeypatch.setattr(session_mod, "orchestrate", fake_orchestrate)

    await _bare_session().run_task("x", trace=False)           # server passes trace=False
    assert list(tmp_path.glob("*.jsonl")) == []                # no per-run trace written


async def test_run_task_records_error_into_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "EventRecorder",
                        lambda sid: EventRecorder(sid, log_dir=tmp_path))

    async def boom(*a, emit=None, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(session_mod, "orchestrate", boom)

    try:
        await _bare_session().run_task("x")
    except RuntimeError:
        pass                                                   # error propagates unchanged
    else:
        raise AssertionError("expected the error to propagate")

    files = list(tmp_path.glob("run-cli-*.jsonl"))
    events = [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines()]
    types = [e["type"] for e in events]
    assert types == ["run_started", "run_error"]               # failure captured, file closed
    assert "kaboom" in events[-1]["message"]


# ----------------------------------------------------------------- tab parking
class FakeTabSession:
    """A stateful stand-in for the Playwright MCP session: tracks a list of tabs and answers the
    browser_tabs (list/new/select/close) and browser_navigate calls _working_tabs / park / restore
    make. Indices are positional and renumber on close, like the real tool."""

    def __init__(self, tabs):
        # tabs: list of (url, is_current)
        self.tabs = [{"url": u, "current": c} for u, c in tabs]
        self.calls = []

    def _list_text(self):
        lines = []
        for i, t in enumerate(self.tabs):
            cur = "(current) " if t["current"] else ""
            lines.append(f"- {i}: {cur}[Tab {i}]({t['url']})")
        return "\n".join(lines)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "browser_tabs":
            action = args.get("action")
            if action == "list":
                return NS(content=[NS(type="text", text=self._list_text())], isError=False)
            if action == "new":
                for t in self.tabs:
                    t["current"] = False
                self.tabs.append({"url": "about:blank", "current": True})
            elif action == "select":
                for i, t in enumerate(self.tabs):
                    t["current"] = (i == args["index"])
            elif action == "close":
                del self.tabs[args["index"]]
        elif name == "browser_navigate":
            cur = next((t for t in self.tabs if t["current"]), None)
            if cur:
                cur["url"] = args["url"]
        return NS(content=[NS(type="text", text="ok")], isError=False)


def _tab_session(fake, *, focus_url=None, origin_tab=None, groq=None, use_extension=False):
    s = AgentSession.__new__(AgentSession)
    s.session = fake
    s.focus_url = focus_url
    s._origin_tab = origin_tab
    s._parked_tabs = []
    s._last_task = None
    s.memory = None
    s.groq = groq
    s.use_extension = use_extension
    return s


async def test_working_tabs_excludes_origin_and_non_http():
    fake = FakeTabSession([("http://example.com/page", True),   # origin (idx 0)
                           ("http://shop.com/a", False),
                           ("http://shop.com/b", False),
                           ("chrome-extension://x/connect.html", False)])  # non-http: skip
    s = _tab_session(fake, focus_url="http://localhost:5173/", origin_tab=(0, "http://example.com/page"))
    working = await s._working_tabs()
    assert working == [(1, "http://shop.com/a"), (2, "http://shop.com/b")]


async def test_working_tabs_empty_when_no_keep_tab():
    # Smoke/CLI: no focus_url and no origin tab => no identifiable keep tab => parking disabled.
    fake = FakeTabSession([("http://shop.com/a", True), ("http://shop.com/b", False)])
    s = _tab_session(fake)  # use_extension=False
    assert await s._working_tabs() == []


async def test_working_tabs_extension_closes_whole_group_except_devui():
    # EXTENSION mode: close every controlled http tab EXCEPT the dev-ui/chat tab — including what
    # used to be the preserved 'origin' tab. The user falls back to the dev-ui; URLs are restored.
    fake = FakeTabSession([("http://localhost:5173/", True),            # dev-ui (chat) -> keep
                           ("http://shop.com/a", False),
                           ("http://example.com/origin", False),         # the old 'origin' -> closes
                           ("chrome-extension://x/connect.html", False)])  # non-http -> skip
    s = _tab_session(fake, focus_url="http://localhost:5173/",
                     origin_tab=(2, "http://example.com/origin"), use_extension=True)
    working = await s._working_tabs()
    assert working == [(1, "http://shop.com/a"), (2, "http://example.com/origin")]


async def test_park_extension_closes_everything_but_devui():
    fake = FakeTabSession([("http://localhost:5173/", True),
                           ("http://shop.com/a", False),
                           ("http://shop.com/b", False)])
    s = _tab_session(fake, focus_url="http://localhost:5173/", use_extension=True)
    parked = await s._park_working_tabs()
    assert parked == ["http://shop.com/a", "http://shop.com/b"]
    assert [t["url"] for t in fake.tabs] == ["http://localhost:5173/"]  # only the dev-ui remains


async def test_park_records_urls_and_closes_only_working_tabs():
    fake = FakeTabSession([("http://example.com/page", True),
                           ("http://shop.com/a", False),
                           ("http://shop.com/b", False)])
    s = _tab_session(fake, origin_tab=(0, "http://example.com/page"))
    parked = await s._park_working_tabs()
    assert parked == ["http://shop.com/a", "http://shop.com/b"]
    assert s._parked_tabs == ["http://shop.com/a", "http://shop.com/b"]
    # Only the origin tab survives; the two working tabs were closed.
    assert [t["url"] for t in fake.tabs] == ["http://example.com/page"]
    # Closed highest-index-first so the captured indices stayed valid.
    closes = [a["index"] for n, a in fake.calls if n == "browser_tabs" and a.get("action") == "close"]
    assert closes == [2, 1]


async def test_park_closes_but_does_not_remember_search_engine_tab():
    # A leftover Google results tab is closed like any working tab, but NOT parked — reopening a SERP
    # on a follow-up is what made the next task scrape search results as fake products.
    fake = FakeTabSession([("http://localhost:5173/", True),
                           ("https://www.google.com/search?q=asics+under+6000", False),
                           ("http://shop.com/a", False)])
    s = _tab_session(fake, focus_url="http://localhost:5173/", use_extension=True)
    parked = await s._park_working_tabs()
    assert parked == ["http://shop.com/a"]                 # the SERP tab is not remembered
    assert s._parked_tabs == ["http://shop.com/a"]
    assert [t["url"] for t in fake.tabs] == ["http://localhost:5173/"]  # but it WAS closed


async def test_restore_reopens_parked_tabs_and_clears():
    fake = FakeTabSession([("http://example.com/page", True)])
    s = _tab_session(fake, origin_tab=(0, "http://example.com/page"))
    s._parked_tabs = ["http://shop.com/a", "http://shop.com/b"]
    reopened = await s._restore_parked_tabs()
    assert reopened == ["http://shop.com/a", "http://shop.com/b"]
    assert s._parked_tabs == []                                # parked list consumed
    urls = [t["url"] for t in fake.tabs]
    assert urls == ["http://example.com/page", "http://shop.com/a", "http://shop.com/b"]


async def test_is_followup_uses_llm_verdict():
    class G:
        def __init__(self, content):
            async def create(**kw):
                return NS(choices=[NS(message=NS(content=content))])
            self.chat = NS(completions=NS(create=create))

    s = _tab_session(FakeTabSession([]), groq=G('{"followup": false}'))
    s._last_task = "buy a phone"
    s._parked_tabs = ["http://shop.com/a"]
    assert await s._is_followup("what's the weather") is False

    s2 = _tab_session(FakeTabSession([]), groq=G('{"followup": true}'))
    s2._last_task = "buy a phone"
    s2._parked_tabs = ["http://shop.com/a"]
    assert await s2._is_followup("which one has the best camera") is True


async def test_is_followup_defaults_true_without_prior_task():
    s = _tab_session(FakeTabSession([]), groq=None)
    assert await s._is_followup("anything") is True            # no prev task / no groq => resume


# ----------------------------------------------------------------- dedicated working tab
async def test_ensure_working_tab_opens_one_when_only_origin():
    # Only your origin tab is open -> the agent must open + select its OWN tab so it never drives
    # yours (this is what makes park/restore possible — there's a working tab to close).
    fake = FakeTabSession([("http://example.com/page", True)])
    s = _tab_session(fake, focus_url="http://localhost:5173/", origin_tab=(0, "http://example.com/page"))
    s.use_extension = True
    await s._ensure_working_tab()
    assert any(n == "browser_tabs" and a.get("action") == "new" for n, a in fake.calls)
    assert len(fake.tabs) == 2 and fake.tabs[-1]["current"] is True


async def test_ensure_working_tab_noop_when_working_tab_exists():
    # A working tab already exists (e.g. parked tabs were just reopened) -> don't open another.
    fake = FakeTabSession([("http://example.com/page", True), ("http://shop.com/a", False)])
    s = _tab_session(fake, focus_url="http://localhost:5173/", origin_tab=(0, "http://example.com/page"))
    s.use_extension = True
    await s._ensure_working_tab()
    assert not any(n == "browser_tabs" and a.get("action") == "new" for n, a in fake.calls)
    assert len(fake.tabs) == 2  # unchanged
