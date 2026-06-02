"""Deterministic regression tests for the REASONER role: the per-turn decision (reasoner.py), the
run_subgoal loop and its guards (orchestrate.py), the independent verifier, and the content-extractor
fallback that replaced vision. Planner-side behaviour lives in test_planner.py.
"""

import copy
from types import SimpleNamespace as NS

from groq import BadRequestError

import browser_agent.agent.orchestrate as orch
from browser_agent.agent.main import run_subgoal
from browser_agent.agent.reasoner import reasoner_decide
from browser_agent.agent.verifier import verify_success
from browser_agent.services.mcp_client import newest_new_tab
from browser_agent.utils.domains import domain_allowed
from browser_agent.utils.text import snapshot_is_sufficient


# ----------------------------------------------------------------- fakes
def make_tc(name, arguments):
    return NS(id="c1", type="function", function=NS(name=name, arguments=arguments))


def reasoner_resp(name, arguments="{}", thought="thinking"):
    return NS(choices=[NS(message=NS(content=thought, tool_calls=[make_tc(name, arguments)]))])


def content_resp(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))])


class FakeGroq:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

        async def create(**kwargs):
            self.calls.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != "tools"}))
            nxt = self.scripted.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        self.chat = NS(completions=NS(create=create))


class FakeSession:
    def __init__(self, url="http://test.local/", raise_on=None, results=None,
                 snapshot_body='- heading "h" [ref=e1]'):
        self.url = url
        self.calls = []
        self.raise_on = set(raise_on or ())
        self.results = results or {}
        self.snapshot_body = snapshot_body  # default has a ref => "sufficient"

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name in self.raise_on:
            raise RuntimeError("boom")
        if name == "browser_snapshot":
            txt = f"### Page\n- Page URL: {self.url}\n### Snapshot\n```yaml\n{self.snapshot_body}\n```"
            return NS(content=[NS(type="text", text=txt)], isError=False)
        return self.results.get(name, NS(content=[NS(type="text", text=f"ok-{name}")], isError=False))


SUBGOAL = {"goal": "g", "success_condition": "c"}


def _run(groq, session, **kw):
    defaults = dict(allowlist=set(), approve=lambda p: True, ask=lambda q: "",
                    observations=[], max_steps=5)
    defaults.update(kw)
    return run_subgoal(groq, session, [], SUBGOAL, **defaults)


def _bad_request(message="tool_use_failed: invalid tool arguments"):
    """A BadRequestError without hitting the network — what Scout throws on a tool-call wobble."""
    e = BadRequestError.__new__(BadRequestError)
    e.body = {"error": {"message": message}}
    return e


# ----------------------------------------------------------------- helpers used by the reasoner
def test_domain_allowed():
    assert domain_allowed("http://anything.com", set()) is True
    assert domain_allowed("https://example.com/p", {"example.com"}) is True
    assert domain_allowed("https://api.example.com", {"example.com"}) is True
    assert domain_allowed("https://evil.com", {"example.com"}) is False


def test_snapshot_is_sufficient():
    assert snapshot_is_sufficient("### Snapshot\n```yaml\n```") is False
    assert snapshot_is_sufficient('```yaml\n- heading "x" [ref=e2]\n```') is True
    assert snapshot_is_sufficient("```yaml\n- generic\n```") is False
    assert snapshot_is_sufficient("```yaml\n" + "- a long descriptive line of text\n" * 2 + "```") is True


def test_newest_new_tab_logic():
    assert newest_new_tab([0], [0, 1]) == 1          # a click opened tab 1 -> follow it
    assert newest_new_tab([0], [0, 1, 2]) == 2        # follow the newest of several
    assert newest_new_tab([0], [0]) is None           # no new tab
    assert newest_new_tab([0, 1], [0, 1]) is None     # nothing opened


async def test_full_snapshot_not_truncated():
    # Candidate extraction reads the FULL snapshot so retail product grids past the 12k cap are
    # visible (the Amazon 0-candidates failure); observe() stays capped for the reasoner.
    from browser_agent.config import MAX_TOOL_RESULT_CHARS
    from browser_agent.services.mcp_client import full_snapshot, observe

    big = "p" * (MAX_TOOL_RESULT_CHARS + 6000)

    class S:
        async def call_tool(self, name, args):
            return NS(content=[NS(type="text", text=f"### Page\n- Page URL: http://x/\n{big}")], isError=False)

    sess = S()
    full, url = await full_snapshot(sess)
    capped, _ = await observe(sess)
    assert url == "http://x/"
    assert len(full) > MAX_TOOL_RESULT_CHARS + 5000          # full text preserved
    assert len(capped) <= MAX_TOOL_RESULT_CHARS + 50          # observe still truncates


def test_reasoner_excludes_noop_trap_tools():
    # The reasoner must never be offered the "free no-op" tools it loops on (snapshot/screenshot are
    # auto-provided; wait/evaluate/hover drain the budget without progress — see the "buy a phone"
    # run where it looped on browser_hover across every retailer).
    from browser_agent.agent.tools import REASONER_EXCLUDED, build_reasoner_tools

    class T:
        def __init__(self, name): self.name, self.description, self.inputSchema = name, "", None
    for t in ("browser_hover", "browser_snapshot", "browser_take_screenshot", "browser_wait_for",
              "browser_evaluate"):
        assert t in REASONER_EXCLUDED
    tools, caps = build_reasoner_tools([T("browser_navigate"), T("browser_hover"), T("browser_click")])
    names = {t["function"]["name"] for t in tools}
    assert "browser_hover" not in names and "browser_navigate" in names


# ----------------------------------------------------------------- verifier (independent check)
async def test_verify_satisfied():
    groq = FakeGroq([content_resp('{"satisfied": true, "reason": "results visible"}')])
    ok, reason = await verify_success(groq, "snapshot", "results are visible")
    assert ok is True and "results" in reason


async def test_verify_malformed():
    groq = FakeGroq([content_resp("totally not json")])
    ok, reason = await verify_success(groq, "snapshot", "x")
    assert ok is False and "parse error" in reason


# ----------------------------------------------------------------- reasoner_decide (one turn)
async def test_reasoner_decide_returns_single_action():
    groq = FakeGroq([reasoner_resp("browser_click", '{"target": "#go"}', thought="clicking")])
    thought, action, args = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert action == "browser_click" and args == {"target": "#go"} and thought == "clicking"


async def test_reasoner_decide_recovers_from_bad_request():
    # Scout's tool_use_failed wobble: a BadRequestError, then a valid call on the retry.
    groq = FakeGroq([_bad_request(), reasoner_resp("browser_navigate", '{"url": "http://x"}')])
    _, action, args = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert action == "browser_navigate" and args == {"url": "http://x"}
    # The retry turn fed the rejection back to the model.
    assert any("rejected" in m["content"] for m in groq.calls[1]["messages"] if m["role"] == "user")


async def test_reasoner_decide_no_tool_call_escalates():
    groq = FakeGroq([content_resp("I am unsure")])
    _, action, _ = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert action == "escalate"


# ----------------------------------------------------------------- run_subgoal loop branches
async def test_subgoal_complete_verified():
    groq = FakeGroq([reasoner_resp("subgoal_complete", '{"note": "done"}'),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    status, _ = await _run(groq, FakeSession())
    assert status == "complete"


async def test_verifier_reject_then_escalate():
    groq = FakeGroq([reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": false, "reason": "not yet"}'),
                     reasoner_resp("escalate", '{"reason": "cannot finish"}')])
    status, detail = await _run(groq, FakeSession())
    assert status == "escalate" and "cannot finish" in detail


async def test_allowlist_blocks_navigation():
    groq = FakeGroq([reasoner_resp("browser_navigate", '{"url": "https://evil.com/x"}'),
                     reasoner_resp("escalate", '{"reason": "blocked"}')])
    session = FakeSession(url="https://example.com/")
    status, _ = await _run(groq, session, allowlist={"example.com"})
    assert status == "escalate"
    assert ("browser_navigate", {"url": "https://evil.com/x"}) not in session.calls


async def test_loop_detection_same_action_same_args():
    # The first loop NUDGES (one chance to change tactics); a SECOND loop escalates. So a reasoner
    # that just keeps repeating still hands back — it just gets one corrective turn first.
    groq = FakeGroq([reasoner_resp("browser_wait_for", '{"time": 1}') for _ in range(8)])
    status, detail = await _run(groq, FakeSession(), max_steps=12)
    assert status == "escalate" and "loop" in detail


async def test_loop_detection_warns_once_before_escalating():
    # Exactly one loop -> a nudge, not an escalate: the reasoner recovers (navigates) and completes.
    groq = FakeGroq([reasoner_resp("browser_click", '{"target": "#x"}'),
                     reasoner_resp("browser_click", '{"target": "#x"}'),
                     reasoner_resp("browser_click", '{"target": "#x"}'),    # 3rd identical -> loop -> NUDGE
                     reasoner_resp("browser_navigate", '{"url": "http://test.local/list"}'),  # recovered
                     reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    status, _ = await _run(groq, FakeSession(), max_steps=12)
    assert status == "complete"


async def test_loop_detection_same_action_varied_args():
    # Same action NAME with wobbling args (the Amazon failure shape) must still trip the looser
    # name-based detector — after the one-shot nudge, a persistent name-loop escalates.
    groq = FakeGroq([reasoner_resp("browser_click", f'{{"target": "#x{i}"}}') for i in range(8)])
    status, detail = await _run(groq, FakeSession(), max_steps=12)
    assert status == "escalate" and "loop" in detail


async def test_repeated_rejected_complete_escalates():
    # The reasoner keeps declaring done; the verifier keeps refusing. Must escalate (not spam
    # subgoal_complete until the budget drains).
    scripted = []
    for _ in range(5):
        scripted.append(reasoner_resp("subgoal_complete"))
        scripted.append(content_resp('{"satisfied": false, "reason": "nothing here"}'))
    groq = FakeGroq(scripted)
    status, detail = await _run(groq, FakeSession(), max_steps=10)
    assert status == "escalate" and "refused completion" in detail


async def test_step_budget_exhausted():
    groq = FakeGroq([reasoner_resp("browser_wait_for", '{"time": 1}'),
                     reasoner_resp("browser_wait_for", '{"time": 2}')])
    status, _ = await _run(groq, FakeSession(), max_steps=2)
    assert status == "exhausted"


async def test_explore_spec_reaches_reasoner_prompt():
    # The reasoner must SEE the explore_spec sources — without them it flails on whatever tab is
    # open instead of routing to the planned sources (the Amazon root-cause).
    groq = FakeGroq([reasoner_resp("escalate", '{"reason": "x"}')])
    sg = {"goal": "Find Levi's 501 jeans retailers", "success_condition": "c", "type": "explore",
          "explore_spec": {"sources": ["Google", "Amazon"], "must_have": ["Levi's 501"],
                           "target_count": 5}}
    await run_subgoal(groq, FakeSession(), [], sg, allowlist=set(), approve=lambda p: True,
                      ask=lambda q: "", observations=[], max_steps=1)
    prompt = groq.calls[0]["messages"][1]["content"]
    assert "Exploration guidance" in prompt and "Google" in prompt and "Amazon" in prompt


# ----------------------------------------------------------------- follow-new-tab
class TabFakeSession:
    """A session whose first browser_click opens a new tab — and (like Playwright's real lag)
    leaves its 'current' on the OPENER. Proves the agent follows + activates the new tab."""
    def __init__(self):
        self.tabs = [(0, "http://work/")]
        self.current = 0
        self.calls = []
        self._opened = False

    def _tabs_text(self):
        return "\n".join(f"- {i}: {'(current) ' if i == self.current else ''}[Tab{i}]({u})"
                         for i, u in self.tabs)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "browser_tabs" and args.get("action") == "list":
            return NS(content=[NS(type="text", text=self._tabs_text())], isError=False)
        if name == "browser_tabs" and args.get("action") == "select":
            self.current = args["index"]
            return NS(content=[NS(type="text", text="ok")], isError=False)
        if name == "browser_snapshot":
            url = dict(self.tabs)[self.current]
            return NS(content=[NS(type="text", text=f"### Page\n- Page URL: {url}\n- heading \"h\" [ref=e1]")],
                      isError=False)
        if name == "browser_click" and not self._opened:
            self._opened = True
            self.tabs.append((1, "http://phone-detail/"))  # opened the detail in a new tab
            return NS(content=[NS(type="text", text="opened a new tab")], isError=False)
        return NS(content=[NS(type="text", text=f"ok-{name}")], isError=False)


async def test_run_subgoal_follows_new_tab():
    groq = FakeGroq([reasoner_resp("browser_click", '{"element": "e1", "target": "#x"}'),
                     reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    sess = TabFakeSession()
    status, _ = await run_subgoal(groq, sess, [], SUBGOAL, allowlist=set(), approve=lambda p: True,
                                  ask=lambda q: "", observations=[], max_steps=5, read_content=False)
    assert status == "complete"
    assert ("browser_tabs", {"action": "select", "index": 1}) in sess.calls  # followed the new tab
    assert sess.current == 1


# ----------------------------------------------------------------- content-extractor fallback
# (replaced the old screenshot/vision fallback — see agent/reader.py + orchestrate.run_subgoal)
async def test_extractor_fallback_on_sparse_snapshot(monkeypatch):
    captured = {}

    async def fake_readable(session):
        captured["called"] = True
        return "# Levi's 501\n\nClassic straight fit. ₹3,499. In stock at multiple retailers."

    monkeypatch.setattr(orch, "readable_text", fake_readable)
    groq = FakeGroq([reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    session = FakeSession(snapshot_body="")  # empty -> insufficient -> extractor runs
    status, _ = await _run(groq, session, read_content=True)
    assert status == "complete" and captured.get("called") is True
    # The extracted text is appended to the page observation the reasoner sees, and NO screenshot is taken.
    prompt = groq.calls[0]["messages"][1]["content"]
    assert isinstance(prompt, str)                          # text-only: never an image content list
    assert "### Extracted readable content" in prompt and "Levi's 501" in prompt
    assert ("browser_take_screenshot", {"type": "png"}) not in session.calls


async def test_extractor_skipped_when_snapshot_sufficient(monkeypatch):
    called = {"n": 0}

    async def fake_readable(session):
        called["n"] += 1
        return "should not be used"

    monkeypatch.setattr(orch, "readable_text", fake_readable)
    groq = FakeGroq([reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    await _run(groq, FakeSession(), read_content=True)      # default body has a ref => sufficient
    assert called["n"] == 0


async def test_extractor_disabled_skips_extraction(monkeypatch):
    called = {"n": 0}

    async def fake_readable(session):
        called["n"] += 1
        return "x"

    monkeypatch.setattr(orch, "readable_text", fake_readable)
    groq = FakeGroq([reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    session = FakeSession(snapshot_body="")                 # insufficient, but read_content=False
    await _run(groq, session, read_content=False)
    assert called["n"] == 0
    prompt = groq.calls[0]["messages"][1]["content"]
    assert "### Extracted readable content" not in prompt
