"""Deterministic regression tests for the REASONER role: the per-turn decision (reasoner.py), the
run_subgoal loop and its guards (orchestrate.py), the independent verifier, and the content-extractor
fallback that replaced vision. Planner-side behaviour lives in test_planner.py.
"""

import copy
import json
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


def reasoner_multi(calls, thought="thinking"):
    """A reasoner response with SEVERAL tool_calls in one turn (calls = [(name, arguments_json), ...])."""
    return NS(choices=[NS(message=NS(content=thought,
                                     tool_calls=[make_tc(n, a) for n, a in calls]))])


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


def _eval_result(value):
    """A browser_evaluate result as Playwright MCP renders it (JSON under '### Result')."""
    return NS(content=[NS(type="text", text=f"### Result\n{json.dumps(value)}\n### Page\n- ...")],
              isError=False)


def _dom_payload(n=3, url="http://test.local/", title="T", scroll_height=500, inner_height=900):
    """A buildDomTree payload with `n` simple clickable elements (indices 0..n-1)."""
    els = [{"i": i, "tag": "button", "type": "", "role": "", "name": f"btn{i}", "text": f"btn{i}",
            "href": "", "inViewport": True} for i in range(n)]
    return {"url": url, "title": title, "scrollY": 0, "scrollHeight": scroll_height,
            "innerHeight": inner_height, "elements": els}


class FakeSession:
    """A fake MCP session. `dom` is the buildDomTree payload index_dom() reads (None forces the a11y
    snapshot fallback); `action_result` is what an index-action eval returns. browser_evaluate is
    routed to one or the other by inspecting the injected function (the build pass clears all
    data-ba-id attributes; an action queries one)."""
    def __init__(self, url="http://test.local/", raise_on=None, results=None,
                 snapshot_body='- heading "h" [ref=e1]', dom=None, action_result=None, dom_vary=False):
        self.url = url
        self.calls = []
        self.raise_on = set(raise_on or ())
        self.results = results or {}
        self.snapshot_body = snapshot_body  # default has a ref => "sufficient"
        self.dom = dom                      # None => index_dom returns None => a11y fallback
        self.action_result = action_result if action_result is not None else {"ok": True}
        self.dom_vary = dom_vary            # bump scrollY each build so the page fingerprint changes
        self._builds = 0

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name in self.raise_on:
            raise RuntimeError("boom")
        if name == "browser_evaluate":
            fn = args.get("function", "")
            if "querySelectorAll('[data-ba-id]')" in fn:          # the buildDomTree pass
                if self.dom is None:
                    return NS(content=[NS(type="text", text="(no result)")], isError=False)
                payload = dict(self.dom)
                if self.dom_vary:                                 # simulate the page changing
                    self._builds += 1
                    payload["scrollY"] = self._builds * 100
                return _eval_result(payload)
            return _eval_result(self.action_result)               # an index-action eval
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


def test_build_reasoner_tools_is_index_based():
    # The reasoner now gets a FIXED index-addressed action set (not the raw MCP element tools), so
    # the old "free no-op" traps (snapshot/screenshot/wait/evaluate/hover) and ref-based clicks are
    # simply absent — replaced by click_element/input_text/etc that act on a stable [index].
    from browser_agent.agent.tools import build_reasoner_tools

    tools, caps = build_reasoner_tools()  # no MCP tool list needed any more
    names = {t["function"]["name"] for t in tools}
    assert {"click_element", "input_text", "select_option", "scroll_page", "navigate"} <= names
    assert {"subgoal_complete", "escalate", "ask_human"} <= names
    # No raw MCP element/no-op tools leak in.
    assert not ({"browser_click", "browser_type", "browser_snapshot", "browser_hover",
                 "browser_evaluate", "browser_wait_for"} & names)
    assert "click_element" in caps and "navigate" in caps  # planner is told the high-level verbs


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
async def test_reasoner_decide_returns_action_list():
    groq = FakeGroq([reasoner_resp("click_element", '{"index": 3}', thought="clicking")])
    thought, actions = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert actions == [("click_element", {"index": 3})] and thought == "clicking"


async def test_reasoner_decide_returns_multiple_actions():
    # The model may chain several tool calls in one turn; reasoner_decide returns them in order.
    groq = FakeGroq([reasoner_multi([("input_text", '{"index": 0, "text": "hi"}'),
                                     ("click_element", '{"index": 1}')])])
    _, actions = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert actions == [("input_text", {"index": 0, "text": "hi"}), ("click_element", {"index": 1})]


async def test_reasoner_decide_recovers_from_bad_request():
    # Scout's tool_use_failed wobble: a BadRequestError, then a valid call on the retry.
    groq = FakeGroq([_bad_request(), reasoner_resp("navigate", '{"url": "http://x"}')])
    _, actions = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert actions == [("navigate", {"url": "http://x"})]
    # The retry turn fed the rejection back to the model.
    assert any("rejected" in m["content"] for m in groq.calls[1]["messages"] if m["role"] == "user")


async def test_reasoner_decide_no_tool_call_escalates():
    groq = FakeGroq([content_resp("I am unsure")])
    _, actions = await reasoner_decide(groq, [], SUBGOAL, "snapshot", [])
    assert actions[0][0] == "escalate"


async def test_reasoner_decide_echoes_previous_thought():
    # The prior turn's note is fed back for continuity.
    groq = FakeGroq([reasoner_resp("scroll_page", '{"direction": "down"}')])
    await reasoner_decide(groq, [], SUBGOAL, "snapshot", [], last_thought="I just searched")
    prompt = groq.calls[0]["messages"][1]["content"]
    assert "Your previous note" in prompt and "I just searched" in prompt


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
    groq = FakeGroq([reasoner_resp("navigate", '{"url": "https://evil.com/x"}'),
                     reasoner_resp("escalate", '{"reason": "blocked"}')])
    session = FakeSession(url="https://example.com/")
    status, _ = await _run(groq, session, allowlist={"example.com"}, read_content=False)
    assert status == "escalate"
    # Blocked BEFORE execution -> the MCP navigate is never dispatched.
    assert ("browser_navigate", {"url": "https://evil.com/x"}) not in session.calls


async def test_loop_detection_same_action_same_args():
    # The first loop NUDGES (one chance to change tactics); a SECOND loop escalates. So a reasoner
    # that just keeps repeating still hands back — it just gets one corrective turn first.
    groq = FakeGroq([reasoner_resp("click_element", '{"index": 1}') for _ in range(8)])
    status, detail = await _run(groq, FakeSession(dom=_dom_payload()), max_steps=12, read_content=False)
    assert status == "escalate" and "loop" in detail


async def test_loop_detection_warns_once_before_escalating():
    # Exactly one loop -> a nudge, not an escalate: the reasoner recovers (navigates) and completes.
    groq = FakeGroq([reasoner_resp("click_element", '{"index": 1}'),
                     reasoner_resp("click_element", '{"index": 1}'),
                     reasoner_resp("click_element", '{"index": 1}'),    # 3rd identical -> loop -> NUDGE
                     reasoner_resp("navigate", '{"url": "http://test.local/list"}'),  # recovered
                     reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    status, _ = await _run(groq, FakeSession(dom=_dom_payload()), max_steps=12, read_content=False)
    assert status == "complete"


async def test_loose_loop_is_inform_only():
    # Same action NAME with wobbling args (varying index) is now INFORM-ONLY: it nudges the reasoner
    # but does NOT force an escalate (only the exact-repeat backstop does). With the page changing
    # each step it runs to the step budget rather than being cut off. dom_vary isolates it from
    # stagnation.
    events = []
    groq = FakeGroq([reasoner_resp("click_element", f'{{"index": {i}}}') for i in range(8)])
    status, _ = await run_subgoal(groq, FakeSession(dom=_dom_payload(), dom_vary=True), [], SUBGOAL,
                                  allowlist=set(), approve=lambda p: True, ask=lambda q: "",
                                  observations=[], max_steps=8, read_content=False, emit=events.append)
    assert status == "exhausted"
    assert any(e["type"] == "loop_nudge" for e in events)   # warned, not forced to quit


async def test_repeated_rejected_complete_escalates_at_backstop():
    # The reasoner keeps declaring done; the verifier keeps refusing. Each refusal is fed back, but
    # after the high backstop (MAX_COMPLETE_REJECTS) it escalates rather than spinning to the budget.
    scripted = []
    for _ in range(orch.MAX_COMPLETE_REJECTS):
        scripted.append(reasoner_resp("subgoal_complete"))
        scripted.append(content_resp('{"satisfied": false, "reason": "nothing here"}'))
    groq = FakeGroq(scripted)
    status, detail = await _run(groq, FakeSession(), max_steps=orch.MAX_COMPLETE_REJECTS + 4)
    assert status == "escalate" and "refused completion" in detail


async def test_step_budget_exhausted():
    groq = FakeGroq([reasoner_resp("click_element", '{"index": 0}'),
                     reasoner_resp("scroll_page", '{"direction": "down"}')])
    status, _ = await _run(groq, FakeSession(dom=_dom_payload()), max_steps=2, read_content=False)
    assert status == "exhausted"


# ----------------------------------------------------------------- multi-action + page guard
async def test_multi_action_batch_runs_chained_actions_in_one_step():
    # Two chainable inputs + a terminating click, all in ONE turn -> all three execute this step.
    groq = FakeGroq([reasoner_multi([("input_text", '{"index": 0, "text": "a"}'),
                                     ("input_text", '{"index": 1, "text": "b"}'),
                                     ("click_element", '{"index": 2}')]),
                     reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    sess = FakeSession(dom=_dom_payload())
    status, _ = await _run(groq, sess, max_steps=5, read_content=False)
    assert status == "complete"
    fns = [a.get("function", "") for n, a in sess.calls if n == "browser_evaluate"]
    assert sum("HTMLInputElement.prototype" in f for f in fns) == 2   # two input_text setters
    assert sum("el.click()" in f for f in fns) == 1                   # one click


async def test_page_guard_drops_actions_queued_after_a_navigation():
    # navigate is terminating: a click queued after it must NOT run this step (the view is now stale).
    groq = FakeGroq([reasoner_multi([("navigate", '{"url": "http://test.local/list"}'),
                                     ("click_element", '{"index": 0}')]),
                     reasoner_resp("subgoal_complete"),
                     content_resp('{"satisfied": true, "reason": "ok"}')])
    sess = FakeSession(dom=_dom_payload())
    status, _ = await _run(groq, sess, max_steps=5, read_content=False)
    assert status == "complete"
    assert ("browser_navigate", {"url": "http://test.local/list"}) in sess.calls
    fns = [a.get("function", "") for n, a in sess.calls if n == "browser_evaluate"]
    assert not any("el.click()" in f for f in fns)                   # the queued click was dropped


# ----------------------------------------------------------------- robustness (C)
async def test_stagnation_is_inform_only():
    # The page never changes but the agent keeps ACTING with DIFFERENT actions: stagnation now only
    # NUDGES (the LLM decides) — it is NOT force-escalated. It runs to the step budget instead.
    # (FakeSession returns a fixed DOM, so the fingerprint is constant no matter the action.)
    events = []
    acts = [reasoner_resp("click_element", '{"index": 0}') if i % 2 == 0
            else reasoner_resp("select_option", '{"index": 0, "value": "x"}') for i in range(8)]
    status, _ = await run_subgoal(FakeGroq(acts), FakeSession(dom=_dom_payload()), [], SUBGOAL,
                                  allowlist=set(), approve=lambda p: True, ask=lambda q: "",
                                  observations=[], max_steps=8, read_content=False, emit=events.append)
    assert status == "exhausted"
    assert any(e["type"] == "stagnation_nudge" for e in events)


async def test_budget_warning_emitted_near_end():
    events = []
    acts = [reasoner_resp("click_element", '{"index": 0}') if i % 2 == 0
            else reasoner_resp("select_option", '{"index": 0, "value": "x"}') for i in range(6)]
    status, _ = await run_subgoal(FakeGroq(acts), FakeSession(dom=_dom_payload()), [], SUBGOAL,
                                  allowlist=set(), approve=lambda p: True, ask=lambda q: "",
                                  observations=[], max_steps=4, read_content=False, emit=events.append)
    assert status == "exhausted"
    assert any(e["type"] == "budget_warning" and e["step"] == 3 for e in events)  # warn at 75%


async def test_reasoner_decide_includes_plan_context():
    groq = FakeGroq([reasoner_resp("scroll_page", '{"direction": "down"}')])
    await reasoner_decide(groq, [], SUBGOAL, "snap", [], plan_context="Subgoal 2 of 5. Earlier: searched.")
    prompt = groq.calls[0]["messages"][1]["content"]
    assert "Plan progress" in prompt and "Subgoal 2 of 5" in prompt


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
    """A session whose first index-action click opens a new tab — and (like Playwright's real lag)
    leaves its 'current' on the OPENER. Proves the agent follows + activates the new tab. The click
    now runs through browser_evaluate (el.click()), not browser_click."""
    def __init__(self):
        self.tabs = [(0, "http://work/")]
        self.current = 0
        self.calls = []
        self._opened = False

    def _tabs_text(self):
        return "\n".join(f"- {i}: {'(current) ' if i == self.current else ''}[Tab{i}]({u})"
                         for i, u in self.tabs)

    def _dom(self):
        url = dict(self.tabs)[self.current]
        return {"url": url, "title": "t", "scrollY": 0, "scrollHeight": 500, "innerHeight": 900,
                "elements": [{"i": 0, "tag": "a", "type": "", "role": "", "name": "open",
                              "text": "open", "href": "", "inViewport": True}]}

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "browser_tabs" and args.get("action") == "list":
            return NS(content=[NS(type="text", text=self._tabs_text())], isError=False)
        if name == "browser_tabs" and args.get("action") == "select":
            self.current = args["index"]
            return NS(content=[NS(type="text", text="ok")], isError=False)
        if name == "browser_evaluate":
            fn = args.get("function", "")
            if "querySelectorAll('[data-ba-id]')" in fn:            # the buildDomTree pass
                return _eval_result(self._dom())
            if not self._opened:                                    # the click action -> opens a tab
                self._opened = True
                self.tabs.append((1, "http://phone-detail/"))
                return _eval_result({"ok": True, "tag": "a"})
            return _eval_result({"ok": True})
        return NS(content=[NS(type="text", text=f"ok-{name}")], isError=False)


async def test_run_subgoal_follows_new_tab():
    groq = FakeGroq([reasoner_resp("click_element", '{"index": 0}'),
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
