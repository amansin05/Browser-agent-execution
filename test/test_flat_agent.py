"""Deterministic tests for the flat agent loop + MCP helpers (no network, no browser)."""

import copy
from types import SimpleNamespace as NS

import httpx
import pytest
from groq import BadRequestError

from browser_agent.agent.flat import agent_loop
from browser_agent.services.mcp_client import mcp_tools_to_groq, parse_tabs, tool_result_to_text


# ----------------------------------------------------------------- fakes
def make_tc(call_id, name, arguments):
    return NS(id=call_id, type="function", function=NS(name=name, arguments=arguments))


def make_resp(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


def make_result(text="ok", is_error=False):
    return NS(content=[NS(type="text", text=text)], isError=is_error)


def make_bad_request(detail: str) -> BadRequestError:
    req = httpx.Request("POST", "https://api.groq.com/v1/chat")
    return BadRequestError(detail, response=httpx.Response(400, request=req),
                           body={"error": {"message": detail}})


class FakeGroq:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

        async def create(**kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            item = self.scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        self.chat = NS(completions=NS(create=create))


class FakeSession:
    def __init__(self, tool_results=None, raise_on=None):
        self.calls = []
        self.tool_results = tool_results or {}
        self.raise_on = set(raise_on or ())

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name in self.raise_on:
            raise RuntimeError("simulated tool failure")
        return self.tool_results.get(name, make_result(f"result-of-{name}"))


def tool_messages(msgs):
    return [m for m in msgs if m.get("role") == "tool"]


# ----------------------------------------------------------------- loop control flow
async def test_terminates_on_final_answer():
    groq = FakeGroq([make_resp(content="all done")])
    session = FakeSession()
    ans = await agent_loop(groq, session, [], "task", max_steps=5)
    assert ans == "all done"
    assert session.calls == []


async def test_executes_tool_then_finishes():
    groq = FakeGroq([
        make_resp(tool_calls=[make_tc("c1", "browser_navigate", '{"url": "http://x"}')]),
        make_resp(content="the heading is Hi"),
    ])
    session = FakeSession(tool_results={"browser_navigate": make_result("navigated")})
    ans = await agent_loop(groq, session, [], "task", max_steps=5)
    assert ans == "the heading is Hi"
    assert session.calls == [("browser_navigate", {"url": "http://x"})]


async def test_tool_result_fed_back():
    groq = FakeGroq([
        make_resp(tool_calls=[make_tc("c1", "browser_navigate", '{"url": "http://x"}')]),
        make_resp(content="done"),
    ])
    session = FakeSession(tool_results={"browser_navigate": make_result("NAV_RESULT_TEXT")})
    await agent_loop(groq, session, [], "task", max_steps=5)
    second = groq.calls[1]["messages"]
    tmsgs = tool_messages(second)
    assert tmsgs and tmsgs[0]["tool_call_id"] == "c1"
    assert "NAV_RESULT_TEXT" in tmsgs[0]["content"]
    assert any("tool_calls" in m for m in second if m["role"] == "assistant")


async def test_bad_request_retry_recovers():
    groq = FakeGroq([
        make_bad_request("tool call validation failed: /boxes expected boolean"),
        make_resp(content="recovered"),
    ])
    ans = await agent_loop(groq, FakeSession(), [], "task", max_steps=5)
    assert ans == "recovered"


async def test_bad_request_gives_up_after_cap():
    groq = FakeGroq([make_bad_request("bad")] * 6)
    with pytest.raises(SystemExit):
        await agent_loop(groq, FakeSession(), [], "task", max_steps=20)
    assert len(groq.calls) == 5


async def test_max_steps_returns_none():
    groq = FakeGroq([make_resp(tool_calls=[make_tc("c", "browser_snapshot", "{}")]) for _ in range(5)])
    session = FakeSession()
    ans = await agent_loop(groq, session, [], "task", max_steps=3)
    assert ans is None
    assert len(session.calls) == 3


async def test_malformed_json_args():
    groq = FakeGroq([
        make_resp(tool_calls=[make_tc("c1", "browser_click", "{not valid json")]),
        make_resp(content="ok done"),
    ])
    session = FakeSession()
    ans = await agent_loop(groq, session, [], "task", max_steps=5)
    assert ans == "ok done"
    assert session.calls == []
    tmsgs = tool_messages(groq.calls[1]["messages"])
    assert tmsgs and "could not parse" in tmsgs[0]["content"]


async def test_tool_exception_handled():
    groq = FakeGroq([
        make_resp(tool_calls=[make_tc("c1", "browser_navigate", '{"url": "http://x"}')]),
        make_resp(content="done despite error"),
    ])
    session = FakeSession(raise_on={"browser_navigate"})
    ans = await agent_loop(groq, session, [], "task", max_steps=5)
    assert ans == "done despite error"
    tmsgs = tool_messages(groq.calls[1]["messages"])
    assert tmsgs and "ERROR calling browser_navigate" in tmsgs[0]["content"]


async def test_multiple_tool_calls_one_step():
    groq = FakeGroq([
        make_resp(tool_calls=[
            make_tc("c1", "browser_snapshot", "{}"),
            make_tc("c2", "browser_evaluate", '{"function": "() => 1"}'),
        ]),
        make_resp(content="both done"),
    ])
    session = FakeSession()
    ans = await agent_loop(groq, session, [], "task", max_steps=5)
    assert ans == "both done"
    assert [c[0] for c in session.calls] == ["browser_snapshot", "browser_evaluate"]
    tmsgs = tool_messages(groq.calls[1]["messages"])
    assert len(tmsgs) == 2 and {t["tool_call_id"] for t in tmsgs} == {"c1", "c2"}


# ----------------------------------------------------------------- mcp helper units
def test_mcp_tools_to_groq():
    fake = [NS(name="browser_click", description="click it", inputSchema={"type": "object"})]
    assert mcp_tools_to_groq(fake) == [{"type": "function", "function": {
        "name": "browser_click", "description": "click it", "parameters": {"type": "object"}}}]


def test_tool_result_to_text():
    assert tool_result_to_text(make_result("hello")) == "hello"
    err = tool_result_to_text(make_result("nope", is_error=True))
    assert err.startswith("ERROR FROM TOOL:") and "nope" in err
    assert tool_result_to_text(NS(content=[], isError=False)) == "(tool returned no content)"


def test_parse_tabs():
    txt = ("- 0: (current) [Welcome](chrome-extension://abc/connect.html)\n"
           "- 1: [GitHub](https://github.com/me)")
    assert parse_tabs(txt) == [(0, True, "chrome-extension://abc/connect.html"),
                               (1, False, "https://github.com/me")]
