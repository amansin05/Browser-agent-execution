"""In-page prompt overlay: payload/JS building, reply parsing, and the show->poll->cleanup flow."""

import base64
import json
from types import SimpleNamespace as NS

from browser_agent.agent.in_page_prompt import (
    APPROVE, DENY, approve_in_page, ask_in_page, build_inject_js, parse_reply,
)


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _result(text):
    return NS(content=[NS(type="text", text=text)], isError=False)


class FakeSession:
    """Returns scripted browser_evaluate results in order. Each call pops the next script."""
    def __init__(self, scripts, inject_error=False):
        self.scripts = list(scripts)
        self.inject_error = inject_error
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        fn = args["function"]
        if "document.createElement" in fn and "appendChild" in fn:          # the inject script
            return NS(content=[NS(type="text", text="true")], isError=self.inject_error)
        if "__agentReply" in fn and "document.getElementById" in fn:        # cleanup
            return _result("true")
        return _result(self.scripts.pop(0))                                  # a poll


# ----------------------------------------------------------------- pure helpers
def test_build_inject_js_substitutes_payload():
    payload = {"kind": "ask", "question": "Budget?", "options": [], "multi": False}
    js = build_inject_js(payload)
    assert "__PAYLOAD__" not in js
    assert json.dumps(payload) in js  # the exact JSON object is spliced in as `const P = {...}`


def test_parse_reply_pending_and_value():
    assert parse_reply("__PENDING__") is None
    assert parse_reply(f'"<<R>>{_b64("Apple")}<<E>>"') == "Apple"
    assert parse_reply(f"### Result\n<<R>>{_b64('a, b')}<<E>>") == "a, b"
    assert parse_reply(f"<<R>>{_b64('₹40k & up')}<<E>>") == "₹40k & up"


# ----------------------------------------------------------------- ask flow
async def test_ask_in_page_returns_answer_after_polling():
    sess = FakeSession(["__PENDING__", f"<<R>>{_b64('Samsung')}<<E>>"])
    ans = await ask_in_page(sess, "Which brand?", options=[{"id": "s", "label": "Samsung"}],
                            interval=0)
    assert ans == "Samsung"
    assert any(name == "browser_evaluate" for name, _ in sess.calls)


async def test_ask_in_page_skip_sentinel():
    sess = FakeSession([f"<<R>>{_b64('__no_preference__')}<<E>>"])
    ans = await ask_in_page(sess, "Brand?", interval=0)
    assert ans == "__no_preference__"


async def test_ask_in_page_none_when_injection_rejected():
    # chrome-extension:// pages reject injection -> isError -> None so the caller falls back.
    sess = FakeSession([], inject_error=True)
    assert await ask_in_page(sess, "Brand?", interval=0) is None


async def test_ask_in_page_none_on_timeout():
    sess = FakeSession(["__PENDING__", "__PENDING__", "__PENDING__"])
    assert await ask_in_page(sess, "Brand?", timeout=0.05, interval=0.02) is None


# ----------------------------------------------------------------- approve flow
async def test_approve_in_page_true_false():
    yes = FakeSession([f"<<R>>{_b64(APPROVE)}<<E>>"])
    assert await approve_in_page(yes, "Submit the form?", interval=0) is True
    no = FakeSession([f"<<R>>{_b64(DENY)}<<E>>"])
    assert await approve_in_page(no, "Submit the form?", interval=0) is False
