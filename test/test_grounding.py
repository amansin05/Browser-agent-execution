"""Deterministic tests for the pre-planning web-search grounding (agent/grounding.py).

No browser: a fake session answers browser_navigate / browser_snapshot / browser_evaluate. We
assert the grounding block is built from the top results, that it no-ops when there's nothing to
discover (no session, or the goal already names a URL), and that it degrades to "" on a bot-wall or
a navigation error — grounding must never block planning.
"""

import json
from types import SimpleNamespace as NS

from browser_agent.agent.grounding import web_grounding


def _eval_result(value):
    """A browser_evaluate result in Playwright MCP's `### Result\\n<json>` shape."""
    return NS(content=[NS(type="text", text=f"### Result\n{json.dumps(value, indent=2)}\n### Page\n- x")],
              isError=False)


class GroundSession:
    def __init__(self, *, snapshot_body="- link \"r\" [ref=e1]", results=None, navigate_error=False):
        self.snapshot_body = snapshot_body
        self.results = results if results is not None else []
        self.navigate_error = navigate_error
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "browser_navigate":
            if self.navigate_error:
                raise RuntimeError("net down")
            return NS(content=[NS(type="text", text="ok")], isError=False)
        if name == "browser_snapshot":
            txt = f"### Page\n- Page URL: https://search/\n```yaml\n{self.snapshot_body}\n```"
            return NS(content=[NS(type="text", text=txt)], isError=False)
        if name == "browser_evaluate":
            return _eval_result(self.results)
        return NS(content=[NS(type="text", text="ok")], isError=False)


# ----------------------------------------------------------------- no-op cases
async def test_no_session_returns_empty():
    assert await web_grounding(None, "buy jeans") == ""


async def test_goal_with_explicit_url_is_skipped():
    sess = GroundSession(results=[{"title": "x", "domain": "y.com"}])
    out = await web_grounding(sess, "open https://example.com/page and read the heading")
    assert out == ""
    assert sess.calls == []          # never even navigated — nothing to discover


# ----------------------------------------------------------------- happy path
async def test_builds_grounding_block_from_results():
    results = [{"title": "Levi's 501 Original - Myntra", "domain": "myntra.com"},
               {"title": "Levi's 501 - Ajio", "domain": "ajio.com"}]
    sess = GroundSession(results=results)
    out = await web_grounding(sess, "buy levis 501 jeans")
    assert out.startswith("### Web grounding")
    assert "1. Levi's 501 Original - Myntra — myntra.com" in out
    assert "ajio.com" in out
    # It searched once (DDG returned results) and did not fall through to Bing.
    assert sum(1 for c in sess.calls if c[0] == "browser_navigate") == 1


async def test_falls_through_to_bing_when_ddg_empty():
    # DDG returns nothing -> a second navigate (Bing) is attempted.
    class TwoEngine(GroundSession):
        def __init__(self):
            super().__init__(results=[])
            self._nav = 0

        async def call_tool(self, name, args):
            if name == "browser_evaluate":
                # empty on the first engine, results on the second
                self.results = ([] if self._nav < 2 else [{"title": "Jeans - Myntra", "domain": "myntra.com"}])
            if name == "browser_navigate":
                self._nav += 1
            return await GroundSession.call_tool(self, name, args)

    sess = TwoEngine()
    out = await web_grounding(sess, "buy jeans")
    assert "myntra.com" in out
    assert sum(1 for c in sess.calls if c[0] == "browser_navigate") == 2


# ----------------------------------------------------------------- failure -> "" (never blocks planning)
async def test_bot_wall_yields_empty():
    sess = GroundSession(snapshot_body="Please complete the CAPTCHA to continue",
                         results=[{"title": "x", "domain": "y.com"}])
    out = await web_grounding(sess, "buy jeans")
    assert out == ""
    # A blocked page short-circuits before evaluating results (both engine attempts).
    assert not any(c[0] == "browser_evaluate" for c in sess.calls)


async def test_navigation_error_yields_empty():
    sess = GroundSession(navigate_error=True)
    assert await web_grounding(sess, "buy jeans") == ""
