"""Deterministic tests for the DOM-side candidate extractor (agent/dom_extract.py).

No browser: we fake the Playwright MCP `browser_evaluate` result (the `### Result\\n<json>` shape
agent/reader.evaluate_json parses), so the in-page JS is stubbed and we test the Python wrapper:
it maps rows, dedupes, returns [] on garbage (never raises), and `extract_with_scroll` keeps
scrolling until a dry pass or the budget — exactly the lazy-grid loop fix 3 adds.
"""

import json
from types import SimpleNamespace as NS

import browser_agent.agent.dom_extract as dom_extract
from browser_agent.agent.dom_extract import extract_candidates_dom, extract_with_scroll


def eval_result(value):
    """A browser_evaluate tool result as Playwright MCP renders it: value under `### Result`."""
    text = f"### Result\n{json.dumps(value, indent=2)}\n### Ran Playwright code\n```js\n/* … */\n```"
    return NS(content=[NS(type="text", text=text)], isError=False)


class FakeSession:
    """Returns scripted browser_evaluate results in order (one per call)."""
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if not self.results:
            raise AssertionError("unexpected extra browser_evaluate call")
        nxt = self.results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


ROWS = [
    {"name": "Think and Grow Rich", "price": "₹139", "rating": 4.5, "review_count": 6060,
     "url": "https://www.amazon.in/dp/9389931525", "source": "amazon.in"},
    {"name": "Think & Grow Rich", "price": "₹107", "rating": 4.5, "review_count": 21877,
     "url": "https://www.flipkart.com/think-grow-rich/p/itmfc8?pid=978", "source": "flipkart.com"},
]


# ----------------------------------------------------------------- extract_candidates_dom
async def test_parses_structured_rows():
    sess = FakeSession([eval_result(ROWS)])
    out = await extract_candidates_dom(sess)
    assert len(out) == 2
    assert out[0]["name"] == "Think and Grow Rich" and out[0]["rating"] == 4.5
    assert out[0]["price"] == "₹139" and out[0]["review_count"] == 6060
    assert sess.calls[0][0] == "browser_evaluate"


async def test_dedupes_by_url_then_name():
    dup = ROWS + [{"name": "Think and Grow Rich (dup)", "price": "₹140", "url":
                   "https://www.amazon.in/dp/9389931525?ref=xyz", "source": "amazon.in"}]  # same url path
    sess = FakeSession([eval_result(dup)])
    out = await extract_candidates_dom(sess)
    assert len(out) == 2  # the 3rd collapses into the 1st (same /dp/ path, query stripped)


async def test_returns_empty_on_non_list():
    sess = FakeSession([eval_result({"oops": 1})])      # a dict, not a list of rows
    assert await extract_candidates_dom(sess) == []


async def test_returns_empty_on_eval_failure():
    sess = FakeSession([NS(content=[NS(type="text", text="no result section here")], isError=False)])
    assert await extract_candidates_dom(sess) == []     # evaluate_json -> None -> []


async def test_never_raises_on_tool_error():
    sess = FakeSession([RuntimeError("mcp down")])
    assert await extract_candidates_dom(sess) == []     # swallowed -> []


# ----------------------------------------------------------------- extract_with_scroll (fix 3)
async def test_scroll_stops_on_dry_pass(monkeypatch):
    # extract grows (A -> A,B) then a dry pass (still A,B) -> stop. Dedupe across passes.
    pages = [[{"name": "A", "url": "a"}],
             [{"name": "A", "url": "a"}, {"name": "B", "url": "b"}],
             [{"name": "B", "url": "b"}]]
    seen = {"n": 0}

    async def fake_extract(session, spec=None, *, max_items=40):
        i = min(seen["n"], len(pages) - 1); seen["n"] += 1
        return pages[i]

    async def fake_scroll(session, direction="down", amount=None):
        return {"ok": True}

    monkeypatch.setattr(dom_extract, "extract_candidates_dom", fake_extract)
    monkeypatch.setattr("browser_agent.agent.dom_index.scroll_page", fake_scroll)
    monkeypatch.setattr(dom_extract, "_SCROLL_SETTLE_S", 0)

    out = await extract_with_scroll(object(), max_scrolls=5)
    assert {r["name"] for r in out} == {"A", "B"}     # deduped across passes
    assert seen["n"] == 3                              # initial + 2 scrolls, stopped on the dry one


async def test_scroll_respects_budget(monkeypatch):
    # Every pass reveals a NEW row (never dry) -> must stop at the scroll budget, not loop forever.
    seen = {"n": 0}

    async def fake_extract(session, spec=None, *, max_items=40):
        n = seen["n"]; seen["n"] += 1
        return [{"name": f"P{n}", "url": f"u{n}"}]

    async def fake_scroll(session, direction="down", amount=None):
        return {"ok": True}

    monkeypatch.setattr(dom_extract, "extract_candidates_dom", fake_extract)
    monkeypatch.setattr("browser_agent.agent.dom_index.scroll_page", fake_scroll)
    monkeypatch.setattr(dom_extract, "_SCROLL_SETTLE_S", 0)

    out = await extract_with_scroll(object(), max_items=100, max_scrolls=3)
    assert seen["n"] == 4                              # 1 initial + 3 scrolls (budget), then stop
    assert len(out) == 4


# ----------------------------------------------------------------- fix 1: product url + asin per row
async def test_extract_preserves_url_asin_and_addtocart():
    rows = [{"name": "Think and Grow Rich", "price": "₹139",
             "url": "https://www.amazon.in/dp/9389931525", "asin": "9389931525", "add_to_cart": True},
            {"name": "No Link Book", "price": "₹99", "url": None, "asin": None, "add_to_cart": False}]
    sess = FakeSession([eval_result(rows)])
    out = await extract_candidates_dom(sess)
    assert out[0]["url"].endswith("9389931525") and out[0]["asin"] == "9389931525"
    assert out[0]["add_to_cart"] is True
    assert out[1]["url"] is None              # url-less card is FLAGGED (kept), not dropped silently


# ----------------------------------------------------------------- cart confirmation (this round, fix 1)
async def test_cart_confirmed_passes_on_post_add_state():
    # A faked post-add eval result (badge>0 / panel) -> confirmed, WITHOUT navigating to /cart.
    sess = FakeSession([eval_result({"ok": True, "signal": "cart badge 1", "badge": 1})])
    ok, signal = await dom_extract.cart_confirmed(sess, "Think and Grow Rich")
    assert ok is True and "cart badge 1" in signal
    assert all(c[0] == "browser_evaluate" for c in sess.calls)   # never navigated to /cart


async def test_cart_confirmed_false_when_no_signal():
    sess = FakeSession([eval_result({"ok": False, "badge": 0})])
    ok, reason = await dom_extract.cart_confirmed(sess)
    assert ok is False and "confirmation" in reason


async def test_cart_confirmed_never_raises():
    sess = FakeSession([RuntimeError("eval down")])
    ok, _ = await dom_extract.cart_confirmed(sess)
    assert ok is False


# ----------------------------------------------------------------- variant/size selection (this round)
async def test_select_required_variant_maps_result():
    # The in-page JS is faked; we test the wrapper passes the preferred size through and maps the dict.
    sess = FakeSession([eval_result({"ok": True, "kind": "radio", "selected": "UK 9"})])
    out = await dom_extract.select_required_variant(sess, preferred="9")
    assert out == {"ok": True, "kind": "radio", "selected": "UK 9"}
    fn = sess.calls[0][1]["function"]
    assert '"9"' in fn                                   # the preferred size was inlined into the JS


async def test_select_required_variant_none_and_never_raises():
    sess = FakeSession([eval_result({"ok": False, "kind": "none"})])
    assert await dom_extract.select_required_variant(sess) == {"ok": False, "kind": "none"}
    # a tool error degrades to a safe no-op, never raises
    sess2 = FakeSession([RuntimeError("eval down")])
    assert await dom_extract.select_required_variant(sess2) == {"ok": False, "kind": "none"}


# ----------------------------------------------------------------- deterministic checkout (this round)
async def test_proceed_to_checkout_maps_result():
    sess = FakeSession([eval_result({"ok": True, "text": "Proceed to checkout"})])
    out = await dom_extract.proceed_to_checkout(sess)
    assert out == {"ok": True, "text": "Proceed to checkout"}


async def test_at_checkout_true_and_false():
    sess = FakeSession([eval_result({"ok": True, "signal": "checkout url"})])
    reached, signal = await dom_extract.at_checkout(sess)
    assert reached is True and "checkout" in signal
    sess2 = FakeSession([eval_result({"ok": False})])
    reached2, _ = await dom_extract.at_checkout(sess2)
    assert reached2 is False


async def test_checkout_helpers_never_raise():
    assert (await dom_extract.proceed_to_checkout(FakeSession([RuntimeError("x")]))).get("ok") is False
    assert (await dom_extract.at_checkout(FakeSession([RuntimeError("x")])))[0] is False
    assert (await dom_extract.select_cod(FakeSession([RuntimeError("x")]))).get("ok") is False
