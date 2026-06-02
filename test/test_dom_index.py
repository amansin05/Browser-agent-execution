"""Deterministic tests for the indexed-DOM perception layer (agent/dom_index.py).

No browser: we fake the Playwright MCP `browser_evaluate` result (the `### Result\\n<JSON>` shape,
same as test_reader) and assert (1) the buildDomTree value parses into a DomState, (2) the render
gives a numbered interactive list with a scroll hint and `*`-marks NEW elements, and (3) the
index-addressed actions emit the right in-page JS and parse their result — degrading gracefully
when the element is gone or the eval fails.
"""

import json
from types import SimpleNamespace as NS

import browser_agent.agent.dom_index as di


# A browser_evaluate tool result as Playwright MCP renders it (JSON under a "### Result" header).
def eval_result(value):
    text = (f"### Result\n{json.dumps(value, indent=2)}\n"
            "### Ran Playwright code\n```js\nawait page.evaluate(/* ... */);\n```")
    return NS(content=[NS(type="text", text=text)], isError=False)


class FakeSession:
    """Returns scripted browser_evaluate results in order; records the function string sent."""
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        nxt = self.results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _dom_value(elements, **meta):
    base = {"url": "https://shop.example/search?q=phone", "title": "Phones",
            "scrollY": 0, "scrollHeight": 3000, "innerHeight": 900, "elements": elements}
    base.update(meta)
    return base


SEARCH = {"i": 0, "tag": "input", "type": "search", "name": "Search products", "text": "",
          "role": "", "href": "", "inViewport": True}
BTN = {"i": 1, "tag": "button", "type": "", "name": "", "text": "Search", "role": "", "href": "",
       "inViewport": True}
LINK = {"i": 2, "tag": "a", "type": "", "name": "", "text": "Cart", "role": "", "href": "/cart",
        "inViewport": True}


# ----------------------------------------------------------------- parsing
def test_parse_dom_state_builds_elements_and_map():
    state = di._parse_dom_state(_dom_value([SEARCH, BTN, LINK]))
    assert state is not None
    assert [e.index for e in state.elements] == [0, 1, 2]
    assert state.selector_map[2].href == "/cart"
    assert state.url.endswith("q=phone") and state.title == "Phones"


def test_parse_dom_state_rejects_bad_shape():
    assert di._parse_dom_state(None) is None
    assert di._parse_dom_state({"no_elements": 1}) is None
    assert di._parse_dom_state("a string") is None


# ----------------------------------------------------------------- render
def test_render_numbers_interactive_elements_with_scroll_hint():
    state = di._parse_dom_state(_dom_value([SEARCH, BTN, LINK]))
    out = state.render()
    assert "[0] <input type=search>" in out and "Search products" in out
    assert "[1] <button> \"Search\"" in out
    assert "[2] <a> \"Cart\" -> /cart" in out
    assert "scroll down for more" in out  # 3000px page, 900px viewport, at top


def test_render_marks_new_elements_with_star():
    first = di._parse_dom_state(_dom_value([SEARCH, BTN]))
    # A dropdown/modal opened: a new element appears alongside the existing ones.
    new = {"i": 2, "tag": "button", "name": "Add to cart", "text": "", "role": "", "type": "",
           "href": "", "inViewport": True}
    second = di._parse_dom_state(_dom_value([SEARCH, BTN, new]))
    out = second.render(previous_keys=first.keys)
    add_line = next(ln for ln in out.splitlines() if "Add to cart" in ln)
    search_line = next(ln for ln in out.splitlines() if "[0] <input" in ln)
    assert add_line.startswith("*")        # NEW since last step
    assert not search_line.startswith("*")  # carried over -> not marked


def test_render_handles_empty_page():
    state = di._parse_dom_state(_dom_value([]))
    assert "no interactive elements detected" in state.render()


def test_offscreen_elements_flagged():
    off = {"i": 0, "tag": "button", "name": "Load more", "text": "", "role": "", "type": "",
           "href": "", "inViewport": False}
    out = di._parse_dom_state(_dom_value([off])).render()
    assert "off-screen" in out


def test_render_shows_input_value():
    # A filled input surfaces its current value so the model sees what's already typed.
    el = {"i": 0, "tag": "input", "type": "search", "name": "Search products", "text": "",
          "role": "", "href": "", "value": "iphone 15", "inViewport": True}
    out = di._parse_dom_state(_dom_value([el])).render()
    assert '="iphone 15"' in out and "Search products" in out


# ----------------------------------------------------------------- index_dom (end to end via eval)
async def test_index_dom_reads_buildDomTree_result():
    sess = FakeSession([eval_result(_dom_value([SEARCH, BTN]))])
    state = await di.index_dom(sess)
    assert state is not None and len(state.elements) == 2
    assert sess.calls[0][0] == "browser_evaluate"
    assert "data-ba-id" in sess.calls[0][1]["function"]  # the script clears/sets the id attribute


async def test_index_dom_none_on_eval_failure():
    sess = FakeSession([RuntimeError("mcp down")])
    assert await di.index_dom(sess) is None  # best-effort, never raises


# ----------------------------------------------------------------- index-addressed actions
async def test_click_element_resolves_by_data_ba_id():
    sess = FakeSession([eval_result({"ok": True, "tag": "button", "text": "Search"})])
    res = await di.click_element(sess, 1)
    assert res["ok"] is True and res["tag"] == "button"
    fn = sess.calls[0][1]["function"]
    assert 'data-ba-id="1"' in fn and "el.click()" in fn and "scrollIntoView" in fn


async def test_input_text_uses_native_setter_and_events():
    sess = FakeSession([eval_result({"ok": True, "tag": "input"})])
    res = await di.input_text(sess, 0, "iphone 15")
    assert res["ok"] is True
    fn = sess.calls[0][1]["function"]
    assert 'data-ba-id="0"' in fn
    assert json.dumps("iphone 15") in fn               # text is JSON-encoded into the JS
    assert "HTMLInputElement.prototype" in fn          # native setter (React-safe)
    assert "new Event('input'" in fn and "new Event('change'" in fn


async def test_select_option_matches_value_or_label():
    sess = FakeSession([eval_result({"ok": True, "value": "L"})])
    res = await di.select_option(sess, 3, "Large")
    assert res["ok"] is True and res["value"] == "L"
    assert json.dumps("Large") in sess.calls[0][1]["function"]


async def test_scroll_page_default_and_up():
    sess = FakeSession([eval_result({"ok": True, "scrollY": 765, "scrollHeight": 3000})])
    res = await di.scroll_page(sess)  # default: down ~one viewport
    assert res["ok"] is True and res["scrollY"] == 765
    assert "window.scrollBy" in sess.calls[0][1]["function"]
    sess2 = FakeSession([eval_result({"ok": True, "scrollY": 0, "scrollHeight": 3000})])
    await di.scroll_page(sess2, "up", 500)
    assert "-(500)" in sess2.calls[0][1]["function"]    # up => negative delta


async def test_action_missing_element_reports_not_ok():
    # The element id is stale (page reflowed): the in-page guard returns ok:false, not an exception.
    sess = FakeSession([eval_result({"ok": False, "reason": "element [4] not found — the page changed; re-observe"})])
    res = await di.click_element(sess, 4)
    assert res["ok"] is False and "not found" in res["reason"]


async def test_action_eval_failure_degrades():
    # evaluate_json returns None (eval blew up) -> a structured not-ok dict, never a crash.
    sess = FakeSession([RuntimeError("eval boom")])
    res = await di.click_element(sess, 1)
    assert res["ok"] is False and "eval failed" in res["reason"]
