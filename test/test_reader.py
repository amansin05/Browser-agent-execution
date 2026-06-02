"""Deterministic tests for the content extractor (agent/reader.py).

No browser: we fake the Playwright MCP `browser_evaluate` tool result, which renders an evaluated
value as `### Result\\n<JSON.stringify(value, null, 2)>` followed by other sections. We assert the
parser isolates that block, that Readability output is preferred, that a too-short article triggers
the trafilatura fallback, and that everything returns None (never raises) when nothing is usable.
"""

import json
from types import SimpleNamespace as NS

import pytest

import browser_agent.agent.reader as reader


# A browser_evaluate tool result as Playwright MCP renders it: the JSON value lives under a
# "### Result" header, with later sections we must ignore.
def eval_result(value, *, with_trailing_sections=True):
    body = json.dumps(value, indent=2)
    text = f"### Result\n{body}"
    if with_trailing_sections:
        text += ("\n### Ran Playwright code\n```js\nawait page.evaluate(/* ... */);\n```\n"
                 "### Page\n- Page URL: http://x/\n- ... snapshot ...")
    return NS(content=[NS(type="text", text=text)], isError=False)


class FakeSession:
    """Returns scripted browser_evaluate results in order (one per call_tool invocation)."""
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


# ----------------------------------------------------------------- result parsing
def test_result_section_isolates_json_block():
    res = eval_result({"ok": True, "title": "T", "text": "body"})
    section = reader._result_section(reader._raw_text(res))
    assert json.loads(section) == {"ok": True, "title": "T", "text": "body"}


def test_result_section_missing_returns_none():
    assert reader._result_section("no result header here") is None


async def test_evaluate_json_decodes_value():
    sess = FakeSession([eval_result(["a", "b"])])
    assert await reader.evaluate_json(sess, "() => ['a','b']") == ["a", "b"]


async def test_evaluate_json_handles_tool_error():
    sess = FakeSession([RuntimeError("mcp down")])
    assert await reader.evaluate_json(sess, "() => 1") is None        # never raises


# ----------------------------------------------------------------- primary: Readability
async def test_readable_text_uses_readability():
    article = "A" * 300            # >= _MIN_READABILITY_CHARS
    sess = FakeSession([eval_result({"ok": True, "title": "Levi's 501", "text": article})])
    out = await reader.readable_text(sess)
    assert out.startswith("# Levi's 501")
    assert article in out
    assert len(sess.calls) == 1    # fallback never needed


async def test_readable_text_falls_back_to_trafilatura(monkeypatch):
    # Readability returns a too-short article -> fall back to trafilatura on the outerHTML.
    short = eval_result({"ok": True, "title": "x", "text": "tiny"})       # < min chars
    html = eval_result("<html><body><article>...</article></body></html>")
    sess = FakeSession([short, html])

    md = "## Extracted\n" + "Clean markdown body that is comfortably longer than the minimum. " * 3
    fake_traf = NS(extract=lambda *a, **k: md)
    monkeypatch.setitem(__import__("sys").modules, "trafilatura", fake_traf)

    out = await reader.readable_text(sess)
    assert out.startswith("## Extracted")
    assert len(sess.calls) == 2    # readability, then outerHTML for the fallback


async def test_readable_text_innertext_fallback(monkeypatch):
    # Readability empty + trafilatura empty -> the rendered innerText tier recovers a product grid.
    notfound = eval_result({"ok": False})
    html = eval_result("<html><body></body></html>")
    grid = eval_result("Samsung M06 5G (Sage Green, 128 GB)  Rs 12,765  4.1 stars  Free delivery\n"
                       "Redmi 13C (Starfrost White, 128 GB)  Rs 9,499  4.0 stars  Bank offer\n"
                       "realme P4 Lite 5G (Mosaic Green, 128 GB)  Rs 14,999  4.3 stars  No cost EMI")
    sess = FakeSession([notfound, html, grid])
    monkeypatch.setitem(__import__("sys").modules, "trafilatura", NS(extract=lambda *a, **k: None))
    out = await reader.readable_text(sess)
    assert out is not None and "Samsung M06 5G" in out and "Rs 12,765" in out
    assert len(sess.calls) == 3    # readability, outerHTML(trafilatura), innerText


async def test_readable_text_none_when_all_three_empty(monkeypatch):
    notfound = eval_result({"ok": False})
    html = eval_result("<html></html>")
    empty_text = eval_result("")                             # innerText also empty
    sess = FakeSession([notfound, html, empty_text])
    monkeypatch.setitem(__import__("sys").modules, "trafilatura", NS(extract=lambda *a, **k: None))
    assert await reader.readable_text(sess) is None


async def test_readable_text_never_raises(monkeypatch):
    # Even if evaluate blows up, the extractor degrades to None rather than taking the run down.
    async def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(reader, "evaluate_json", boom)
    assert await reader.readable_text(FakeSession([])) is None
