"""Content extractor (replaces the old screenshot/vision fallback).

When a page's accessibility snapshot is too sparse to reason over (a JS-heavy app, a canvas, a
content blob with no refs), we turn the LIVE page into clean, LLM-readable text instead of taking a
screenshot. Two stages, both run in the page via the Playwright MCP `browser_evaluate` tool:

  1. PRIMARY  — Mozilla Readability.js (vendored, injected in-page). It reads the live DOM (so it
                works on client-rendered SPAs), strips nav/ads/boilerplate, and returns the article
                title + main text. This is the algorithm behind Firefox Reader Mode.
  2. FALLBACK — if Readability finds no article, grab the page's outerHTML and run it through
                `trafilatura` (Python) to extract the main content as markdown.

Everything is best-effort: any failure (offline, opaque page, eval error) returns None and the
caller just proceeds with the plain snapshot. `browser_evaluate` is otherwise hidden from the
reasoner (see tools.REASONER_EXCLUDED); only this module drives it.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from browser_agent.log import get_logger

log = get_logger(__name__)

# A Readability article shorter than this is treated as "found nothing useful" -> try the fallback.
_MIN_READABILITY_CHARS = 200
# trafilatura markdown shorter than this is treated as a miss -> try the next fallback.
_MIN_TRAFILATURA_CHARS = 80
# Rendered-innerText shorter than this is treated as a miss -> give up (return None).
_MIN_INNERTEXT_CHARS = 120
# Cap article/markdown output so a giant article can't blow the reasoner's context.
_MAX_OUTPUT_CHARS = 8000
# innerText (the last-resort tier) gets a larger cap: it's how we read JS-rendered PRODUCT GRIDS
# (Reliance/Croma/Flipkart) that Readability (needs an article) and trafilatura (needs parseable
# HTML) both miss. The grid's visible text — names + prices — often sits past the first 8k, and the
# explore extractor reads a wide window, so keep more of it.
_MAX_INNERTEXT_CHARS = 30000

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "readability" / "Readability.js"


@lru_cache(maxsize=1)
def _readability_source() -> str:
    """The vendored Readability.js source (read once). Empty string if it's missing."""
    try:
        return _VENDOR.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("vendored Readability.js unreadable (%r); content extractor disabled", e)
        return ""


def _result_section(tool_text: str) -> str | None:
    """Pull the `### Result` section out of a browser_evaluate tool result. Playwright MCP renders
    the evaluated value as `### Result\\n<JSON.stringify(value, null, 2)>` followed by other
    sections (`### Ran Playwright code`, `### Page`, ...). We isolate the Result block so the JSON
    parses cleanly."""
    m = re.search(r"^### Result\s*\n(.*?)(?=^### |\Z)", tool_text, flags=re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else None


def _raw_text(result) -> str:
    """Join an MCP CallToolResult's text blocks WITHOUT the 12k truncation tool_result_to_text
    applies — a page's outerHTML is routinely larger than that and truncation would corrupt the
    JSON string we need to parse."""
    return "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")


async def evaluate_json(session, function: str):
    """Run a JS function in the page via the MCP `browser_evaluate` tool and return its decoded
    value (dict/list/str/None). Playwright JSON-encodes the function's return value into the result
    `### Result` section, so we recover it with json.loads. Never raises (returns None on failure).
    Shared by the content extractor and the web-grounding step."""
    try:
        result = await session.call_tool("browser_evaluate", {"function": function})
    except Exception as e:
        log.debug("browser_evaluate raised: %r", e)
        return None
    section = _result_section(_raw_text(result))
    if not section:
        return None
    try:
        return json.loads(section)
    except (json.JSONDecodeError, TypeError):
        log.debug("could not parse browser_evaluate result section")
        return None


def _readability_function() -> str:
    """JS that inlines Readability.js, parses a CLONE of the live document (so the page isn't
    mutated), and returns {ok, title, byline, siteName, text}. Returned as an OBJECT (not a
    stringified one) so Playwright JSON-encodes it once and `_result_section` recovers it directly.
    """
    source = _readability_source()
    return (
        "() => {\n"
        "  try {\n"
        f"    {source}\n"
        "    const doc = document.cloneNode(true);\n"
        "    const article = new Readability(doc).parse();\n"
        "    if (!article || !article.textContent) return { ok: false };\n"
        "    const text = article.textContent.replace(/[ \\t]+\\n/g, '\\n')"
        ".replace(/\\n{3,}/g, '\\n\\n').trim();\n"
        "    return { ok: true, title: article.title || '', byline: article.byline || '',"
        " siteName: article.siteName || '', text };\n"
        "  } catch (e) { return { ok: false, error: String(e) }; }\n"
        "}"
    )


async def _readability_extract(session) -> str | None:
    """Primary path: inject Readability.js and read the article. Returns formatted text or None."""
    if not _readability_source():
        return None
    val = await evaluate_json(session, _readability_function())
    if not (isinstance(val, dict) and val.get("ok") and val.get("text")):
        if isinstance(val, dict) and val.get("error"):
            log.debug("Readability error in page: %s", str(val["error"])[:120])
        return None
    text = str(val["text"]).strip()
    if len(text) < _MIN_READABILITY_CHARS:
        return None
    title = str(val.get("title") or "").strip()
    out = (f"# {title}\n\n{text}" if title else text)
    log.debug("Readability extracted %d chars (title=%r)", len(text), title[:60])
    return out[:_MAX_OUTPUT_CHARS]


async def _trafilatura_extract(session) -> str | None:
    """Fallback path: grab the page's outerHTML and run trafilatura -> markdown. Returns None if
    trafilatura is unavailable or finds nothing."""
    try:
        import trafilatura  # local import: it's only needed on the fallback path
    except ImportError:
        log.debug("trafilatura not installed; skipping fallback extractor")
        return None
    html = await evaluate_json(session, "() => document.documentElement.outerHTML")
    if not isinstance(html, str) or not html:
        return None
    try:
        md = trafilatura.extract(html, output_format="markdown", include_comments=False,
                                 include_tables=True, favor_recall=True)
    except Exception as e:
        log.debug("trafilatura.extract raised: %r", e)
        return None
    if not md or len(md.strip()) < _MIN_TRAFILATURA_CHARS:
        return None
    log.debug("trafilatura extracted %d chars", len(md))
    return md.strip()[:_MAX_OUTPUT_CHARS]


async def _innertext_extract(session) -> str | None:
    """Last resort: the page's RENDERED visible text (document.body.innerText). Unlike Readability
    (needs an article) and trafilatura (needs parseable static HTML), innerText reflects what the
    browser actually shows after JS runs — so a product grid's names/prices ARE captured. This is
    what lets explore salvage read heavy retail SPAs (the Reliance/Croma 0-candidate dead-ends)."""
    text = await evaluate_json(session, "() => (document.body && document.body.innerText) || ''")
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"[ \t]+\n", "\n", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < _MIN_INNERTEXT_CHARS:
        return None
    log.debug("innerText extracted %d chars", len(cleaned))
    return cleaned[:_MAX_INNERTEXT_CHARS]


async def readable_text(session) -> str | None:
    """Turn the current live page into clean, LLM-readable text, in three tiers:
      1. Readability.js (clean article),
      2. trafilatura over the raw HTML (boilerplate-stripped markdown),
      3. document.body.innerText (the rendered visible text — recovers JS product grids).
    Returns the text, or None if nothing useful could be extracted. Best-effort — never raises."""
    try:
        primary = await _readability_extract(session)
        if primary:
            return primary
        fallback = await _trafilatura_extract(session)
        if fallback:
            return fallback
        return await _innertext_extract(session)
    except Exception as e:  # defensive: a fallback must never take the run down
        log.debug("readable_text failed: %r", e)
        return None
