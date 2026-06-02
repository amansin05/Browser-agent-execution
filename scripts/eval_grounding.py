r"""
GROUNDING eval set — 50 cases, deterministic (no live network).

Grounding has two layers, and we eval both:
  A. JS EXTRACTION (40 cases): the in-page `_EXTRACT_RESULTS` function (DuckDuckGo-redirect decode,
     dedup by domain, junk/search-engine filtering, title trim). We run the REAL JS in Node against
     mocked DOM anchor sets — the same code the browser runs — and assert the extracted
     {title, domain} list.
  B. PY ORCHESTRATION (10 cases): `web_grounding`'s control flow (blocked -> Bing fallback, URL in
     goal skip, no session, empty results, block formatting) via fake MCP sessions.

Run:
    .\.venv\Scripts\python.exe scripts\eval_grounding.py
"""

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from browser_agent.agent.grounding import _EXTRACT_RESULTS, web_grounding

OUT = pathlib.Path(__file__).parent / "eval_results" / "grounding.json"


def ddg(real_url):
    """A DuckDuckGo /html/ result href: a redirect wrapping the real URL in ?uddg=."""
    from urllib.parse import quote
    return f"https://duckduckgo.com/l/?uddg={quote(real_url, safe='')}&rut=abc"


# ----------------------------------------------------------------- A. JS extraction cases (Node)
A_CASES = [
    {"name": "two real results", "anchors": [
        {"href": "https://www.shop-a.com/p", "text": "Great Product A"},
        {"href": "https://shop-b.com/x", "text": "Another Product B"}],
     "domains": ["shop-a.com", "shop-b.com"]},
    {"name": "dedup same domain", "anchors": [
        {"href": "https://shop.com/1", "text": "Product One"},
        {"href": "https://shop.com/2", "text": "Product Two"}],
     "domains": ["shop.com"]},
    {"name": "strip www", "anchors": [{"href": "https://www.myntra.com/jeans", "text": "Jeans on Myntra"}],
     "domains": ["myntra.com"]},
    {"name": "protocol-relative", "anchors": [{"href": "//www.flipkart.com/x", "text": "Flipkart Listing"}],
     "domains": ["flipkart.com"]},
    {"name": "skip bare duckduckgo (no uddg)", "anchors": [
        {"href": "https://duckduckgo.com/about", "text": "About DuckDuckGo"},
        {"href": "https://realstore.com/x", "text": "Real Store Product"}],
     "domains": ["realstore.com"]},
    {"name": "decode uddg redirect", "anchors": [{"href": ddg("https://www.amazon.in/dp/123"), "text": "Amazon India Listing"}],
     "domains": ["amazon.in"]},
    {"name": "skip bing/microsoft", "anchors": [
        {"href": "https://www.bing.com/search?q=x", "text": "Bing Search Results"},
        {"href": "https://go.microsoft.com/x", "text": "Microsoft Link Here"},
        {"href": "https://good.com/x", "text": "Good Result Here"}],
     "domains": ["good.com"]},
    {"name": "skip short link text", "anchors": [
        {"href": "https://a.com/x", "text": "buy"},
        {"href": "https://b.com/x", "text": "Long Enough Title"}],
     "domains": ["b.com"]},
    {"name": "skip javascript: href", "anchors": [
        {"href": "javascript:void(0)", "text": "Click Me Here Now"},
        {"href": "https://c.com/x", "text": "Valid Product Listing"}],
     "domains": ["c.com"]},
    {"name": "skip mailto:", "anchors": [
        {"href": "mailto:hi@x.com", "text": "Email Us Right Now"},
        {"href": "https://d.com/x", "text": "Valid Listing Page"}],
     "domains": ["d.com"]},
    {"name": "skip # anchor", "anchors": [
        {"href": "#section", "text": "Jump To Section Now"},
        {"href": "https://e.com/x", "text": "Valid Listing Page"}],
     "domains": ["e.com"]},
    {"name": "skip relative path", "anchors": [
        {"href": "/category/jeans", "text": "Jeans Category Link"},
        {"href": "https://f.com/x", "text": "Valid Listing Page"}],
     "domains": ["f.com"]},
    {"name": "cap at N=8", "anchors": [
        {"href": f"https://s{i}.com/x", "text": f"Result Number {i} Here"} for i in range(12)],
     "count": 8},
    {"name": "order preserved", "anchors": [
        {"href": "https://one.com/x", "text": "First Result Item"},
        {"href": "https://two.com/x", "text": "Second Result Item"},
        {"href": "https://three.com/x", "text": "Third Result Item"}],
     "domains": ["one.com", "two.com", "three.com"]},
    {"name": "title trimmed to 140", "anchors": [
        {"href": "https://long.com/x", "text": "T" * 300}],
     "domains": ["long.com"], "title_max": 140},
    {"name": "whitespace collapsed", "anchors": [
        {"href": "https://ws.com/x", "text": "Spaced    Out   \n  Title"}],
     "domains": ["ws.com"], "title_has": "Spaced Out Title"},
    {"name": "junk plus good", "anchors": [
        {"href": "javascript:0", "text": "junk junk junk junk"},
        {"href": "#x", "text": "anchor anchor anchor"},
        {"href": "https://gold.com/x", "text": "The Good Result"}],
     "domains": ["gold.com"]},
    {"name": "http allowed", "anchors": [{"href": "http://oldsite.com/x", "text": "Old HTTP Listing"}],
     "domains": ["oldsite.com"]},
    {"name": "ddg link missing uddg -> skip", "anchors": [
        {"href": "https://duckduckgo.com/l/?rut=abc", "text": "Broken Redirect Link"},
        {"href": "https://ok.com/x", "text": "Okay Listing Page"}],
     "domains": ["ok.com"]},
    {"name": "skip google/gstatic", "anchors": [
        {"href": "https://www.google.com/x", "text": "Google Result Link"},
        {"href": "https://gstatic.com/x", "text": "Static Asset Link"},
        {"href": "https://store.com/x", "text": "Store Listing Page"}],
     "domains": ["store.com"]},
    {"name": "dedup www vs non-www", "anchors": [
        {"href": "https://www.shop.com/1", "text": "First Listing Here"},
        {"href": "https://shop.com/2", "text": "Second Listing Here"}],
     "domains": ["shop.com"]},
    {"name": "empty anchors", "anchors": [], "count": 0},
    {"name": "all junk", "anchors": [
        {"href": "javascript:0", "text": "click click click"},
        {"href": "/rel", "text": "relative link here"}],
     "count": 0},
    {"name": "subdomain kept", "anchors": [{"href": "https://m.example.com/x", "text": "Mobile Site Listing"}],
     "domains": ["m.example.com"]},
    {"name": "path/query ignored", "anchors": [{"href": "https://q.com/a/b/c?x=1&y=2#z", "text": "Deep Path Listing"}],
     "domains": ["q.com"]},
    {"name": "uppercase host lowered", "anchors": [{"href": "https://WWW.UPPER.COM/x", "text": "Uppercase Host Listing"}],
     "domains": ["upper.com"]},
    {"name": "skip yahoo", "anchors": [
        {"href": "https://search.yahoo.com/x", "text": "Yahoo Search Result"},
        {"href": "https://realy.com/x", "text": "Real Listing Page"}],
     "domains": ["realy.com"]},
    {"name": "skip w3/schema", "anchors": [
        {"href": "https://www.w3.org/x", "text": "W3 Spec Document Page"},
        {"href": "https://schema.org/x", "text": "Schema Definition Page"},
        {"href": "https://realz.com/x", "text": "Real Listing Page"}],
     "domains": ["realz.com"]},
    {"name": "text exactly 8 chars passes", "anchors": [{"href": "https://eight.com/x", "text": "abcdefgh"}],
     "domains": ["eight.com"]},
    {"name": "many dedup to few", "anchors": [
        {"href": f"https://dup.com/{i}", "text": f"Dup Listing {i}"} for i in range(10)],
     "domains": ["dup.com"]},
    {"name": "two ddg redirects", "anchors": [
        {"href": ddg("https://www.ajio.com/x"), "text": "Ajio Listing Page"},
        {"href": ddg("https://nykaa.com/y"), "text": "Nykaa Listing Page"}],
     "domains": ["ajio.com", "nykaa.com"]},
    {"name": "ddg redirect to skip-domain", "anchors": [
        {"href": ddg("https://www.bing.com/search"), "text": "Bing via Redirect"},
        {"href": "https://realq.com/x", "text": "Real Listing Page"}],
     "domains": ["realq.com"]},
    {"name": "trim spaces in text", "anchors": [{"href": "https://trim.com/x", "text": "   Padded Title Here   "}],
     "domains": ["trim.com"], "title_has": "Padded Title Here"},
    {"name": "host with port", "anchors": [{"href": "https://hostport.com:8443/x", "text": "Port Host Listing"}],
     "domains": ["hostport.com"]},
    {"name": "skip data: href", "anchors": [
        {"href": "data:text/html,xx", "text": "Data URI Link Here"},
        {"href": "https://datok.com/x", "text": "Real Listing Page"}],
     "domains": ["datok.com"]},
    {"name": "skip tel:", "anchors": [
        {"href": "tel:+91999", "text": "Call Us Right Now"},
        {"href": "https://telok.com/x", "text": "Real Listing Page"}],
     "domains": ["telok.com"]},
    {"name": "relative redirect not protocol", "anchors": [
        {"href": "/redirect?u=//evil.com", "text": "Sneaky Relative Link"},
        {"href": "https://safe.com/x", "text": "Safe Listing Page"}],
     "domains": ["safe.com"]},
    {"name": "uppercase scheme", "anchors": [{"href": "HTTPS://capscheme.com/x", "text": "Caps Scheme Listing"}],
     "domains": ["capscheme.com"]},
    {"name": "dup titles diff domains kept", "anchors": [
        {"href": "https://da.com/x", "text": "Identical Title Here"},
        {"href": "https://db.com/x", "text": "Identical Title Here"}],
     "domains": ["da.com", "db.com"]},
    {"name": "10 anchors 4 unique cap", "anchors": [
        {"href": f"https://u{i % 4}.com/{i}", "text": f"Listing Item {i}"} for i in range(10)],
     "domains": ["u0.com", "u1.com", "u2.com", "u3.com"]},
    # Real-world phone-search results page: sponsored + organic + tracking links mixed together.
    {"name": "real phone search results", "anchors": [
        {"href": "https://duckduckgo.com/y.js?ad=1", "text": "Ad - Buy Phones Cheap"},
        {"href": ddg("https://www.amazon.in/mobiles/b?node=1389401031"), "text": "Mobile Phones: Buy Mobiles Online - Amazon.in"},
        {"href": ddg("https://www.flipkart.com/mobiles/pr?sid=tyy,4io"), "text": "Mobiles Online at Best Prices - Flipkart.com"},
        {"href": ddg("https://www.reliancedigital.in/collection/mobiles"), "text": "Buy Mobile Phones Online - Reliance Digital"}],
     "domains": ["amazon.in", "flipkart.com", "reliancedigital.in"]},
    # Tracking/utm params on the real target must not split the domain.
    {"name": "utm params ignored", "anchors": [
        {"href": ddg("https://www.myntra.com/jeans?utm_source=ddg&utm_medium=cpc"), "text": "Jeans Online Shopping - Myntra"}],
     "domains": ["myntra.com"]},
]


def run_js_cases():
    """Run the real _EXTRACT_RESULTS JS in Node against each case's mocked DOM."""
    js_fn = _EXTRACT_RESULTS % 8
    harness = (
        "const JS = " + json.dumps(js_fn) + ";\n"
        "const fn = eval('(' + JS + ')');\n"
        "const cases = " + json.dumps(A_CASES) + ";\n"
        "const out = cases.map(c => {\n"
        "  global.document = { querySelectorAll: () => c.anchors.map(a => ({\n"
        "    getAttribute: (k) => k === 'href' ? a.href : null, textContent: a.text })) };\n"
        "  try { return fn(); } catch (e) { return { __error: String(e) }; }\n"
        "});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(harness)
        path = f.name
    try:
        proc = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed: {proc.stderr[:300]}")
    outputs = json.loads(proc.stdout)

    results = []
    for case, got in zip(A_CASES, outputs):
        domains = [r.get("domain") for r in got] if isinstance(got, list) else None
        ok, notes = True, []
        if domains is None:
            ok, notes = False, [f"js error: {got}"]
        else:
            if "domains" in case and domains != case["domains"]:
                ok = False; notes.append(f"domains {domains} != {case['domains']}")
            if "count" in case and len(got) != case["count"]:
                ok = False; notes.append(f"count {len(got)} != {case['count']}")
            if "title_max" in case and got and len(got[0]["title"]) > case["title_max"]:
                ok = False; notes.append(f"title len {len(got[0]['title'])} > {case['title_max']}")
            if "title_has" in case and (not got or case["title_has"] not in got[0]["title"]):
                ok = False; notes.append(f"title missing {case['title_has']!r}")
        results.append({"name": case["name"], "ok": ok, "got": domains, "notes": notes})
    return results


# ----------------------------------------------------------------- B. Python orchestration cases
def _eval_result(value):
    return NS(content=[NS(type="text", text=f"### Result\n{json.dumps(value, indent=2)}\n### Page\n- x")],
              isError=False)


class FakeSession:
    """browser_evaluate returns scripted result-lists (in order); browser_snapshot returns a
    scripted body (per navigation index, to simulate engine #1 vs #2)."""
    def __init__(self, *, eval_results=None, snapshots=None, navigate_error=False):
        self.eval_results = list(eval_results or [])
        self.snapshots = list(snapshots or ["- link [ref=e1]"])
        self.navigate_error = navigate_error
        self.calls = []
        self._nav = 0

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "browser_navigate":
            if self.navigate_error:
                raise RuntimeError("net down")
            self._nav += 1
            return NS(content=[NS(type="text", text="ok")], isError=False)
        if name == "browser_snapshot":
            body = self.snapshots[min(self._nav - 1, len(self.snapshots) - 1)] if self.snapshots else ""
            return NS(content=[NS(type="text", text=f"### Page\n- Page URL: https://s/\n```yaml\n{body}\n```")],
                      isError=False)
        if name == "browser_evaluate":
            return _eval_result(self.eval_results.pop(0) if self.eval_results else [])
        return NS(content=[NS(type="text", text="ok")], isError=False)


GOOD = [{"title": "Levi's Jeans - Myntra", "domain": "myntra.com"},
        {"title": "Jeans - Ajio", "domain": "ajio.com"}]


async def b_cases():
    out = []

    async def case(name, coro, check):
        try:
            res = await coro
            ok, note = check(res)
        except Exception as e:
            ok, note = False, f"exception {type(e).__name__}: {e}"
        out.append({"name": name, "ok": ok, "note": note})

    # 41 happy path
    s = FakeSession(eval_results=[GOOD])
    await case("happy: builds block", web_grounding(s, "buy jeans"),
               lambda r: ("myntra.com" in r and r.startswith("### Web grounding"), r[:60]))
    # 42 DDG blocked -> Bing returns
    s = FakeSession(eval_results=[GOOD], snapshots=["Please complete the CAPTCHA to continue", "- link [ref=e1]"])
    await case("ddg blocked -> bing", web_grounding(s, "buy jeans"),
               lambda r: ("myntra.com" in r and sum(1 for c in s.calls if c[0] == "browser_navigate") == 2,
                          f"navs={sum(1 for c in s.calls if c[0]=='browser_navigate')}"))
    # 43 both blocked -> ""
    s = FakeSession(eval_results=[GOOD], snapshots=["captcha", "are you a robot"])
    await case("both blocked -> empty", web_grounding(s, "buy jeans"), lambda r: (r == "", repr(r[:40])))
    # 44 empty results both -> ""
    s = FakeSession(eval_results=[[], []])
    await case("empty results -> empty", web_grounding(s, "buy jeans"), lambda r: (r == "", repr(r[:40])))
    # 45 URL in goal -> "" no navigation
    s = FakeSession(eval_results=[GOOD])
    await case("url in goal skipped", web_grounding(s, "open https://x.com and read"),
               lambda r: (r == "" and s.calls == [], f"calls={len(s.calls)}"))
    # 46 None session
    await case("none session -> empty", web_grounding(None, "buy jeans"), lambda r: (r == "", repr(r)))
    # 47 empty goal
    s = FakeSession(eval_results=[GOOD])
    await case("empty goal -> empty", web_grounding(s, ""), lambda r: (r == "", repr(r)))
    # 48 navigate raises
    s = FakeSession(eval_results=[GOOD], navigate_error=True)
    await case("navigate error -> empty", web_grounding(s, "buy jeans"), lambda r: (r == "", repr(r[:40])))
    # 49 block format: numbered "N. title — domain"
    s = FakeSession(eval_results=[GOOD])
    await case("block formatting", web_grounding(s, "buy jeans"),
               lambda r: ("1. Levi's Jeans - Myntra — myntra.com" in r, r.split(chr(10))[2] if r.count(chr(10)) >= 2 else r[:60]))
    # 50 dedup-from-engine already deduped (extractor side) — here just ensure both domains listed
    s = FakeSession(eval_results=[GOOD])
    await case("lists all returned domains", web_grounding(s, "buy jeans"),
               lambda r: ("myntra.com" in r and "ajio.com" in r, r[:60]))
    return out


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    a = run_js_cases()
    b = await b_cases()
    print(f"Grounding eval: {len(a)} JS-extraction (Node) + {len(b)} orchestration (fakes)\n")
    results = ([{"layer": "js", **r} for r in a] + [{"layer": "py", **r} for r in b])

    for r in results:
        tag = "PASS" if r["ok"] else "FAIL"
        detail = "" if r["ok"] else f"  <-- {r.get('notes') or r.get('note')}"
        print(f"  [{tag}] ({r['layer']}) {r['name'][:40]:<42}{detail}")

    n = len(results)
    n_pass = sum(1 for r in results if r["ok"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rate": f"{n_pass}/{n}", "results": results}, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"GROUNDING SUCCESS RATE: {n_pass}/{n} = {100 * n_pass // n}%")
    print(f"Details -> {OUT}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
