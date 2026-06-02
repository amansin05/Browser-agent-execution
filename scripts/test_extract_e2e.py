r"""
LIVE end-to-end test of the CONTENT EXTRACTOR (agent/reader.py) — replaced the old vision fallback.

Proves the full extractor pipeline on a real page in a real browser:
  1. navigate to a content-rich article page (nav + ads + footer boilerplate around the article),
  2. readable_text() injects the vendored Readability.js, parses the live DOM, and returns the
     ARTICLE text — with the nav/ads/footer boilerplate stripped out.

Real browser (smoke mode); no Groq needed. Deterministic enough to assert on, but kept as a live
script (it needs Node + Playwright MCP), not a pytest.

Run:
    .\.venv\Scripts\python.exe scripts\test_extract_e2e.py
"""

import asyncio
import functools
import http.server
import pathlib
import sys
import threading

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from browser_agent.agent.reader import readable_text
from browser_agent.services.mcp_client import build_server_params, observe
from browser_agent.utils.text import snapshot_is_sufficient

HERE = pathlib.Path(__file__).parent.resolve()


class _Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Q, directory=str(directory)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    port = serve(HERE / "pages")
    page = f"http://127.0.0.1:{port}/article.html"

    async with stdio_client(build_server_params(False, "chrome")) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("browser_navigate", {"url": page})

            snap, _ = await observe(session)
            print(f"[1] snapshot_is_sufficient = {snapshot_is_sufficient(snap)}")

            text = await readable_text(session)
            print(f"[2] extracted {len(text or '')} chars\n----\n{(text or '')[:600]}\n----")

    low = (text or "").lower()
    got_article = "fit comes first" in low and "where the value is" in low and "levi" in low
    stripped_boilerplate = "sign in" not in low and "subscribe to our newsletter" not in low
    ok = bool(text) and got_article and stripped_boilerplate
    print("\n" + "=" * 60)
    print(f"{'PASS' if ok else 'FAIL'}: article extracted (got_article={got_article}, "
          f"boilerplate_stripped={stripped_boilerplate})")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
