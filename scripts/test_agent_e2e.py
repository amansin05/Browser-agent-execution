r"""
LIVE end-to-end smoke test for the full agent (Scout brain + Playwright MCP hands).

Unlike test_agent_loop.py (fakes, deterministic), this makes a REAL Groq call and launches
a REAL browser in smoke mode against a locally-served test page, then asserts the agent's
final answer contains the expected heading. Non-deterministic and slower — run it to confirm
the whole stack works, not on every edit.

Requires GROQ_API_KEY and Node/npx. Run:
    .\.venv\Scripts\python.exe test_agent_e2e.py
"""

import asyncio
import functools
import http.server
import pathlib
import sys
import threading

from browser_agent.agent.flat import run_agent

HERE = pathlib.Path(__file__).parent.resolve()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def start_server(directory: pathlib.Path):
    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _, port = start_server(HERE / "pages")
    page = f"http://127.0.0.1:{port}/test_page.html"
    task = (f"Go to {page} and tell me the exact text of the main h1 heading on the page. "
            "Do not click or type anything.")

    print(f"E2E: running the full agent against {page}\n")
    answer = await run_agent(task, use_extension=False, browser="chrome", max_steps=12)

    expected = "MCP Tool Test Page"
    ok = bool(answer) and expected in answer
    print("\n" + "=" * 60)
    if ok:
        print(f"PASS: agent returned the heading ({expected!r} found in answer).")
    else:
        print(f"FAIL: expected {expected!r} in the answer, got:\n{answer!r}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
