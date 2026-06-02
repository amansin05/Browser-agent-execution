r"""
LIVE end-to-end test of the two-tier (planner+reasoner+verifier) agent.

Real Groq + real browser (smoke mode) against a locally-served multi-element page, with a
multi-step read-only goal. Asserts the orchestrator reaches "Done." (every subgoal verified).
Non-deterministic; run to confirm the whole brain works, not on every edit.

Run:
    .\.venv\Scripts\python.exe test_planner_reasoner_e2e.py
"""

import asyncio
import functools
import http.server
import pathlib
import sys
import threading

from browser_agent.agent.main import run

HERE = pathlib.Path(__file__).parent.resolve()


class _Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Q, directory=str(directory)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


async def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    port = serve(HERE / "pages")
    page = f"http://127.0.0.1:{port}/test_page.html"
    goal = (f"Open {page}. Report the exact text of the main heading, and list the options "
            "in the dropdown menu. Read only — do not click buttons, type, or submit anything.")

    def approve(prompt):
        print(f"[auto-approve in test] {prompt}")
        return True

    print(f"E2E: two-tier agent on {page}\n")
    result = await run(goal, use_extension=False, browser="chrome",
                       approve=approve, ask=lambda q: "", max_steps=8, max_replans=2)

    ok = result == "Done."
    print("\n" + "=" * 60)
    print(f"{'PASS' if ok else 'FAIL'}: orchestrator result = {result!r}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
