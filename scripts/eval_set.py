r"""
Tiny EVAL SET for the agent (Phase 6).

A handful of answer-checkable tasks against the local test page, run through the flat agent
(`agent_loop`, which returns a final answer). Reuses ONE Playwright MCP server across all tasks
for speed. Prints a per-task PASS/FAIL table and an overall success rate, so you can track
whether prompt/model tweaks help or hurt.

It is non-deterministic (real Scout) — treat the success rate as a metric, not a gate.

Run:
    .\.venv\Scripts\python.exe eval_set.py
"""

import asyncio
import functools
import http.server
import os
import pathlib
import sys
import threading

from groq import AsyncGroq
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from browser_agent.agent.flat import agent_loop
from browser_agent.services.mcp_client import build_server_params, mcp_tools_to_groq

HERE = pathlib.Path(__file__).parent.resolve()


class _Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Q, directory=str(directory)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


# Each task: (name, goal, required-substrings, mode). mode "all" => every substring must appear.
def build_tasks(page1, page2):
    return [
        ("read heading",
         f"Go to {page1} and report the exact text of the main h1 heading.",
         ["MCP Tool Test Page"], "all"),
        ("list dropdown options",
         f"Go to {page1} and list every option in the dropdown menu (id 'dropdown').",
         ["Alpha", "Beta", "Gamma"], "all"),
        ("read paragraph",
         f"Go to {page1} and tell me the text of the paragraph with id 'para'.",
         ["quick brown fox"], "all"),
        ("checkbox state",
         f"Go to {page1}. Is the checkbox (id 'checkbox') currently checked or unchecked? Answer with one word.",
         ["uncheck"], "any"),
        ("link target",
         f"Go to {page1}. The link 'Go to page two' — what page does its href point to?",
         ["test_page2"], "any"),
        ("multi-step navigate",
         f"Go to {page1}, click the link 'Go to page two', then report the main heading of the page you land on.",
         ["Page Two"], "all"),
    ]


def passes(answer: str, expected: list[str], mode: str) -> bool:
    if not answer:
        return False
    low = answer.lower()
    hits = [e for e in expected if e.lower() in low]
    return len(hits) == len(expected) if mode == "all" else len(hits) > 0


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    port = serve(HERE / "pages")
    page1 = f"http://127.0.0.1:{port}/test_page.html"
    page2 = f"http://127.0.0.1:{port}/test_page2.html"
    tasks = build_tasks(page1, page2)

    groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=8)
    results = []

    async with stdio_client(build_server_params(False, "chrome")) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = mcp_tools_to_groq((await session.list_tools()).tools)
            print(f"Eval set: {len(tasks)} tasks (flat agent, shared browser)\n")

            for name, goal, expected, mode in tasks:
                print(f"--- {name} ---")
                try:
                    answer = await agent_loop(groq, session, tools, goal, max_steps=12)
                except Exception as e:
                    answer = None
                    print(f"  (exception: {type(e).__name__}: {e})")
                ok = passes(answer or "", expected, mode)
                results.append((name, ok, (answer or "").replace("\n", " ")[:80]))
                print(f"  => {'PASS' if ok else 'FAIL'}\n")

    n = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 64)
    print(f"EVAL SUCCESS RATE: {n_pass}/{n} = {100 * n_pass // n}%")
    for name, ok, ans in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<22} {ans}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
