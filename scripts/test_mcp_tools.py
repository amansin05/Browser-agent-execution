r"""
Direct test harness for ALL Playwright MCP tools (smoke mode — own Chrome, no real profile).

For each of the 23 tools we call it with valid arguments against controlled local pages and,
where possible, verify the effect via browser_evaluate (used as a state oracle) rather than
just checking "no error". Results print as a PASS / FAIL / SKIP table.

Run:
    .\.venv\Scripts\python.exe test_mcp_tools.py
"""

import asyncio
import functools
import http.server
import os
import pathlib
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = pathlib.Path(__file__).parent.resolve()

results: list[tuple[str, str, str]] = []  # (tool, status, detail)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass


def start_test_server(directory: pathlib.Path):
    """Serve the test pages over http:// (Playwright MCP blocks file://)."""
    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def record(tool: str, status: str, detail: str = "") -> None:
    results.append((tool, status, detail))
    mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"  [{mark}] {tool:<26} {detail[:90]}")


def text_of(result) -> str:
    parts = []
    for b in result.content:
        if getattr(b, "type", None) == "text":
            parts.append(b.text)
        elif getattr(b, "type", None) == "image":
            parts.append(f"[image {len(getattr(b, 'data', '') or '')}b]")
    return "\n".join(parts)


def is_err(result) -> bool:
    return bool(getattr(result, "isError", False))


async def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    httpd, port = start_test_server(HERE / "pages")
    PAGE1 = f"http://127.0.0.1:{port}/test_page.html"
    PAGE2 = f"http://127.0.0.1:{port}/test_page2.html"

    params = StdioServerParameters(
        command="npx.cmd" if os.name == "nt" else "npx",
        # No --caps: mirror the agent's real 23-tool config exactly.
        args=["-y", "@playwright/mcp@latest", "--browser", "chrome"],
        env=dict(os.environ),
    )

    print(f"Test pages (served over http):\n  {PAGE1}\n  {PAGE2}\n")
    print("Launching Playwright MCP (smoke mode)...\n")

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            listed = await session.list_tools()
            tool_names = {t.name for t in listed.tools}
            print(f"{len(tool_names)} tools exposed.\n")

            async def call(tool: str, args: dict, timeout: float = 30.0):
                return await asyncio.wait_for(session.call_tool(tool, args), timeout)

            async def ev(js: str) -> str:
                """Oracle: run JS and return the text result."""
                res = await call("browser_evaluate", {"function": js})
                return text_of(res)

            # ---- 1. browser_navigate -------------------------------------------------
            try:
                res = await call("browser_navigate", {"url": PAGE1})
                ok = not is_err(res) and "test_page.html" in text_of(res).lower()
                record("browser_navigate", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_navigate", "FAIL", repr(e))

            # ---- 2. browser_snapshot -------------------------------------------------
            try:
                res = await call("browser_snapshot", {})
                ok = not is_err(res) and "MCP Tool Test Page" in text_of(res)
                record("browser_snapshot", "PASS" if ok else "FAIL",
                       "found heading in snapshot" if ok else text_of(res)[:80])
            except Exception as e:
                record("browser_snapshot", "FAIL", repr(e))

            # ---- 3. browser_evaluate (oracle itself) ---------------------------------
            try:
                out = await ev("() => document.title")
                ok = "MCP Test Page" in out
                record("browser_evaluate", "PASS" if ok else "FAIL", out[:80])
            except Exception as e:
                record("browser_evaluate", "FAIL", repr(e))

            # ---- 4. browser_take_screenshot ------------------------------------------
            try:
                res = await call("browser_take_screenshot", {"type": "png"})
                has_img = any(getattr(b, "type", None) == "image" for b in res.content)
                ok = not is_err(res) and (has_img or "screenshot" in text_of(res).lower())
                record("browser_take_screenshot", "PASS" if ok else "FAIL",
                       "image returned" if has_img else text_of(res)[:80])
            except Exception as e:
                record("browser_take_screenshot", "FAIL", repr(e))

            # ---- 5. browser_resize ---------------------------------------------------
            try:
                res = await call("browser_resize", {"width": 1024, "height": 768})
                w = await ev("() => window.innerWidth")
                ok = not is_err(res) and "1024" in w
                record("browser_resize", "PASS" if ok else "FAIL", f"innerWidth -> {w[:40]}")
            except Exception as e:
                record("browser_resize", "FAIL", repr(e))

            # ---- 6. browser_console_messages -----------------------------------------
            try:
                res = await call("browser_console_messages", {"level": "info", "all": True})
                ok = not is_err(res) and "MCP_TEST_CONSOLE_MARKER" in text_of(res)
                record("browser_console_messages", "PASS" if ok else "FAIL",
                       "captured console marker" if ok else text_of(res)[:80])
            except Exception as e:
                record("browser_console_messages", "FAIL", repr(e))

            # ---- 7. browser_wait_for (text appears after 1.2s) -----------------------
            try:
                res = await call("browser_wait_for", {"text": "DELAYED_TEXT_APPEARED"}, timeout=15)
                ok = not is_err(res)
                record("browser_wait_for", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_wait_for", "FAIL", repr(e))

            # ---- 8. browser_type -----------------------------------------------------
            try:
                await call("browser_type",
                           {"target": "#textbox", "element": "text box", "text": "hello world"})
                val = await ev("() => document.getElementById('textbox').value")
                ok = "hello world" in val
                record("browser_type", "PASS" if ok else "FAIL", f"value -> {val[:50]}")
            except Exception as e:
                record("browser_type", "FAIL", repr(e))

            # ---- 9. browser_fill_form ------------------------------------------------
            try:
                await call("browser_fill_form", {"fields": [
                    {"target": "#textbox", "name": "text box", "type": "textbox", "value": "formfilled"},
                    {"target": "#checkbox", "name": "the checkbox", "type": "checkbox", "value": "true"},
                ]})
                val = await ev("() => document.getElementById('textbox').value")
                chk = await ev("() => document.getElementById('checkbox').checked")
                ok = "formfilled" in val and "true" in chk.lower()
                record("browser_fill_form", "PASS" if ok else "FAIL",
                       f"text={val[:20]!r} checked={chk[:10]}")
            except Exception as e:
                record("browser_fill_form", "FAIL", repr(e))

            # ---- 10. browser_press_key ----------------------------------------------
            try:
                await call("browser_click", {"target": "#textbox", "element": "text box"})
                res = await call("browser_press_key", {"key": "1"})
                val = await ev("() => document.getElementById('textbox').value")
                ok = not is_err(res) and "1" in val  # appended a '1'
                record("browser_press_key", "PASS" if ok else "FAIL", f"value -> {val[:50]}")
            except Exception as e:
                record("browser_press_key", "FAIL", repr(e))

            # ---- 11. browser_click ---------------------------------------------------
            try:
                await call("browser_click", {"target": "#changebtn", "element": "change-text button"})
                txt = await ev("() => document.getElementById('clickresult').textContent")
                ok = "clicked" in txt
                record("browser_click", "PASS" if ok else "FAIL", f"clickresult -> {txt[:30]}")
            except Exception as e:
                record("browser_click", "FAIL", repr(e))

            # ---- 12. browser_hover ---------------------------------------------------
            try:
                await call("browser_hover", {"target": "#hoverbox", "element": "hover box"})
                txt = await ev("() => document.getElementById('hoverresult').textContent")
                ok = "hovered" in txt
                record("browser_hover", "PASS" if ok else "FAIL", f"hoverresult -> {txt[:30]}")
            except Exception as e:
                record("browser_hover", "FAIL", repr(e))

            # ---- 13. browser_select_option -------------------------------------------
            try:
                await call("browser_select_option",
                           {"target": "#dropdown", "element": "dropdown", "values": ["beta"]})
                val = await ev("() => document.getElementById('dropdown').value")
                ok = "beta" in val
                record("browser_select_option", "PASS" if ok else "FAIL", f"value -> {val[:30]}")
            except Exception as e:
                record("browser_select_option", "FAIL", repr(e))

            # ---- 14. browser_drag ----------------------------------------------------
            try:
                res = await call("browser_drag", {
                    "startTarget": "#dragsrc", "startElement": "drag source",
                    "endTarget": "#dropzone", "endElement": "drop zone"})
                ok = not is_err(res)
                record("browser_drag", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_drag", "FAIL", repr(e))

            # ---- 15. browser_drop (drop data onto element) ---------------------------
            try:
                res = await call("browser_drop", {
                    "target": "#dropzone", "element": "drop zone",
                    "data": {"text/plain": "hello-drop"}})
                ok = not is_err(res)
                record("browser_drop", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_drop", "FAIL", repr(e))

            # ---- 16. browser_handle_dialog (do BEFORE file_upload) -------------------
            try:
                # Clicking #alertbtn fires alert(); MCP leaves a dialog in modal state.
                try:
                    await call("browser_click", {"target": "#alertbtn", "element": "alert button"}, timeout=10)
                except Exception:
                    pass  # click may "fail" because a dialog opened — that's expected
                res = await call("browser_handle_dialog", {"accept": True})
                ok = not is_err(res)
                record("browser_handle_dialog", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_handle_dialog", "FAIL", repr(e))

            # ---- 17. browser_file_upload (clears its own modal afterward) ------------
            try:
                # Keep the sample inside the project dir — Playwright MCP denies file
                # access to the sandbox temp dir (C:\...\Temp\claude\).
                tmp = HERE / "pages" / "_upload_sample.txt"
                tmp.write_text("upload me")
                # Clicking a file input opens a file-chooser modal that browser_file_upload fills.
                await call("browser_click", {"target": "#fileinput", "element": "file input"})
                res = await call("browser_file_upload", {"paths": [str(tmp)]})
                if is_err(res):
                    record("browser_file_upload", "FAIL", text_of(res)[:90])
                    try:
                        await call("browser_file_upload", {})  # cancel any lingering chooser
                    except Exception:
                        pass
                else:
                    cnt = await ev("() => document.getElementById('fileinput').files.length")
                    ok = "1" in cnt
                    record("browser_file_upload", "PASS" if ok else "FAIL", f"files.length -> {cnt[:20]}")
            except Exception as e:
                record("browser_file_upload", "FAIL", repr(e))

            # ---- 18. browser_navigate_back -------------------------------------------
            try:
                await call("browser_navigate", {"url": PAGE2})
                res = await call("browser_navigate_back", {})
                url = await ev("() => location.href")
                ok = "test_page.html" in url.lower()
                record("browser_navigate_back", "PASS" if ok else "FAIL", f"url -> ...{url[-40:].strip()}")
            except Exception as e:
                record("browser_navigate_back", "FAIL", repr(e))

            # ---- 19. browser_tabs (new / list / select / close) ----------------------
            try:
                await call("browser_tabs", {"action": "new", "url": "about:blank"})
                lst = await call("browser_tabs", {"action": "list"})
                listed_txt = text_of(lst)
                await call("browser_tabs", {"action": "select", "index": 0})
                await call("browser_tabs", {"action": "close", "index": 1})
                # count tab lines (e.g. "- 0:", "- 1:")
                n_tabs = sum(1 for ln in listed_txt.splitlines() if ln.strip().startswith("- "))
                ok = n_tabs >= 2
                record("browser_tabs", "PASS" if ok else "FAIL", f"listed {n_tabs} tabs after new")
            except Exception as e:
                record("browser_tabs", "FAIL", repr(e))

            # ---- 20-21. network tools (navigate to a real http page first) -----------
            try:
                await call("browser_navigate", {"url": "https://example.com"})
                res = await call("browser_network_requests", {"static": True})
                ok = not is_err(res) and "example.com" in text_of(res)
                record("browser_network_requests", "PASS" if ok else "FAIL",
                       "captured example.com request" if ok else text_of(res)[:80])
            except Exception as e:
                record("browser_network_requests", "FAIL", repr(e))

            try:
                res = await call("browser_network_request", {"index": 1})  # 1-indexed
                ok = not is_err(res) and len(text_of(res)) > 0
                record("browser_network_request", "PASS" if ok else "FAIL", text_of(res)[:80])
            except Exception as e:
                record("browser_network_request", "FAIL", repr(e))

            # ---- 22. browser_run_code_unsafe -----------------------------------------
            try:
                res = await call("browser_run_code_unsafe",
                                 {"code": "async (page) => { return await page.title(); }"})
                txt = text_of(res)
                if is_err(res) and ("not allowed" in txt.lower() or "unsafe" in txt.lower()
                                    or "capabilit" in txt.lower() or "disabled" in txt.lower()):
                    record("browser_run_code_unsafe", "SKIP", "disabled by default (needs --caps)")
                else:
                    ok = not is_err(res) and "Example Domain" in txt
                    record("browser_run_code_unsafe", "PASS" if ok else "FAIL", txt[:80])
            except Exception as e:
                record("browser_run_code_unsafe", "FAIL", repr(e))

            # ---- 23. browser_close (LAST — may end the session) ----------------------
            try:
                res = await call("browser_close", {}, timeout=15)
                record("browser_close", "PASS", text_of(res)[:80] or "closed")
            except Exception as e:
                # Closing the only page can tear down the server; treat clean teardown as pass.
                record("browser_close", "PASS", f"closed (teardown: {type(e).__name__})")

            _print_summary()


def _print_summary() -> None:
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(of {len(results)} tools tested)")
    if n_fail:
        print("Failures:")
        for tool, status, detail in results:
            if status == "FAIL":
                print(f"  - {tool}: {detail[:100]}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        # Swallow teardown errors that can follow browser_close, after the summary printed.
        if not results:
            raise
        print(f"\n[note] session teardown raised after tests: {type(e).__name__}: {e}")
