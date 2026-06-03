"""Playwright MCP service — the agent's hands.

Named mcp_client (not mcp.py) so it doesn't shadow the installed `mcp` package. Owns the
stdio connection to `npx @playwright/mcp`, schema conversion, and snapshot reads.
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from browser_agent.config import MAX_TOOL_RESULT_CHARS
from browser_agent.log import get_logger

log = get_logger(__name__)


def npx_command() -> str:
    """On Windows the npm shim is npx.cmd; a bare `npx` won't spawn via stdio."""
    return "npx.cmd" if os.name == "nt" else "npx"


def _valid_host(netloc: str) -> bool:
    """A navigable host: a dotted domain or localhost. Rejects bare hosts like `gp` that produce the
    unresolvable `https://gp/product/...` (a scraped relative href that wasn't origin-prefixed)."""
    host = (netloc or "").split("@")[-1].split(":")[0].lower()
    return ("." in host and not host.endswith(".")) or host in ("localhost", "127.0.0.1")


def resolve_url(raw: str, base: str = "") -> str | None:
    """Resolve a navigate target. Returns an absolute http(s) URL, or None if it's malformed/unusable.

    A scraped href is often RELATIVE (`/gp/product/B0…`); navigate must origin-prefix it against the
    current page (urljoin) or it becomes `https://gp/product/…` → ERR_NAME_NOT_RESOLVED. Absolute
    http(s) URLs pass through; a non-http scheme (javascript:/mailto:/…) or a bare/dotless host is
    rejected so the reasoner re-decides instead of navigating somewhere broken."""
    raw = (raw or "").strip()
    if not raw:
        return None
    p = urlparse(raw)
    if p.scheme in ("http", "https"):
        return raw if _valid_host(p.netloc) else None
    if p.scheme:                       # javascript:, mailto:, data:, tel:, about: … — not navigable
        return None
    if base:                           # relative ('/gp/product/…' or 'gp/product/…') -> join to origin
        joined = urljoin(base, raw)
        jp = urlparse(joined)
        if jp.scheme in ("http", "https") and _valid_host(jp.netloc):
            return joined
    return None


def build_server_params(use_extension: bool, browser: str, headless: bool = False) -> StdioServerParameters:
    # `-y` so npx never blocks on an interactive "Ok to proceed?" download prompt.
    args = ["-y", "@playwright/mcp@latest"]
    env = dict(os.environ)  # inherit PATH etc.

    if use_extension:
        args.append("--extension")
        token = os.environ.get("PLAYWRIGHT_MCP_EXTENSION_TOKEN")
        if token:
            env["PLAYWRIGHT_MCP_EXTENSION_TOKEN"] = token
        else:
            log.warning("--extension set but PLAYWRIGHT_MCP_EXTENSION_TOKEN is empty. "
                        "You'll have to approve the connection in the browser each run.")
    else:
        # Smoke mode: reuse the installed Chrome instead of downloading Chromium.
        args += ["--browser", browser]

    if headless:
        # Ephemeral parallel workers (agent/gather.py): run offscreen and ISOLATED (in-memory
        # profile) so they never pop a window or clash with your live Chrome's profile lock.
        args += ["--headless", "--isolated"]

    return StdioServerParameters(command=npx_command(), args=args, env=env)


@asynccontextmanager
async def open_session(use_extension: bool, browser: str, headless: bool = False):
    """Spawn the Playwright MCP server and yield an initialized (session, params)."""
    params = build_server_params(use_extension, browser, headless=headless)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session, params


def mcp_tools_to_groq(tools) -> list[dict]:
    """Convert MCP tool definitions into Groq/OpenAI tool-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "")[:1024],
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def tool_result_to_text(result) -> str:
    """Flatten an MCP CallToolResult's content blocks into text for the model."""
    parts = []
    for block in result.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(block.text)
        elif btype == "image":
            parts.append("[image returned by tool — omitted from text context]")
        else:
            parts.append(str(getattr(block, "text", block)))
    text = "\n".join(parts).strip() or "(tool returned no content)"
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n…[truncated]"
    if getattr(result, "isError", False):
        text = "ERROR FROM TOOL:\n" + text
    return text


_TAB_LINE = re.compile(r"-\s*(\d+):\s*(\(current\))?\s*\[.*?\]\((.*?)\)")


def parse_tabs(text: str):
    """Parse `browser_tabs list` output into (index, is_current, url) tuples."""
    return [
        (int(m.group(1)), bool(m.group(2)), m.group(3))
        for m in (_TAB_LINE.search(ln) for ln in text.splitlines())
        if m
    ]


async def list_tabs_state(session) -> tuple[int | None, list[int]]:
    """(current tab index or None, [all tab indices]). Tolerant: returns (None, []) if the tab
    list can't be read/parsed, so callers degrade to a no-op rather than crashing."""
    try:
        tabs = parse_tabs(tool_result_to_text(
            await session.call_tool("browser_tabs", {"action": "list"})))
    except Exception:
        return None, []
    cur = next((i for i, is_cur, _ in tabs if is_cur), None)
    return cur, [i for i, _, _ in tabs]


def newest_new_tab(before_idxs: list[int], after_idxs: list[int]) -> int | None:
    """If an action opened new tab(s), the index of the newest one — else None. A click that opens
    content in a new tab takes the user's focus there, so the agent should FOLLOW that tab (work on
    what the user is now looking at) rather than keep driving the opener."""
    new = [i for i in after_idxs if i not in before_idxs]
    return max(new) if new else None


async def select_tab(session, idx: int) -> None:
    try:
        await session.call_tool("browser_tabs", {"action": "select", "index": idx})
    except Exception:
        pass


async def wait_for_real_tab(session, max_polls: int = 45) -> bool:
    """In --extension mode the only initial 'tab' is the extension's connect/picker page.
    Your real tabs stay hidden until you choose one in Chrome. Poll until a real http(s)
    tab shows up, select it, and proceed. Returns True if a real tab was attached."""
    log.info("extension: switch to Chrome — the Playwright extension opened a tab PICKER. "
             "Click the (low-stakes, logged-in) tab you want the agent to control "
             "(waiting up to ~90s).")
    for _ in range(max_polls):
        res = await session.call_tool("browser_tabs", {"action": "list"})
        tabs = parse_tabs(tool_result_to_text(res))
        real = [t for t in tabs if t[2].startswith("http")]
        if real:
            idx, is_current, url = real[0]
            if not is_current:
                await session.call_tool("browser_tabs", {"action": "select", "index": idx})
            log.info("extension: attached to tab %d: %s", idx, url[:100])
            return True
        await asyncio.sleep(2)
    log.warning("extension: timed out; proceeding with the current (picker) tab.")
    return False


async def observe(session) -> tuple[str, str]:
    """Take a fresh snapshot (the observation). Returns (snapshot_text, current_url)."""
    res = await session.call_tool("browser_snapshot", {})
    text = tool_result_to_text(res)
    m = re.search(r"Page URL:\s*(\S+)", text)
    return text, (m.group(1) if m else "")


async def full_snapshot(session) -> tuple[str, str]:
    """Like observe(), but WITHOUT the MAX_TOOL_RESULT_CHARS truncation — for candidate EXTRACTION
    on heavy retail pages. Amazon/Flipkart bury the product grid below a huge nav/department/filter
    sidebar, so the first 12k chars (all chrome) is all observe() returns and extraction sees zero
    products (the "i want to buy a phone" → 0-candidates failure on Amazon). Returns the full text."""
    res = await session.call_tool("browser_snapshot", {})
    text = "\n".join(b.text for b in res.content if getattr(b, "type", None) == "text").strip()
    m = re.search(r"Page URL:\s*(\S+)", text)
    return text, (m.group(1) if m else "")
