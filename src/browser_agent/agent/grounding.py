"""Web-search grounding (runs BEFORE planning).

The planner used to pick sources blind — told to "discover sources by searching," it had no real
signal about what sites actually sell the thing, so an "order jeans" goal could get routed toward an
electronics store. This step does a quick live web search of the goal with the SAME browser the
agent already drives, reads the top results' titles + domains, and hands them to the planner as a
grounding block. Now the planner anchors on real, goal-appropriate sources (jeans -> fashion
retailers, not tech stores).

Best-effort by design: any failure (offline, a bot-wall, a parse miss, no session) returns "" and
planning proceeds exactly as before. It also no-ops when the goal already names an explicit URL —
there's nothing to discover.
"""

import re
from urllib.parse import quote_plus

from browser_agent.agent.reader import evaluate_json
from browser_agent.log import get_logger
from browser_agent.services.mcp_client import observe
from browser_agent.utils.text import page_looks_blocked

log = get_logger(__name__)

_URL_IN_GOAL = re.compile(r"https?://", re.IGNORECASE)

# Snapshot-friendly, no-JS search endpoints. DuckDuckGo's html endpoint renders results server-side
# (and wraps each result href in a /l/?uddg=<real-url> redirect — we decode that below); Bing is the
# fallback when DDG throws a bot-wall.
_DDG = "https://html.duckduckgo.com/html/?q={q}"
_BING = "https://www.bing.com/search?q={q}"

# JS run in the results page: collect the top external results as {title, domain}, de-duplicated by
# domain. Decodes DuckDuckGo's uddg redirect, and skips the search engines' own / tracking domains.
_EXTRACT_RESULTS = r"""
(() => {
  const N = %d;
  const SKIP = /(duckduckgo|bing|microsoft|msn|yahoo|google|gstatic|w3\.org|schema\.org)/i;
  const seen = new Set(), out = [];
  for (const a of document.querySelectorAll('a[href]')) {
    let href = a.getAttribute('href') || '';
    let text = (a.textContent || '').trim().replace(/\s+/g, ' ');
    if (href.startsWith('//')) href = 'https:' + href;
    if (!/^https?:/i.test(href)) continue;
    let host;
    try {
      let u = new URL(href);
      if (/duckduckgo/i.test(u.hostname)) {            // unwrap DDG redirect
        const real = u.searchParams.get('uddg');
        if (!real) continue;
        u = new URL(real);
        href = u.href;
      }
      host = u.hostname.replace(/^www\./, '');
    } catch (e) { continue; }
    if (SKIP.test(host)) continue;
    if (text.length < 8) continue;
    if (seen.has(host)) continue;
    seen.add(host);
    out.push({ title: text.slice(0, 140), domain: host });
    if (out.length >= N) break;
  }
  return out;
})
"""


async def _search(session, url: str, max_results: int):
    """Navigate to a search URL and pull the top results. Returns a list of {title, domain} or []
    (also [] when the page is a bot-wall)."""
    try:
        await session.call_tool("browser_navigate", {"url": url})
    except Exception as e:
        log.debug("grounding navigate failed (%s): %r", url, e)
        return []
    snapshot_text, _ = await observe(session)
    if page_looks_blocked(snapshot_text):
        log.debug("grounding: %s looks blocked; will try the next engine", url)
        return []
    results = await evaluate_json(session, _EXTRACT_RESULTS % max_results)
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict) and r.get("domain")]


async def web_grounding(session, goal: str, *, max_results: int = 8) -> str:
    """Search the live web for `goal` and return a compact grounding block for the planner, or ""
    on any failure / when grounding doesn't apply. Best-effort — never raises."""
    if session is None or not goal:
        return ""
    if _URL_IN_GOAL.search(goal):
        # The goal already names where to go (e.g. "open https://… and …") — nothing to discover.
        return ""
    try:
        q = quote_plus(goal[:200])
        results = await _search(session, _DDG.format(q=q), max_results)
        if not results:
            results = await _search(session, _BING.format(q=q), max_results)
        if not results:
            log.debug("web grounding: no results for %r", goal[:80])
            return ""
        lines = [f"### Web grounding (live search for {goal[:120]!r})",
                 "Top results from a quick web search — use these to pick realistic, "
                 "goal-appropriate sources:"]
        for i, r in enumerate(results[:max_results], 1):
            lines.append(f"{i}. {r['title']} — {r['domain']}")
        log.info("web grounding: %d sources for %r", len(results), goal[:60])
        return "\n".join(lines)
    except Exception as e:  # best-effort: planning must proceed even if grounding blows up
        log.debug("web_grounding failed: %r", e)
        return ""
