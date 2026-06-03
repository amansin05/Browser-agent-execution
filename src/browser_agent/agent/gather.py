"""Parallel multi-source gathering for explore subgoals.

When a plan has several explore subgoals (one per source), running them one-after-another on the
live browser is slow. This runs them CONCURRENTLY — each on its OWN ephemeral, headless + isolated
Playwright-MCP session, so the workers never pop a window or touch your live Chrome / its profile —
and aggregates the candidates. It trades the sequential path's adaptive re-planning for breadth +
speed: every source is tried at once, and a dead source simply contributes nothing while the others
cover it.

Best-effort throughout: a worker that errors (a browser that won't launch, a bot wall, a parse
miss) yields [] for that source and never takes the run down. NOTE: this drops the live/logged-in
profile for the gather (fresh isolated browsers) — chosen deliberately for speed; act subgoals that
need your real session still run on the live browser.
"""

import asyncio

from browser_agent.agent.dom_extract import extract_with_scroll
from browser_agent.agent.extract import extract_candidates
from browser_agent.agent.reader import readable_text
from browser_agent.config import MODEL
from browser_agent.log import get_logger
from browser_agent.services.mcp_client import full_snapshot, open_session
from browser_agent.utils.text import page_looks_blocked

log = get_logger(__name__)

# Cap how many headless browsers run at once — each is a real Chromium/Chrome launch, so a typical
# 2-4 source plan runs fully concurrently while a larger one queues rather than swamping the machine.
DEFAULT_CONCURRENCY = 4


def _auto_approve(_prompt):
    return True


def _auto_ask(_q, **_kw):
    return ""


async def _gather_one_source(groq, reasoner_tools, sg, *, allowlist, max_steps, model,
                             read_content, browser) -> list[dict]:
    """Open a fresh headless session, drive this explore subgoal to its listings, and extract the
    candidates (with the same Readability salvage the sequential path uses)."""
    # Lazy import: orchestrate imports this module, so importing run_subgoal at top would cycle.
    from browser_agent.agent.orchestrate import run_subgoal
    async with open_session(use_extension=False, browser=browser, headless=True) as (sess, _params):
        await run_subgoal(groq, sess, reasoner_tools, sg, allowlist=allowlist,
                          approve=_auto_approve, ask=_auto_ask, observations=[],
                          max_steps=max_steps, model=model, read_content=read_content, emit=None)
        # DOM-first (deterministic, reads the live grid); a11y/text LLM extraction only as fallback.
        cands = await extract_with_scroll(sess, sg.get("explore_spec"))
        if not cands:
            snap, _url = await full_snapshot(sess)
            cands = await extract_candidates(groq, snap, sg.get("explore_spec"), model=model)
            if not cands and read_content and not page_looks_blocked(snap):
                extra = await readable_text(sess)
                if extra:
                    cands = await extract_candidates(groq, extra, sg.get("explore_spec"), model=model)
        return cands


async def gather_parallel(groq, reasoner_tools, subgoals, *, allowlist, max_steps, model=MODEL,
                          read_content=True, browser="chrome", emit=None,
                          max_concurrency=DEFAULT_CONCURRENCY) -> dict:
    """Run each explore subgoal concurrently on its own headless session; return {id: [candidates]}.
    Emits per-source subgoal_start / candidates / subgoal_end so the UI shows every source. Never
    raises — a failed source maps to []."""
    emit = emit or (lambda e: None)
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def worker(sg):
        sid = sg.get("id")
        emit({"type": "subgoal_start", "id": sid, "goal": sg.get("goal", ""), "kind": "explore",
              "tier": "auto", "success_condition": sg.get("success_condition", ""),
              "needs_approval": False})
        async with sem:
            try:
                cands = await _gather_one_source(
                    groq, reasoner_tools, sg, allowlist=allowlist, max_steps=max_steps,
                    model=model, read_content=read_content, browser=browser)
            except Exception as e:  # a worker (browser launch, eval, parse) failed -> no candidates
                log.warning("parallel source %s failed (%r)", sid, e)
                cands = []
        emit({"type": "candidates", "id": sid, "count": len(cands), "candidates": cands[:10]})
        emit({"type": "subgoal_end", "id": sid,
              "status": "complete" if cands else "escalate",
              "detail": f"gathered {len(cands)} candidates" if cands else "no candidates"})
        return sid, cands

    log.info("parallel gather: %d source(s), concurrency=%d", len(subgoals), max_concurrency)
    results = await asyncio.gather(*(worker(sg) for sg in subgoals))
    return dict(results)
