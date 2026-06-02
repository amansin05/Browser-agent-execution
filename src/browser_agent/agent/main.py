r"""
Two-tier browser agent: PLANNER -> per-subgoal REASONER -> independent VERIFIER, re-planning on
escalation, gated by an approval gate + domain allowlist (Phase 5), with a web-grounded planner and
a content-extractor fallback for opaque pages (Readability.js -> trafilatura; replaced the old
vision fallback), session memory (rolling context + scratchpad), and persistent tabs across tasks.

CLI:
    python -m browser_agent.agent.main "search Wikipedia for Mars and report the first sentence"
    python -m browser_agent.agent.main --extension --allow github.com "star the repo I have open"
    python -m browser_agent.agent.main --session mywork "now scroll down and summarize"  # remembers
"""

import argparse
import asyncio

from browser_agent.agent.orchestrate import run_subgoal  # re-exported for callers/tests
from browser_agent.agent.session import AgentSession
from browser_agent.log import configure_logging, get_logger
from browser_agent.memory import MemorySession, MemoryStore, ProfileMemory
from browser_agent.utils.io import cli_approve, cli_ask, configure_console

__all__ = ["run", "run_subgoal", "AgentSession"]

log = get_logger(__name__)


async def run(goal, *, use_extension=False, browser="chrome", allow=None, pick_tab=False,
              max_steps=12, max_replans=3, approve=cli_approve, ask=cli_ask, read_content=True,
              ground=True, emit=None, memory=None, profile=None) -> str | None:
    """One-shot convenience: open a session, run a single goal, close. For multi-task sessions
    with persistent tabs, use AgentSession directly."""
    configure_logging()  # idempotent — ensure the console + server.log file handlers are attached
    sess = AgentSession(use_extension=use_extension, browser=browser, pick_tab=pick_tab,
                        memory=memory, profile=profile, approve=approve, ask=ask)
    await sess.open()
    try:
        return await sess.run_task(goal, agent="two-tier", allow=allow, max_steps=max_steps,
                                   max_replans=max_replans, read_content=read_content,
                                   ground=ground, emit=emit)
    finally:
        # close the browser; keep memory open if the caller owns it (passed in), else finalize.
        if memory is None:
            await sess.close()
        else:
            await sess.close_browser()


def main() -> None:
    configure_console()
    configure_logging()
    p = argparse.ArgumentParser(description="Two-tier (planner+reasoner+verifier) browser agent")
    p.add_argument("goal", help="natural-language goal")
    p.add_argument("--extension", action="store_true", help="drive your live logged-in Chrome")
    p.add_argument("--browser", default="chrome", help="(smoke mode) browser channel to launch")
    p.add_argument("--allow", default="", help="comma-separated domain allowlist (e.g. github.com,google.com)")
    p.add_argument("--pick-tab", action="store_true", help="(extension) wait for you to pick a tab")
    p.add_argument("--max-steps", type=int, default=12, help="reasoner step budget per subgoal")
    p.add_argument("--max-replans", type=int, default=3, help="re-plan budget")
    p.add_argument("--no-extract", action="store_true",
                   help="disable the content extractor fallback for sparse/opaque pages")
    p.add_argument("--no-grounding", action="store_true",
                   help="skip the pre-planning web search that grounds the planner's source choices")
    p.add_argument("--session", default="", help="persist rolling memory under this session id")
    p.add_argument("--profile", default="default", help="persona/preferences/history profile id")
    args = p.parse_args()

    async def go():
        store = MemoryStore()
        profile = ProfileMemory(store, args.profile)   # personalizes the planner + learns
        memory = MemorySession(store, session_id=args.session) if args.session else None
        sess = AgentSession(use_extension=args.extension, browser=args.browser,
                            pick_tab=args.pick_tab, memory=memory, profile=profile)
        await sess.open()
        try:
            return await sess.run_task(
                args.goal, agent="two-tier",
                allow=args.allow.split(",") if args.allow else None,
                max_steps=args.max_steps, max_replans=args.max_replans,
                read_content=not args.no_extract, ground=not args.no_grounding)
        finally:
            await sess.close()

    result = asyncio.run(go())
    log.info("RESULT: %s", result)


if __name__ == "__main__":
    main()
