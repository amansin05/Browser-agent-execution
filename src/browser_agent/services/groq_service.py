"""Groq LLM service — the agent's brain.

Named *_service (not groq.py) so it doesn't shadow the installed `groq` package.
"""

import os

from groq import AsyncGroq

from browser_agent.log import get_logger

log = get_logger(__name__)


def make_groq_client(max_retries: int = 5) -> AsyncGroq:
    """Build an AsyncGroq client from GROQ_API_KEY (loaded from .env by the package init)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        raise SystemExit("GROQ_API_KEY is not set. Put it in .env or the environment.")
    log.debug("created AsyncGroq client (max_retries=%d)", max_retries)
    return AsyncGroq(api_key=api_key, max_retries=max_retries)
