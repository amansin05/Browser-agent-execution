r"""
Phase 1 - Hello-world agent.

Proves the full observe -> decide -> act -> verify loop on a non-logged-in site,
with Groq's llama-4-scout as the brain.

Run:
    .\.venv\Scripts\python.exe hello_world.py
"""

import asyncio
import os

from dotenv import load_dotenv

from browser_use import Agent
from browser_use.llm import ChatGroq

# Load GROQ_API_KEY from .env (falls back to the real environment if not present).
load_dotenv()


def build_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            "  1. Get a free key at https://console.groq.com/keys\n"
            "  2. Copy .env.example to .env and paste your key, OR\n"
            "     run:  $env:GROQ_API_KEY = 'gsk_...'  (PowerShell, this session only)"
        )
    # ChatGroq talks to Groq natively - no OpenAI-compat base_url needed in this version.
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=api_key,
        temperature=0.0,  # deterministic-ish; the reasoner should not get creative
    )


async def main() -> None:
    llm = build_llm()

    agent = Agent(
        task=(
            "Go to en.wikipedia.org, search for 'Mars', open the article, "
            "and report the first sentence of the article."
        ),
        llm=llm,
        # --- Phase 2 guardrails, available natively in this browser-use version ---
        loop_detection_enabled=True,   # stop if it spins on the same action with no page change
        max_failures=3,                # give up an action after 3 consecutive failures
        max_actions_per_step=3,        # don't let it batch a huge chain of blind actions
        use_vision=False,              # DOM-first; vision is a Phase 6 fallback
    )

    # max_steps is the hard per-task cap (Phase 2 guardrail).
    history = await agent.run(max_steps=25)

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print(history.final_result())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
