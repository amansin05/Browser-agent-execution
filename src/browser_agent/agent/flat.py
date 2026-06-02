r"""
Flat single-tier agent: Scout picks one Playwright MCP tool per turn until it answers.

The simple baseline. `agent_loop` takes injected dependencies (groq client, MCP session) so it
can be driven by fakes in tests.

CLI:
    python -m browser_agent.agent.flat "go to example.com and read the heading"
    python -m browser_agent.agent.flat --extension "summarize my open GitHub tab"
"""

import argparse
import asyncio
import json

from groq import BadRequestError

from browser_agent.agent.prompts import FLAT_SYSTEM
from browser_agent.agent.tools import flat_tools
from browser_agent.config import MODEL
from browser_agent.log import configure_logging, get_logger
from browser_agent.services.groq_service import make_groq_client
from browser_agent.services.mcp_client import (
    open_session,
    tool_result_to_text,
    wait_for_real_tab,
)
from browser_agent.utils.io import cli_ask, configure_console, maybe_await

log = get_logger(__name__)


async def agent_loop(groq, session, tools: list[dict], task: str, max_steps: int,
                     model: str = MODEL, emit=None, history=None, ask=None):
    """The core observe->decide->act loop. Returns the final answer text, or None if it hit
    max_steps without one. `emit(event_dict)` (optional) streams progress. `history` (optional
    list of {role, content}) seeds prior session turns. `ask` handles the ask_user tool (the
    MCQ modal) when the model needs to disambiguate."""
    emit = emit or (lambda e: None)
    ask = ask or cli_ask
    messages = [{"role": "system", "content": FLAT_SYSTEM}]
    for h in history or []:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": task})

    tool_format_retries = 0
    for step in range(1, max_steps + 1):
        try:
            resp = await groq.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_completion_tokens=1024,
            )
        except BadRequestError as e:
            # Scout's tool-call JSON wobbled and Groq's validation rejected it. Feed the exact
            # error back and let Scout retry, up to a small cap.
            body = getattr(e, "body", None) or {}
            detail = body.get("error", {}).get("message", str(e))
            tool_format_retries += 1
            log.warning("step %d: tool-call rejected (%d): %s", step, tool_format_retries, detail[:160])
            if tool_format_retries > 4:
                raise SystemExit("Too many malformed tool calls from the model; giving up.")
            messages.append({
                "role": "user",
                "content": (
                    f"Your last tool call was rejected by the API: {detail}\n"
                    "Re-issue it with ONLY the parameters you need, using correct JSON "
                    "types (booleans as true/false, numbers unquoted). Omit optional params."
                ),
            })
            continue

        tool_format_retries = 0
        msg = resp.choices[0].message

        # Record the assistant turn (with any tool calls) verbatim.
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # No tool call => Scout is done; this is the final answer.
        if not msg.tool_calls:
            log.info("FINAL ANSWER:\n%s", msg.content)
            emit({"type": "final_answer", "answer": msg.content})
            return msg.content

        # Execute each requested tool call and feed results back.
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result_text = f"ERROR: could not parse tool arguments as JSON: {e}"
                log.warning("step %d: %s(<bad json>) -> %s", step, name, result_text)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                continue

            log.info("step %d: %s(%s)", step, name, json.dumps(raw_args)[:120])
            emit({"type": "step", "step": step, "action": name, "args": raw_args})

            # ask_user pops the MCQ modal instead of hitting the browser.
            if name == "ask_user":
                answer = await maybe_await(ask(raw_args.get("question", ""),
                                               options=raw_args.get("options"),
                                               allow_free_text=raw_args.get("allow_free_text", True),
                                               multi_select=raw_args.get("multi_select", False)))
                result_text = f"User answered: {answer}"
                emit({"type": "action_result", "action": name, "outcome": result_text})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                continue

            try:
                result = await session.call_tool(name, raw_args)
                result_text = tool_result_to_text(result)
            except Exception as e:  # tool/transport failure -> let Scout react
                result_text = f"ERROR calling {name}: {e!r}"
            outcome = result_text[:200].replace(chr(10), " ")
            log.debug("step %d result: %s", step, outcome)
            emit({"type": "action_result", "action": name, "outcome": outcome})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    log.warning("stop: hit max_steps=%d without a final answer.", max_steps)
    return None


async def run_agent(task: str, use_extension: bool, browser: str, max_steps: int,
                    pick_tab: bool = False, emit=None, ask=cli_ask) -> str | None:
    groq = make_groq_client()
    async with open_session(use_extension, browser) as (session, params):
        log.info("setup: launching Playwright MCP: %s %s", params.command, " ".join(params.args))
        listed = await session.list_tools()
        tools = flat_tools(listed.tools)  # excludes browser_close (we keep tabs open)
        log.info("setup: %d browser tools available: %s",
                 len(tools), ", ".join(t.name for t in listed.tools))

        if use_extension and pick_tab:
            await wait_for_real_tab(session)
        elif use_extension:
            log.info("extension: connected. Operating on the controlled tab (your real profile, "
                     "so navigations use your existing logins).")

        return await agent_loop(groq, session, tools, task, max_steps, emit=emit, ask=ask)


def main() -> None:
    configure_console()
    configure_logging()
    parser = argparse.ArgumentParser(description="Flat browser agent: Groq Scout + Playwright MCP")
    parser.add_argument("task", nargs="?", default="Go to https://example.com and tell me the page heading.",
                        help="natural-language task for the agent")
    parser.add_argument("--extension", action="store_true",
                        help="attach to your live logged-in Chrome via the Playwright extension")
    parser.add_argument("--browser", default="chrome",
                        help="(smoke mode only) browser channel for Playwright MCP to launch")
    parser.add_argument("--max-steps", type=int, default=25, help="hard cap on agent steps")
    parser.add_argument("--pick-tab", action="store_true",
                        help="(extension mode) wait for you to choose an existing tab")
    args = parser.parse_args()
    asyncio.run(run_agent(args.task, args.extension, args.browser, args.max_steps, args.pick_tab))


if __name__ == "__main__":
    main()
