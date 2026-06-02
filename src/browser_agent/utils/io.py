"""Console / interaction helpers."""

import asyncio
import sys


def configure_console() -> None:
    """Windows consoles default to cp1252, which crashes on emoji/Unicode the model emits."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cli_approve(prompt: str) -> bool:
    try:
        return input(f"\n[APPROVAL NEEDED] {prompt}\nProceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cli_ask(question: str, options=None, allow_free_text: bool = True,
            multi_select: bool = False) -> str:
    """Disambiguation prompt. Shows selectable options (if any) AND accepts free text — never
    trap the human in buttons (B4). For multi_select, accept several ids (comma-separated)."""
    lines = [f"\n[AGENT ASKS] {question}"]
    for o in options or []:
        lines.append(f"  [{o.get('id')}] {o.get('label')}" + (f" — {o['detail']}" if o.get("detail") else ""))
    if multi_select and options:
        lines.append("  (choose one or more, comma-separated)")
    if allow_free_text:
        lines.append("  (or type a free-text answer)")
    try:
        return input("\n".join(lines) + "\n> ").strip()
    except EOFError:
        return ""


async def maybe_await(value):
    """Allow approve/ask callbacks to be sync (CLI) or async (web round-trip over a socket)."""
    if asyncio.iscoroutine(value):
        return await value
    return value
