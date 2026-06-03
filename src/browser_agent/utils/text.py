"""Pure text/JSON helpers used by the agent brain."""

import json
import re


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating ```json fences, <think> blocks, and prose."""
    if text is None:
        raise ValueError("empty model reply")
    t = text.strip()
    # Thinking models (e.g. qwen3, the planner) may prepend a <think>…</think> block. Drop it so
    # the JSON that follows parses cleanly.
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} or [...] block.
        m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


# Phrases that signal the page is a bot-check / captcha / access wall rather than real content.
# Lower-cased substring match against the snapshot. These are the pages that make an explore
# subgoal silently yield zero candidates: the DOM has refs (a form, a button) so it looks
# "sufficient", but there are no listings to read — only a "prove you're human" gate.
_BLOCKED_SIGNALS = (
    "enter the characters you see below",       # amazon captcha
    "type the characters you see",
    "sorry, we just need to make sure you're not a robot",
    "to discuss automated access",              # amazon bot notice
    "automated access",
    "robot check",
    "are you a robot",
    "are you a human",
    "verify you are a human",
    "verify you're a human",
    "our systems have detected unusual traffic",  # google
    "unusual traffic from your computer network",
    "detected unusual traffic",
    "access denied",
    "request blocked",
    "captcha",
    "press & hold",                              # press-and-hold human check
    "complete the security check",
)


def page_looks_blocked(snapshot_text: str) -> str | None:
    """If the snapshot looks like a bot-check / captcha / access-denied wall (rather than real
    page content), return a short human reason; else None. Used by explore to tell a re-plan that
    a source is gated, so it routes to a DIFFERENT source instead of retrying the same wall."""
    low = (snapshot_text or "").lower()
    for sig in _BLOCKED_SIGNALS:
        if sig in low:
            return f"bot-check / access wall (matched {sig!r})"
    return None


# A sign-in / authentication wall. URL path is the strong signal; the body needs BOTH a password cue
# and a sign-in cue so a page that merely has a "Log in" nav link doesn't false-positive.
_LOGIN_URL = re.compile(r"/(ap/signin|signin|sign-in|login|log-in|account/login|auth/|sso|oauth)", re.I)


def page_looks_like_login(snapshot_text: str, url: str = "") -> bool:
    """True if the current page is a sign-in / authentication wall. Used to hand off to the human
    (ask_human) instead of letting the reasoner type fabricated credentials (extension mode is the
    user's already-logged-in browser, so they can authenticate)."""
    if _LOGIN_URL.search((url or "").lower()):
        return True
    low = (snapshot_text or "").lower()
    has_pwd = "password" in low or 'type=password' in low or "type=\"password\"" in low
    has_signin = any(s in low for s in ("sign in", "sign-in", "log in", "log-in", "signin"))
    return has_pwd and has_signin


def snapshot_is_sufficient(snapshot_text: str) -> bool:
    """Heuristic: is the DOM/accessibility snapshot usable on its own? A page that's a canvas,
    a CAPTCHA, or otherwise opaque to the a11y tree yields an empty/near-empty yaml body with no
    element refs — that's when we fall back to the content extractor (agent/reader.py)."""
    m = re.search(r"```yaml\s*(.*?)```", snapshot_text, flags=re.DOTALL)
    body = (m.group(1) if m else snapshot_text).strip()
    if not body:
        return False
    if "[ref=" in body:           # has addressable elements -> usable
        return True
    return len(body) >= 40        # no refs: only trust it if there's real textual substance
