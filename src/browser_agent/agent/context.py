"""Tier-1 basic context (B1): ephemeral facts rebuilt fresh each run and injected into the
planner prompt — time, device, and the active profile. Location/currency are pulled from the
profile persona when known (we don't geolocate)."""

import platform
from datetime import datetime


def build_basic_context(profile=None) -> dict:
    now = datetime.now().astimezone()
    ctx: dict = {
        "time": {"now": now.isoformat(timespec="seconds"), "tz": str(now.tzname() or now.tzinfo),
                 "day": now.strftime("%A")},
        "device": {"type": "desktop", "os": platform.system() or "unknown", "browser": "Chrome"},
    }
    persona = profile.persona() if profile else {}
    if persona.get("location"):
        ctx["location"] = persona["location"]
    ctx["profile"] = {
        "id": getattr(profile, "profile_id", "default"),
        "logged_in_sites": persona.get("logged_in_sites", []),
    }
    return ctx


def context_text(ctx: dict) -> str:
    """Compact, human-readable context block for the planner prompt."""
    lines = ["### Basic context"]
    t = ctx.get("time", {})
    if t:
        lines.append(f"- time: {t.get('now')} ({t.get('day')}, {t.get('tz')})")
    d = ctx.get("device", {})
    if d:
        lines.append(f"- device: {d.get('type')} · {d.get('os')} · {d.get('browser')}")
    loc = ctx.get("location")
    if loc:
        lines.append(f"- location: {loc}")
    p = ctx.get("profile", {})
    if p:
        sites = ", ".join(p.get("logged_in_sites") or []) or "none known"
        lines.append(f"- profile: {p.get('id')} (logged in: {sites})")
    return "\n".join(lines)
