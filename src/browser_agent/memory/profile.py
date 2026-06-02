"""Tier-2 personal memory (B1/B2): persona + preferences + episodic history, persistent across
sessions and keyed by profile id.

- persona: stable, edited only on explicit confirmation.
- preferences: evolving, refined with recency weighting (EMA) — one observation never flips a
  preference, but a sustained trend drifts it; an explicit user correction overrides outright.
- history: append-only episodic records; only a few *relevant* ones are surfaced to the planner.
"""

import json

from browser_agent.log import get_logger

log = get_logger(__name__)

DEFAULT_PERSONA = {
    "name": "", "risk_tolerance": "cautious", "values": [],
    "location": {}, "logged_in_sites": [],
}
DEFAULT_PREFERENCES = {
    "shopping": {"max_price_sensitivity": 0.5, "preferred_brands": [],
                 "delivery_speed_weight": 0.5, "min_rating": 4.0},
    "ui": {"confirm_before_pay": True},
}


def _get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


class ProfileMemory:
    def __init__(self, store, profile_id: str = "default"):
        self.store = store
        self.profile_id = profile_id
        persona, prefs = store.get_profile(profile_id)
        self._persona = persona if persona else dict(DEFAULT_PERSONA)
        self._prefs = prefs if prefs else json.loads(json.dumps(DEFAULT_PREFERENCES))
        if persona is None and prefs is None:
            self._save()

    def _save(self) -> None:
        self.store.save_profile(self.profile_id, self._persona, self._prefs)

    # --- read --------------------------------------------------------------
    def persona(self) -> dict:
        return self._persona

    def preferences(self) -> dict:
        return self._prefs

    def core_blocks_text(self) -> str:
        parts = []
        persona_bits = {k: v for k, v in self._persona.items()
                        if k not in ("location", "logged_in_sites") and v}
        if persona_bits:
            parts.append("### Persona\n" + json.dumps(persona_bits, ensure_ascii=False))
        if self._prefs:
            parts.append("### Preferences (recency-weighted)\n" + json.dumps(self._prefs, ensure_ascii=False))
        return "\n\n".join(parts)

    # --- history -----------------------------------------------------------
    def relevant_history(self, goal: str, limit: int = 3) -> list[dict]:
        """Naive keyword-overlap retrieval — surface only history relevant to this goal."""
        eps = self.store.episodes(self.profile_id, limit=100)
        goal_words = {w for w in _words(goal) if len(w) > 3}
        if not goal_words:
            return eps[-limit:]
        scored = []
        for e in eps:
            overlap = len(goal_words & {w for w in _words(e.get("task", "")) if len(w) > 3})
            if overlap:
                scored.append((overlap, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def history_text(self, goal: str, limit: int = 3) -> str:
        eps = self.relevant_history(goal, limit)
        if not eps:
            return ""
        lines = ["### Relevant past tasks"]
        for e in eps:
            extra = f" (chose {e['chosen']})" if e.get("chosen") else ""
            lines.append(f"- {e.get('task')} -> {e.get('outcome')}{extra}")
        return "\n".join(lines)

    # --- updates (B2) ------------------------------------------------------
    def append_episode(self, task: str, outcome: str, *, chosen=None, rejected=None,
                       on_time=None, meta=None) -> None:
        self.store.add_episode(self.profile_id, task, outcome, chosen=chosen,
                               rejected=rejected, on_time=on_time, meta=meta)

    def refine_preferences(self, updates: dict, alpha: float = 0.3) -> None:
        """Blend observed numeric signals into stored preferences with an exponential moving
        average: new = alpha*observed + (1-alpha)*long_term. Non-numeric signals are ignored
        here (use correct() for those)."""
        changed = False
        for dotted, observed in (updates or {}).items():
            if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                continue
            cur = _get_path(self._prefs, dotted)
            new = alpha * float(observed) + (1 - alpha) * float(cur) if isinstance(cur, (int, float)) else float(observed)
            _set_path(self._prefs, dotted, round(new, 4))
            changed = True
        if changed:
            log.debug("profile %s preferences refined: %s", self.profile_id, list((updates or {}).keys()))
            self._save()

    def correct(self, dotted: str, value) -> None:
        """An explicit user correction overrides outright (outweighs silent observations)."""
        log.info("profile %s correction: %s = %r", self.profile_id, dotted, value)
        _set_path(self._prefs, dotted, value)
        self._save()

    def set_persona(self, key: str, value) -> None:
        """Persona edits are gated by explicit confirmation upstream; this just persists."""
        self._persona[key] = value
        self._save()

    def prune_history(self, keep: int = 200) -> None:
        self.store.prune_episodes(self.profile_id, keep)


def _words(text: str) -> set[str]:
    return {w.strip(".,:;!?'\"()[]").lower() for w in (text or "").split()}
