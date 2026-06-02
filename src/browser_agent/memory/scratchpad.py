"""Scratchpad memory: free-form notes the agent gathers during a session."""

from browser_agent.log import get_logger

log = get_logger(__name__)

MAX_NOTE_CHARS = 1000


class Scratchpad:
    def __init__(self, store, session_id: str, max_notes: int = 20):
        self.store = store
        self.sid = session_id
        self.max_notes = max_notes

    def write(self, note: str, key: str | None = None) -> None:
        note = (note or "").strip()
        if note:
            log.debug("scratchpad[%s] note: %.80r", self.sid, note)
            self.store.add_note(self.sid, note[:MAX_NOTE_CHARS], key)

    def notes(self) -> list[str]:
        return self.store.notes(self.sid, self.max_notes)

    def text(self) -> str:
        return "\n".join(f"- {n}" for n in self.notes())
