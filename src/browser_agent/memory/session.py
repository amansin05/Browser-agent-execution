"""MemorySession — ties a session id to rolling context + scratchpad over a store."""

from browser_agent.log import get_logger
from browser_agent.memory.rolling import RollingContext
from browser_agent.memory.scratchpad import Scratchpad

log = get_logger(__name__)


class MemorySession:
    def __init__(self, store, session_id: str | None = None, *, max_turns: int = 12, max_notes: int = 20):
        self.store = store
        self.id = store.create_session(session_id)
        self.rolling = RollingContext(store, self.id, max_turns=max_turns)
        self.scratchpad = Scratchpad(store, self.id, max_notes=max_notes)
        log.debug("memory session %s opened", self.id)

    def preamble(self) -> str:
        """Memory context to prepend to a planning prompt so the agent can follow up on
        earlier tasks in this session."""
        parts = []
        rc = self.rolling.text()
        sp = self.scratchpad.text()
        if rc:
            parts.append("### Recent session history (oldest first)\n" + rc)
        if sp:
            parts.append("### Scratchpad (notes gathered this session)\n" + sp)
        return "\n\n".join(parts)

    def history_messages(self) -> list[dict]:
        """Prior turns as {role, content} for seeding the flat agent's message list."""
        return self.rolling.window()

    def record_task(self, task: str, result: str | None) -> None:
        log.debug("session %s recording task: %.80r", self.id, task)
        self.rolling.add("user", task)
        self.rolling.add("assistant", result or "(no result)")

    def note(self, text: str, key: str | None = None) -> None:
        self.scratchpad.write(text, key)

    def close(self) -> None:
        self.store.end_session(self.id)
