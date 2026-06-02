"""Rolling-context-window memory: a bounded view over a session's recent messages."""

from browser_agent.log import get_logger

log = get_logger(__name__)

MAX_MESSAGE_CHARS = 4000


class RollingContext:
    def __init__(self, store, session_id: str, max_turns: int = 12, max_chars: int = 4000):
        self.store = store
        self.sid = session_id
        self.max_turns = max_turns
        self.max_chars = max_chars

    def add(self, role: str, content: str) -> None:
        log.debug("rolling[%s] +%s msg (%d chars)", self.sid, role, len(content or ""))
        self.store.add_message(self.sid, role, (content or "")[:MAX_MESSAGE_CHARS])

    def window(self) -> list[dict]:
        """Most-recent messages, capped by turn count AND a rolling char budget."""
        msgs = self.store.recent_messages(self.sid, self.max_turns)
        out: list[dict] = []
        total = 0
        for m in reversed(msgs):  # newest first, keep until the budget is spent
            total += len(m["content"])
            if total > self.max_chars and out:
                break
            out.append(m)
        return list(reversed(out))

    def text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.window())
