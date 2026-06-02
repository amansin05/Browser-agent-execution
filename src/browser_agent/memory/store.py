"""SQLite-backed session memory store.

One DB (`.browser_agent_memory/memory.db` by default) holds all sessions, partitioned by
session_id: a `sessions` table plus per-session `messages` (rolling context) and `notes`
(scratchpad). The store is sync (SQLite is local + fast); a lock guards the single connection
shared across the server's event loop.
"""

import json
import pathlib
import sqlite3
import threading
import time
import uuid

from browser_agent.log import get_logger

log = get_logger(__name__)

DEFAULT_DIR = ".browser_agent_memory"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, closed_at REAL, meta TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, role TEXT, content TEXT
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, key TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS profile (
  profile_id TEXT PRIMARY KEY, persona TEXT, preferences TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT, ts REAL,
  task TEXT, outcome TEXT, chosen TEXT, rejected TEXT, on_time INTEGER, meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_notes ON notes(session_id, id);
CREATE INDEX IF NOT EXISTS idx_episodes ON episodes(profile_id, id);
"""


class MemoryStore:
    def __init__(self, directory: str | None = None):
        self.dir = pathlib.Path(directory or DEFAULT_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "memory.db"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
        log.debug("memory store opened at %s", self.path)

    # --- sessions -----------------------------------------------------------
    def create_session(self, session_id: str | None = None, meta: dict | None = None) -> str:
        sid = session_id or uuid.uuid4().hex
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at, meta) VALUES(?,?,?,?)",
                (sid, now, now, json.dumps(meta or {})),
            )
        return sid

    def end_session(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE sessions SET closed_at=? WHERE id=?", (time.time(), session_id))

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, created_at, updated_at, closed_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [{"id": i, "created_at": c, "updated_at": u, "closed_at": cl} for i, c, u, cl in rows]

    # --- rolling messages ---------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages(session_id, ts, role, content) VALUES(?,?,?,?)",
                (session_id, time.time(), role, content),
            )
            self._db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (time.time(), session_id))

    def recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # --- scratchpad notes ---------------------------------------------------
    def add_note(self, session_id: str, note: str, key: str | None = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO notes(session_id, ts, key, note) VALUES(?,?,?,?)",
                (session_id, time.time(), key, note),
            )

    def notes(self, session_id: str, limit: int = 30) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT note FROM notes WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [n for (n,) in reversed(rows)]

    # --- profile (Tier-2: persona + preferences, persistent across sessions) ----
    def get_profile(self, profile_id: str):
        with self._lock:
            row = self._db.execute(
                "SELECT persona, preferences FROM profile WHERE profile_id=?", (profile_id,)
            ).fetchone()
        if not row:
            return None, None
        return json.loads(row[0] or "{}"), json.loads(row[1] or "{}")

    def save_profile(self, profile_id: str, persona: dict, preferences: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO profile(profile_id, persona, preferences, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(profile_id) DO UPDATE SET persona=excluded.persona, "
                "preferences=excluded.preferences, updated_at=excluded.updated_at",
                (profile_id, json.dumps(persona), json.dumps(preferences), time.time()),
            )

    # --- episodes (Tier-2: episodic history, append-only) -----------------------
    def add_episode(self, profile_id: str, task: str, outcome: str, *, chosen=None,
                    rejected=None, on_time=None, meta=None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO episodes(profile_id, ts, task, outcome, chosen, rejected, on_time, meta) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (profile_id, time.time(), task, outcome, chosen,
                 json.dumps(rejected or []), None if on_time is None else int(bool(on_time)),
                 json.dumps(meta or {})),
            )

    def episodes(self, profile_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, task, outcome, chosen, rejected, on_time FROM episodes "
                "WHERE profile_id=? ORDER BY id DESC LIMIT ?", (profile_id, limit)
            ).fetchall()
        return [{"ts": ts, "task": t, "outcome": o, "chosen": c,
                 "rejected": json.loads(r or "[]"), "on_time": ot}
                for ts, t, o, c, r, ot in reversed(rows)]

    def prune_episodes(self, profile_id: str, keep: int = 200) -> None:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM episodes WHERE profile_id=? AND id NOT IN "
                "(SELECT id FROM episodes WHERE profile_id=? ORDER BY id DESC LIMIT ?)",
                (profile_id, profile_id, keep),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()
