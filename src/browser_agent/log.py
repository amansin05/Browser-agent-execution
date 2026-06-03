"""Central logging for browser_agent — one logger tree under `browser_agent.*`.

Every module does `log = get_logger(__name__)` and logs through it. `configure_logging()` attaches
the console + file handlers once (the server calls it at startup; the CLI calls it in main). Until
then a NullHandler keeps things silent, so importing the package never prints "No handlers" noise.

Env: BROWSER_AGENT_LOG_DIR (default .browser_agent_memory/logs), BROWSER_AGENT_LOG_LEVEL (INFO).
The stdlib `logging` is imported absolutely, so this module name does not shadow it.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(os.environ.get("BROWSER_AGENT_LOG_DIR", ".browser_agent_memory/logs"))

_ROOT = "browser_agent"
_configured = False

# Silent by default until configure_logging() runs.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())


def get_logger(name: str = "") -> logging.Logger:
    """A child logger under `browser_agent` (e.g. get_logger(__name__) -> browser_agent.orchestrate)."""
    tag = name.rsplit(".", 1)[-1] if name else ""
    return logging.getLogger(f"{_ROOT}.{tag}" if tag else _ROOT)


def configure_logging(level: str | None = None) -> logging.Logger:
    """Attach console + file handlers to the `browser_agent` logger once (idempotent)."""
    global _configured
    logger = logging.getLogger(_ROOT)
    if _configured:
        return logger
    logger.setLevel((level or os.environ.get("BROWSER_AGENT_LOG_LEVEL", "INFO")).upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8", delay=True)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass  # console logging still works without a writable disk
    logger.propagate = False  # handlers live here; children propagate up to us, not to the root
    _configured = True
    return logger


def trace_enabled() -> bool:
    """True when BROWSER_AGENT_LOG_LEVEL is DEBUG or TRACE — the gate for the *deep* per-turn trace
    (subgoal/observation summary/raw args/verifier reasons). Default (INFO) keeps the JSONL stream
    to the high-level events so a normal run's trace stays small; flip the env to get the firehose."""
    return os.environ.get("BROWSER_AGENT_LOG_LEVEL", "INFO").upper() in ("DEBUG", "TRACE")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventRecorder:
    """Append every agent `emit` event (plan, subgoal_start, step, action_result, verifier, replan,
    candidates, run_started/run_finished, …) to a per-run JSONL file under the log dir. That stream
    IS the trace you'd otherwise scrape out of the console to debug a failed run — now on disk, one
    run per file, replayable. Best-effort: any I/O or serialization problem is swallowed so logging
    never breaks a run. Lives here (not in server/) so the CLI/session can record traces without
    depending on the server package."""

    def __init__(self, session_id, log_dir: Path = LOG_DIR):
        self.session_id = session_id
        self.path: Path | None = None
        self._fh = None
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            # Filesystem-safe timestamp (no colons) so it works on Windows too.
            ts = _stamp().replace(":", "").replace(".", "").replace("+", "Z")
            self.path = log_dir / f"run-{session_id}-{ts}.jsonl"
            self._fh = open(self.path, "a", encoding="utf-8")
        except OSError:
            self._fh = None

    def record(self, event: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps({"ts": _stamp(), **event}, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
