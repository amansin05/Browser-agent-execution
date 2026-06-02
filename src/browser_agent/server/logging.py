"""Backward-compat shim: `EventRecorder`, `configure_logging`, and `LOG_DIR` now live in
`browser_agent.log` so non-server code (the CLI/session) can record traces without depending on
the server package. Re-exported here because existing imports point at this module."""

from browser_agent.log import LOG_DIR, EventRecorder, configure_logging

__all__ = ["EventRecorder", "configure_logging", "LOG_DIR"]
