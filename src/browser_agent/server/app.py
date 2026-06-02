r"""
WebSocket backend for the browser-agent chat UI (dev-ui/).

Each WebSocket connection is a SESSION: it holds one persistent Playwright MCP browser and one
MemorySession (rolling context + scratchpad in SQLite). After each task the agent's working tabs
are parked — closed, with their URLs saved — and reopened on the next task only if it's a
follow-up (see AgentSession), so the screen isn't left cluttered between unrelated tasks. The
browser is torn down — and any remaining tabs closed, in smoke mode — only when the connection
closes. Agent `emit` events stream to the UI (including tabs_parked / tabs_reopened) and
approval / ask-human prompts round-trip back over the socket.

Run:
    python -m uvicorn browser_agent.server.app:app --port 8000 --reload
    # or: browser-agent-server
"""

import asyncio
import itertools
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from browser_agent.agent.session import AgentSession
from browser_agent.log import configure_logging, get_logger
from browser_agent.memory import MemorySession, MemoryStore, ProfileMemory
from browser_agent.server.logging import EventRecorder

app = FastAPI(title="Browser Agent UI backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

configure_logging()
log = get_logger(__name__)
_ids = itertools.count(1)
_store = MemoryStore()  # one SQLite store (.browser_agent_memory/memory.db) for all sessions


@app.get("/api/health")
async def health():
    return {"ok": True, "groq_key": bool(os.environ.get("GROQ_API_KEY")),
            "agents": ["two-tier", "flat"], "sessions": len(_store.list_sessions())}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    pending: dict[int, asyncio.Future] = {}
    agent_task: asyncio.Task | None = None
    memory = MemorySession(_store)             # rolling context + scratchpad for this connection
    profile = ProfileMemory(_store, "default")  # persistent persona/preferences/history (learns)
    agent_session: AgentSession | None = None
    chat_url: str | None = None                 # the UI tab's URL, to switch back to when asking
    recorder = EventRecorder(memory.id)         # per-connection JSONL trace for debugging
    log.info("ws connected: session=%s -> %s", memory.id, recorder.path)

    def emit(event: dict):
        recorder.record(event)                  # persist the full event for offline debugging
        log.debug("[%s] %s", memory.id, event.get("type"))
        queue.put_nowait(event)

    async def approve(prompt: str) -> bool:
        rid = next(_ids)
        fut = loop.create_future()
        pending[rid] = fut
        emit({"type": "approval_request", "id": rid, "prompt": prompt})
        return bool(await fut)

    async def ask(question: str, options=None, allow_free_text: bool = True,
                  multi_select: bool = False) -> str:
        rid = next(_ids)
        fut = loop.create_future()
        pending[rid] = fut
        emit({"type": "ask_request", "id": rid, "question": question,
              "options": options or [], "allow_free_text": allow_free_text,
              "multi_select": multi_select})
        return str(await fut)

    async def pump():
        while True:
            await websocket.send_json(await queue.get())

    async def run_one(cfg: dict):
        nonlocal agent_session
        agent = cfg.get("agent", "two-tier")
        task = (cfg.get("task") or "").strip()
        pick_tab = bool(cfg.get("pickTab", False))
        log.info("[%s] run started: agent=%s task=%r", memory.id, agent, task[:120])
        emit({"type": "run_started", "agent": agent, "mode": "extension", "task": task})
        try:
            if not task:
                raise ValueError("empty task")
            # Smoke mode removed: the product always drives the live, logged-in Chrome.
            if agent_session is None:
                agent_session = AgentSession(use_extension=True, browser="chrome",
                                             pick_tab=pick_tab, memory=memory, profile=profile,
                                             approve=approve, ask=ask, focus_url=chat_url)
                await agent_session.open()
            allow = [d for d in (cfg.get("allow") or "").split(",") if d.strip()]
            result = await agent_session.run_task(
                task, agent=agent, allow=allow or None, max_steps=int(cfg.get("maxSteps", 12)),
                max_replans=int(cfg.get("maxReplans", 3)),
                # `extract` is the new key; fall back to the legacy `vision` key for old clients.
                read_content=bool(cfg.get("extract", cfg.get("vision", True))),
                ground=bool(cfg.get("grounding", True)),
                # Parallel multi-source explore (headless workers; drops the live profile for the
                # gather). Defaults on; the dev-ui exposes a toggle.
                parallel=bool(cfg.get("parallel", True)),
                approve=approve, ask=ask, emit=emit,
                trace=False)  # the connection already records its own per-connection JSONL trace
            log.info("[%s] run finished: %s", memory.id, str(result)[:120])
            emit({"type": "run_finished", "result": result})
        except asyncio.CancelledError:
            log.info("[%s] run cancelled", memory.id)
            emit({"type": "run_cancelled"})
            raise
        except Exception as e:  # surface any failure to the UI instead of dying silently
            log.exception("[%s] run error", memory.id)
            emit({"type": "run_error", "message": f"{type(e).__name__}: {e}"})

    pump_task = asyncio.create_task(pump())
    emit({"type": "ready", "groq_key": bool(os.environ.get("GROQ_API_KEY")), "session": memory.id})
    try:
        while True:
            msg = await websocket.receive_json()
            kind = msg.get("type")
            if kind == "hello":
                chat_url = msg.get("url")
                if agent_session is not None:
                    agent_session.focus_url = chat_url
            elif kind == "start":
                if agent_task and not agent_task.done():
                    emit({"type": "run_error", "message": "a run is already in progress"})
                else:
                    agent_task = asyncio.create_task(run_one(msg))
            elif kind == "approval_response":
                fut = pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(bool(msg.get("approved")))
            elif kind == "ask_response":
                fut = pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(msg.get("answer", ""))
            elif kind == "cancel":
                if agent_task and not agent_task.done():
                    agent_task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        # Session closed -> NOW close the browser (smoke: closes tabs; extension: disconnects).
        if agent_task and not agent_task.done():
            agent_task.cancel()
        if agent_session is not None:
            try:
                await agent_session.close()
            except Exception:
                pass
        else:
            memory.close()
        recorder.close()
        log.info("ws closed: session=%s", memory.id)
        pump_task.cancel()


def main() -> None:
    """Console-script entry point: `browser-agent-server`."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
