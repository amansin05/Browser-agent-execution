r"""
Drive the browser-agent BACKEND end-to-end as a WebSocket client (the role the dev-ui plays).

Boots the FastAPI server, connects to ws://127.0.0.1:8000/ws, submits a task, auto-answers any
ask/approval prompts, and streams every agent event to stdout — so we can re-test a real run
through the full server pipeline (extension mode → live Chrome) without the React UI.

For ask_user it answers with the "explore broadly" skip sentinel; for any approval it DENIES (a
research/shopping run should stop at the shortlist, never commit a purchase unattended).

Run:
    .\.venv\Scripts\python.exe scripts\drive_backend.py "i want to buy a phone"
"""

import asyncio
import json
import subprocess
import sys
import urllib.request

import websockets

SKIP_SENTINEL = "__no_preference__"
PORT = 8000
HEALTH = f"http://127.0.0.1:{PORT}/api/health"
WS = f"ws://127.0.0.1:{PORT}/ws"
RUN_TIMEOUT = 360.0          # whole-run ceiling (the prior failing run took ~2 min)


def log(*a):
    print(*a, flush=True)


def short(v, n=120):
    s = v if isinstance(v, str) else json.dumps(v)
    return (s[:n] + "…") if len(s) > n else s


async def wait_health(timeout=40.0):
    import time
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH, timeout=3) as r:
                data = json.loads(r.read())
                return data
        except Exception:
            await asyncio.sleep(1.0)
    return None


async def drive(task: str):
    final = {"result": None, "events": 0}
    async with websockets.connect(WS, max_size=2 ** 24) as ws:
        await ws.send(json.dumps({"type": "hello", "url": "http://127.0.0.1:5173/"}))
        cfg = {"type": "start", "task": task, "agent": "two-tier", "allow": "",
               "maxSteps": 12, "maxReplans": 3, "extract": True, "grounding": True, "pickTab": False}
        started = False

        async def loop():
            nonlocal started
            while True:
                msg = json.loads(await ws.recv())
                t = msg.get("type")
                final["events"] += 1
                if t == "ready":
                    log(f"[ready] session={msg.get('session')} groq_key={msg.get('groq_key')}")
                    if not started:
                        log(f"[>>] starting task: {task!r}")
                        await ws.send(json.dumps(cfg)); started = True
                elif t == "run_started":
                    log(f"[run_started] agent={msg.get('agent')} mode={msg.get('mode')}")
                elif t == "grounding":
                    log(f"[grounding] {short(msg.get('text',''), 200)}")
                elif t == "plan":
                    sgs = msg.get("subgoals", [])
                    log(f"[plan] {len(sgs)} subgoals:")
                    for sg in sgs:
                        log(f"    {sg.get('id')}. ({sg.get('type')}) {sg.get('goal')}")
                elif t == "subgoal_start":
                    log(f"  [subgoal {msg.get('id')}] ({msg.get('kind')}/{msg.get('tier')}) {msg.get('goal')}")
                elif t == "step":
                    log(f"    step {msg.get('step')}: {msg.get('action')}({short(msg.get('args'),60)})")
                elif t == "extract":
                    log(f"    [content-extractor fired: {msg.get('chars')} chars]")
                elif t == "action_result":
                    if msg.get("blocked"):
                        log(f"      -> BLOCKED: {short(msg.get('outcome',''),80)}")
                elif t == "verifier":
                    log(f"    [verifier] satisfied={msg.get('satisfied')} ({short(msg.get('reason',''),70)})")
                elif t == "candidates":
                    log(f"  [candidates] +{msg.get('count')} (total {msg.get('total')}) "
                        f"source={msg.get('source')} blocked={msg.get('blocked')}")
                elif t == "exploit":
                    sel = msg.get("selected")
                    log(f"  [exploit] selected={short(sel,80) if sel else None} near_tie={msg.get('near_tie')}")
                elif t == "subgoal_end":
                    log(f"  [subgoal {msg.get('id')} -> {msg.get('status')}] {short(msg.get('detail',''),90)}")
                elif t == "replan":
                    log(f"[replan #{msg.get('n')}] {short(msg.get('reason',''),90)}")
                elif t == "present":
                    log(f"[present] shortlist ({len(msg.get('markdown',''))} chars)")
                elif t == "ask_request":
                    log(f"  [ask] {short(msg.get('question',''),100)} -> answering '(explore broadly)'")
                    await ws.send(json.dumps({"type": "ask_response", "id": msg.get("id"),
                                              "answer": SKIP_SENTINEL}))
                elif t == "approval_request":
                    log(f"  [approval] {short(msg.get('prompt',''),100)} -> DENY (safety)")
                    await ws.send(json.dumps({"type": "approval_response", "id": msg.get("id"),
                                              "approved": False}))
                elif t == "memory_update":
                    log(f"[memory] outcome={msg.get('outcome')}")
                elif t in ("run_finished", "run_error", "run_cancelled"):
                    final["result"] = msg.get("result") or msg.get("message") or t
                    log(f"\n[{t.upper()}] {short(final['result'], 400)}")
                    if msg.get("type") == "present" and msg.get("markdown"):
                        log(msg["markdown"])
                    return

        try:
            await asyncio.wait_for(loop(), timeout=RUN_TIMEOUT)
        except asyncio.TimeoutError:
            final["result"] = f"(client timed out after {RUN_TIMEOUT:.0f}s)"
            log(f"\n[TIMEOUT] no terminal event in {RUN_TIMEOUT:.0f}s")
        try:
            await ws.send(json.dumps({"type": "cancel"}))
        except Exception:
            pass
    return final


async def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "i want to buy a phone"
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log("[boot] starting uvicorn backend on :%d …" % PORT)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "browser_agent.server.app:app", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        health = await wait_health()
        if not health:
            log("[boot] backend did not become healthy — aborting"); return 1
        log(f"[boot] healthy: {health}")
        final = await drive(task)
        log("\n" + "=" * 70)
        log(f"FINAL RESULT: {short(final['result'], 500)}")
        log(f"(events streamed: {final['events']})")
        log("=" * 70)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
