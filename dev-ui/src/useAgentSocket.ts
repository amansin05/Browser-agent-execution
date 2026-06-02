import { useCallback, useEffect, useReducer, useRef } from "react";
import type { AgentEvent, ConnState, Run, Settings, SubgoalCard, Step } from "./types";

const WS_URL = (import.meta.env.VITE_WS_URL as string) || "ws://localhost:8000/ws";

interface State {
  conn: ConnState;
  groqKey: boolean | null;
  runs: Run[];
}

type Action =
  | { kind: "conn"; conn: ConnState }
  | { kind: "newRun"; run: Run }
  | { kind: "event"; ev: AgentEvent }
  | { kind: "clearApproval" }
  | { kind: "clearAsk" }
  | { kind: "reset" };

// Find the subgoal card that steps/results should attach to. For the flat agent (which emits
// no plan/subgoal events) we lazily create one implicit container.
function lastSubgoal(run: Run): SubgoalCard | null {
  for (let i = run.items.length - 1; i >= 0; i--) {
    if (run.items[i].kind === "subgoal") return run.items[i] as SubgoalCard;
  }
  return null;
}

function ensureContainer(run: Run): SubgoalCard {
  let sg = lastSubgoal(run);
  if (!sg) {
    sg = { kind: "subgoal", id: "flat", goal: "Agent steps", steps: [], verifiers: [] };
    run.items.push(sg);
  }
  return sg;
}

function applyEvent(run: Run, ev: AgentEvent): Run {
  // Mutate a shallow clone (caller already cloned the run).
  switch (ev.type) {
    case "run_started":
      run.agent = ev.agent; run.mode = ev.mode; run.task = ev.task; run.status = "running";
      break;
    case "plan":
      run.items.push({ kind: "plan", subgoals: ev.subgoals });
      break;
    case "subgoal_start":
      run.items.push({
        kind: "subgoal", id: ev.id, goal: ev.goal, success: ev.success_condition,
        needsApproval: ev.needs_approval, tier: ev.tier, steps: [], verifiers: [], status: "running",
      });
      break;
    case "extract":
      run.pendingExtract = true;
      break;
    case "grounding":
      run.grounding = ev.text;
      run.items.push({ kind: "grounding", text: ev.text });
      break;
    case "step": {
      const sg = ensureContainer(run);
      const step: Step = {
        n: ev.step, thought: ev.thought, action: ev.action, args: ev.args, actions: ev.actions,
        extract: run.pendingExtract || false,
      };
      run.pendingExtract = false;
      sg.steps.push(step);
      break;
    }
    case "action_result": {
      const sg = lastSubgoal(run);
      const step = sg?.steps[sg.steps.length - 1];
      if (step) { step.result = ev.outcome; step.blocked = ev.blocked; step.ok = ev.ok; }
      break;
    }
    case "candidates":
      run.items.push({ kind: "candidates", count: ev.count, total: ev.total,
                       items: ev.candidates || [], blocked: ev.blocked, source: ev.source });
      break;
    case "exploit":
      run.items.push({ kind: "exploit", selected: ev.selected, top: ev.top || [], nearTie: ev.near_tie });
      break;
    case "present":
      run.items.push({ kind: "present", markdown: ev.markdown });
      break;
    case "loop_nudge":
      run.items.push({ kind: "note", tone: "warn",
                       text: `Loop detected on ${ev.action} — nudged to change tactics.` });
      break;
    case "stagnation_nudge":
      run.items.push({ kind: "note", tone: "warn",
                       text: "Page isn't changing — nudged to try a different approach." });
      break;
    case "budget_warning":
      run.items.push({ kind: "note", tone: "warn",
                       text: `Step budget ${ev.step}/${ev.max_steps} — time to wrap up or escalate.` });
      break;
    case "verifier": {
      const sg = lastSubgoal(run);
      if (sg) sg.verifiers.push({ satisfied: ev.satisfied, reason: ev.reason });
      break;
    }
    case "subgoal_end": {
      // attach to the most recent subgoal with this id
      for (let i = run.items.length - 1; i >= 0; i--) {
        const it = run.items[i];
        if (it.kind === "subgoal" && it.id === ev.id) { it.status = ev.status; it.detail = ev.detail; break; }
      }
      break;
    }
    case "replan":
      run.items.push({ kind: "replan", n: ev.n, reason: ev.reason });
      run.items.push({ kind: "plan", subgoals: ev.subgoals });
      break;
    case "tabs_parked":
      run.items.push({ kind: "tabs", mode: "parked", urls: ev.urls });
      break;
    case "tabs_reopened":
      run.items.push({ kind: "tabs", mode: "reopened", urls: ev.urls });
      break;
    case "approval_request":
      run.approval = { id: ev.id, prompt: ev.prompt };
      break;
    case "ask_request":
      run.ask = { id: ev.id, question: ev.question, options: ev.options,
                  allow_free_text: ev.allow_free_text, multi_select: ev.multi_select };
      break;
    case "ask_answer": {
      const sg = lastSubgoal(run);
      if (sg) sg.steps.push({ n: 0, action: "ask_human (answered)", args: ev.answer });
      break;
    }
    case "final_answer":
      run.result = ev.answer;
      break;
    case "run_finished":
      run.status = "finished"; run.result = ev.result;
      break;
    case "run_error":
      run.status = "error"; run.error = ev.message;
      break;
    case "run_cancelled":
      run.status = "cancelled";
      break;
  }
  return run;
}

function reducer(state: State, action: Action): State {
  switch (action.kind) {
    case "conn":
      return { ...state, conn: action.conn };
    case "reset":
      return { ...state, runs: [] };
    case "newRun":
      return { ...state, runs: [...state.runs, action.run] };
    case "clearApproval": {
      const runs = state.runs.slice();
      const r = { ...runs[runs.length - 1], approval: null };
      runs[runs.length - 1] = r;
      return { ...state, runs };
    }
    case "clearAsk": {
      const runs = state.runs.slice();
      const r = { ...runs[runs.length - 1], ask: null };
      runs[runs.length - 1] = r;
      return { ...state, runs };
    }
    case "event": {
      if (action.ev.type === "ready") return { ...state, groqKey: action.ev.groq_key };
      if (state.runs.length === 0) return state;
      const runs = state.runs.slice();
      const cloned: Run = {
        ...runs[runs.length - 1],
        items: runs[runs.length - 1].items.map((it) =>
          it.kind === "subgoal"
            ? { ...it, steps: it.steps.map((s) => ({ ...s })), verifiers: it.verifiers.slice() }
            : { ...it },
        ),
      };
      runs[runs.length - 1] = applyEvent(cloned, action.ev);
      return { ...state, runs };
    }
  }
}

export function useAgentSocket() {
  const [state, dispatch] = useReducer(reducer, { conn: "connecting", groqKey: null, runs: [] });
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    dispatch({ kind: "conn", conn: "connecting" });
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => {
      dispatch({ kind: "conn", conn: "open" });
      // Tell the backend which tab we are, so it can switch focus back here when it asks.
      ws.send(JSON.stringify({ type: "hello", url: window.location.href }));
    };
    ws.onclose = () => dispatch({ kind: "conn", conn: "closed" });
    ws.onerror = () => dispatch({ kind: "conn", conn: "closed" });
    ws.onmessage = (m) => {
      try { dispatch({ kind: "event", ev: JSON.parse(m.data) as AgentEvent }); } catch { /* ignore */ }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const send = (obj: unknown) => wsRef.current?.send(JSON.stringify(obj));

  const start = useCallback((task: string, s: Settings) => {
    const run: Run = { task, agent: s.agent, mode: "live", status: "running", items: [], approval: null, ask: null };
    dispatch({ kind: "newRun", run });
    send({ type: "start", task, agent: s.agent, allow: s.allow,
           maxSteps: s.maxSteps, extract: s.extract, grounding: s.grounding, pickTab: s.pickTab,
           parallel: s.parallel });
  }, []);

  const respondApproval = useCallback((id: number, approved: boolean) => {
    send({ type: "approval_response", id, approved });
    dispatch({ kind: "clearApproval" });
  }, []);

  const respondAsk = useCallback((id: number, answer: string) => {
    send({ type: "ask_response", id, answer });
    dispatch({ kind: "clearAsk" });
  }, []);

  const cancel = useCallback(() => send({ type: "cancel" }), []);
  const newChat = useCallback(() => dispatch({ kind: "reset" }), []);

  return { state, start, respondApproval, respondAsk, cancel, newChat, reconnect: connect };
}
