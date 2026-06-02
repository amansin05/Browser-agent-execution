import { useEffect, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { useAgentSocket } from "./useAgentSocket";
import { useAttention } from "./useAttention";
import type { Run, Settings, Step, SubgoalCard, TimelineItem } from "./types";
import {
  ArrowUp, Chevron, Doc, Eye, Globe, Plus, Robot, Search, Sliders, Spark, Stop,
} from "./icons";

const DEFAULT_SETTINGS: Settings = {
  agent: "two-tier", allow: "", maxSteps: 12, extract: true, grounding: true, pickTab: false,
};

const QUICK_START = [
  { icon: Globe, title: "Read a page",
    desc: "Open example.com and report its main heading.",
    task: "Go to https://example.com and tell me the exact text of the main heading." },
  { icon: Search, title: "Search Wikipedia",
    desc: "Look up Mars and quote the first sentence.",
    task: "Go to en.wikipedia.org, search for Mars, and report the first sentence of the article." },
  { icon: Doc, title: "Extract data",
    desc: "List the top stories on Hacker News.",
    task: "Go to news.ycombinator.com and list the titles of the top 5 stories." },
  { icon: Spark, title: "Summarize my tab",
    desc: "Summarize the page open in your live Chrome.",
    task: "Summarize the page in my active browser tab in two sentences." },
];

function greeting(): string {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

export default function App() {
  const { state, start, respondApproval, respondAsk, cancel, newChat } = useAgentSocket();
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [draft, setDraft] = useState("");
  const [showCustomize, setShowCustomize] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const last = state.runs[state.runs.length - 1];
  const running = last?.status === "running";
  const empty = state.runs.length === 0;

  // The browser can't switch tabs to us (dev-ui is outside Playwright's tab group), so when the
  // agent is waiting on an answer/approval, alert in-page: flash the title, beep, notify.
  const pending = last?.ask ?? last?.approval ?? null;
  useAttention(!!pending, last?.ask?.question || last?.approval?.prompt || "Your input is needed");

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [state.runs]);

  const submit = () => {
    const task = draft.trim();
    if (!task || running || state.conn !== "open") return;
    start(task, settings);
    setDraft("");
  };

  const composer = (
    <Composer
      draft={draft} setDraft={setDraft} onSubmit={submit} onCancel={cancel}
      running={running} conn={state.conn} settings={settings} setSettings={setSettings}
      onCustomize={() => setShowCustomize((v) => !v)} taRef={taRef}
    />
  );

  return (
    <div className="app">
      <Sidebar
        settings={settings} setSettings={setSettings}
        conn={state.conn} groqKey={state.groqKey} runs={state.runs}
        showCustomize={showCustomize} setShowCustomize={setShowCustomize}
        onNewChat={() => { newChat(); setDraft(""); taRef.current?.focus(); }}
        onJump={(i) => document.getElementById(`run-${i}`)?.scrollIntoView({ behavior: "smooth" })}
      />

      <main className="main">
        <header className="topbar">
          <div className="seg">
            {(["two-tier", "flat"] as const).map((a) => (
              <button key={a} className={settings.agent === a ? "seg-on" : ""}
                onClick={() => setSettings({ ...settings, agent: a })}>
                {a === "two-tier" ? "Two-tier" : "Flat"}
              </button>
            ))}
          </div>
        </header>

        {empty ? (
          <div className="hero">
            <div className="pill">
              Live profile
              <span className="pill-sep" />
              <span className="pill-accent">{settings.agent === "two-tier" ? "Two-tier" : "Flat"}</span>
            </div>
            <h1 className="greeting">{greeting()}</h1>
            <p className="tagline">What should the browser agent do?</p>
            <div className="hero-composer">{composer}</div>
            <div className="qs-label">QUICK START</div>
            <div className="qs-grid">
              {QUICK_START.map((q) => (
                <button key={q.title} className="qs-card"
                  onClick={() => { setDraft(q.task); taRef.current?.focus(); }}>
                  <span className="qs-icon"><q.icon /></span>
                  <div>
                    <div className="qs-title">{q.title}</div>
                    <div className="qs-desc">{q.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="thread" ref={threadRef}>
              {state.runs.map((run, i) => (
                <div id={`run-${i}`} key={i}>
                  <RunView run={run} onApprove={(id, ok) => respondApproval(id, ok)} />
                </div>
              ))}
            </div>
            <div className="dock">{composer}</div>
          </>
        )}
      </main>

      {last?.ask && <AskModal ask={last.ask} onSubmit={(a) => respondAsk(last!.ask!.id, a)} />}
    </div>
  );
}

/* ------------------------------------------------------------------ sidebar */
function Sidebar(props: {
  settings: Settings; setSettings: (s: Settings) => void;
  conn: string; groqKey: boolean | null; runs: Run[];
  showCustomize: boolean; setShowCustomize: (v: boolean) => void;
  onNewChat: () => void; onJump: (i: number) => void;
}) {
  const { settings, setSettings, conn, groqKey, runs, showCustomize, setShowCustomize, onNewChat, onJump } = props;
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings({ ...settings, [k]: v });
  const recents = runs.map((r, i) => ({ r, i })).reverse();

  return (
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Robot width={16} height={16} /></span> Browser Agent</div>

      <div className="side-actions">
        <button onClick={onNewChat}><Plus /> New run</button>
        <button className={showCustomize ? "active" : ""} onClick={() => setShowCustomize(!showCustomize)}>
          <Sliders /> Customize
        </button>
      </div>

      {showCustomize && (
        <div className="customize">
          <div className="customize-note">Runs in your live, logged-in Chrome (Playwright extension).</div>
          <label className="check"><input type="checkbox" checked={settings.pickTab}
            onChange={(e) => set("pickTab", e.target.checked)} /> Wait for me to pick a tab</label>
          {settings.agent === "two-tier" && (
            <>
              <label className="field"><span>Domain allowlist</span>
                <input type="text" placeholder="github.com, google.com" value={settings.allow}
                  onChange={(e) => set("allow", e.target.value)} /></label>
              <label className="check"><input type="checkbox" checked={settings.grounding}
                onChange={(e) => set("grounding", e.target.checked)} /> Web grounding (search before planning)</label>
              <label className="check"><input type="checkbox" checked={settings.extract}
                onChange={(e) => set("extract", e.target.checked)} /> Content extractor (opaque pages)</label>
            </>
          )}
          <label className="field"><span>Max steps</span>
            <input type="number" min={1} max={50} value={settings.maxSteps}
              onChange={(e) => set("maxSteps", Number(e.target.value) || 12)} /></label>
        </div>
      )}

      <div className="recents-label">RECENTS</div>
      <div className="recents">
        {recents.length === 0 && <div className="recents-empty">No runs yet.</div>}
        {recents.map(({ r, i }) => (
          <button key={i} className="recent" onClick={() => onJump(i)}>
            <div className="recent-title">{r.task}</div>
            <div className="recent-sub">{r.agent} · {r.status}</div>
          </button>
        ))}
      </div>

      <div className="profile">
        <span className="avatar">BA</span>
        <div className="profile-meta">
          <div className="profile-name">{conn === "open" ? "Connected" : conn}</div>
          <div className="profile-sub">
            <span className={`dot ${groqKey ? "ok" : "bad"}`} /> Groq key {groqKey == null ? "…" : groqKey ? "set" : "missing"}
          </div>
        </div>
        <Chevron width={16} height={16} />
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ composer */
function Composer(props: {
  draft: string; setDraft: (s: string) => void; onSubmit: () => void; onCancel: () => void;
  running: boolean; conn: string; settings: Settings; setSettings: (s: Settings) => void;
  onCustomize: () => void; taRef: RefObject<HTMLTextAreaElement>;
}) {
  const { draft, setDraft, onSubmit, onCancel, running, conn, settings, setSettings, onCustomize, taRef } = props;
  const disabled = conn !== "open";
  return (
    <div className="composer">
      <textarea
        ref={taRef} value={draft} rows={1} disabled={disabled}
        placeholder={running ? "Agent is running…" : "How can I help you automate today?  (Enter to send, Shift+Enter for a new line)"}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSubmit(); } }}
      />
      <div className="composer-foot">
        <div className="composer-tools">
          <button className="tool" title="Customize" onClick={onCustomize}><Sliders /></button>
          {settings.agent === "two-tier" && (
            <>
              <button className={`tool ${settings.grounding ? "tool-on" : ""}`}
                title="Web grounding (search before planning)"
                onClick={() => setSettings({ ...settings, grounding: !settings.grounding })}><Search /></button>
              <button className={`tool ${settings.extract ? "tool-on" : ""}`}
                title="Content extractor (opaque pages)"
                onClick={() => setSettings({ ...settings, extract: !settings.extract })}><Eye /></button>
            </>
          )}
        </div>
        <div className="composer-right">
          <span className="live-chip"><Globe width={14} height={14} /> Live profile</span>
          {running
            ? <button className="send danger" onClick={onCancel} title="Cancel"><Stop /></button>
            : <button className="send" onClick={onSubmit} disabled={!draft.trim() || disabled} title="Send"><ArrowUp /></button>}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ actors */
// Each part of the run is attributed to an "actor" (nanobrowser-style), so you can see WHICH agent
// is thinking/acting in the stream — Planner decides the plan, Navigator drives the page, Verifier
// checks success, System reports grounding/candidates/results.
type Actor = "system" | "planner" | "navigator" | "verifier";
const ACTORS: Record<Actor, { name: string; color: string }> = {
  system: { name: "System", color: "#2f6fb0" },
  planner: { name: "Planner", color: "#c2620e" },
  navigator: { name: "Navigator", color: "#2a9d8f" },
  verifier: { name: "Verifier", color: "#b5478f" },
};

function ActorTag({ actor }: { actor: Actor }) {
  const a = ACTORS[actor];
  return (
    <span className="actor">
      <span className="actor-dot" style={{ background: a.color }} />
      <span className="actor-name" style={{ color: a.color }}>{a.name}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ run thread */
function RunView({ run, onApprove }: {
  run: Run; onApprove: (id: number, ok: boolean) => void;
}) {
  return (
    <div className="run">
      <div className="msg user"><div className="bubble">{run.task}</div></div>
      <div className="msg agent">
        <span className="msg-av"><Robot width={16} height={16} /></span>
        <div className="agent-body">
          <div className="run-meta">
            <Tag>{run.agent}</Tag><Tag>{run.mode}</Tag><StatusTag status={run.status} />
          </div>
          {run.items.map((it, i) => <TimelineItemView key={i} item={it} />)}

          {run.approval && (
            <div className="prompt">
              <div className="prompt-title">Approval required</div>
              <div className="prompt-body">{run.approval.prompt}</div>
              <div className="prompt-actions">
                <button className="btn-approve" onClick={() => onApprove(run.approval!.id, true)}>Approve</button>
                <button className="btn-deny" onClick={() => onApprove(run.approval!.id, false)}>Deny</button>
              </div>
            </div>
          )}
          {run.status === "finished" && run.result &&
            <div className="result ok"><div className="result-label">Result</div>{run.result}</div>}
          {run.status === "error" &&
            <div className="result err"><div className="result-label">Error</div>{run.error}</div>}
          {run.status === "cancelled" &&
            <div className="result warn"><div className="result-label">Cancelled</div>Run cancelled.</div>}
          {run.status === "running" && !run.approval && !run.ask && <ProgressLine run={run} />}
        </div>
      </div>
    </div>
  );
}

// A transient progress bar attributed to whichever actor is currently working (nanobrowser shows an
// animated bar that the next real message replaces). We infer the actor from the latest timeline item.
function ProgressLine({ run }: { run: Run }) {
  const last = run.items[run.items.length - 1];
  const actor: Actor =
    !last ? "system"
    : last.kind === "plan" || last.kind === "replan" ? "planner"
    : last.kind === "subgoal" ? "navigator"
    : "system";
  const a = ACTORS[actor];
  return (
    <div className="progress">
      <span className="actor-dot" style={{ background: a.color }} />
      <span className="actor-name" style={{ color: a.color }}>{a.name}</span>
      <span className="progress-bar"><span className="progress-fill" /></span>
    </div>
  );
}

function TimelineItemView({ item }: { item: TimelineItem }) {
  switch (item.kind) {
    case "plan":
      return (
        <div className="block">
          <ActorTag actor="planner" />
          <div className="plan">
            <ol className="plan-list">
              {item.subgoals.map((s) => (
                <li key={s.id}>{s.goal}{s.needs_approval && <span className="chip warn">needs approval</span>}
                  <div className="plan-success">✓ {s.success_condition}</div></li>
              ))}
            </ol>
          </div>
        </div>
      );
    case "replan":
      return (
        <div className="block">
          <ActorTag actor="planner" />
          <div className="replan">↻ Re-plan #{item.n} — {item.reason}</div>
        </div>
      );
    case "grounding":
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className="sysnote"><b>Web grounding</b>
            <div className="sysnote-body">{shorten(item.text, 600)}</div></div>
        </div>
      );
    case "candidates":
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className={`sysnote ${item.count === 0 ? "sysnote-warn" : ""}`}>
            {item.count > 0 ? (
              <>
                <b>Gathered {item.count} candidate{item.count === 1 ? "" : "s"}</b>
                {item.total ? ` (${item.total} total)` : ""}{item.source ? ` from ${item.source}` : ""}
                <ul className="cand-list">
                  {item.items.slice(0, 5).map((c, i) => <li key={i}>{candLabel(c)}</li>)}
                </ul>
              </>
            ) : (
              <><b>No candidates</b>{item.source ? ` from ${item.source}` : ""}
                {item.blocked ? " — page looked blocked" : ""}</>
            )}
          </div>
        </div>
      );
    case "exploit":
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className="sysnote">
            <b>Ranked picks</b>{item.nearTie && <span className="chip warn">near tie</span>}
            {item.selected
              ? <div className="cand-top">★ {candLabel(item.selected)}</div>
              : <div>no candidates to rank</div>}
          </div>
        </div>
      );
    case "present":
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className="present">{item.markdown}</div>
        </div>
      );
    case "note":
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className={`sysnote ${item.tone === "warn" ? "sysnote-warn" : ""}`}>{item.text}</div>
        </div>
      );
    case "tabs": {
      const verb = item.mode === "parked" ? "Parked & closed" : "Reopened for follow-up";
      const n = item.urls.length;
      return (
        <div className="block">
          <ActorTag actor="system" />
          <div className="sysnote" title={item.urls.join("\n")}>
            ⧉ {verb} {n} tab{n === 1 ? "" : "s"} — {item.urls.join(", ")}
          </div>
        </div>
      );
    }
    default:  // SubgoalCard — the Navigator working a subgoal
      return (
        <div className="block">
          <ActorTag actor="navigator" />
          <SubgoalView sg={item} />
        </div>
      );
  }
}

function candLabel(c: Record<string, unknown>): string {
  const name = String(c.name ?? c.title ?? c.model ?? "item");
  const price = c.price != null ? ` — ${c.price}` : "";
  const src = c.source != null ? ` (${c.source})` : "";
  return `${name}${price}${src}`;
}

function SubgoalView({ sg }: { sg: SubgoalCard }) {
  return (
    <div className={`subgoal s-${sg.status ?? "running"}`}>
      <div className="subgoal-head">
        <span>{sg.goal}</span>
        <span className="subgoal-tags">
          {sg.tier && sg.tier !== "auto" && <span className={`chip tier-${sg.tier}`}>{sg.tier}</span>}
          {sg.status && <span className={`chip s-${sg.status}`}>{sg.status}</span>}
        </span>
      </div>
      {sg.steps.map((s, i) => <StepView key={i} step={s} />)}
      {sg.verifiers.map((v, i) => (
        <div key={`v${i}`} className={`verifier ${v.satisfied ? "ok" : "no"}`}>
          <span className="actor-name" style={{ color: ACTORS.verifier.color }}>Verifier</span>
          {" "}{v.satisfied ? "satisfied" : "not satisfied"} — {v.reason}
        </div>
      ))}
    </div>
  );
}

function StepView({ step }: { step: Step }) {
  // A turn may chain several actions (multi-action); render each, with the thought above them.
  const acts = step.actions && step.actions.length ? step.actions : [{ action: step.action, args: step.args }];
  return (
    <div className="step">
      {step.thought && <div className="thought">{step.thought}</div>}
      {acts.map((a, i) => (
        <div className="step-line" key={i}>
          {i === 0 && step.n > 0 && <span className="step-n">{step.n}</span>}
          {i > 0 && <span className="step-n step-n-cont">↳</span>}
          <span className="action">{a.action}</span>
          <span className="args">{shorten(JSON.stringify(a.args), 120)}</span>
          {i === 0 && step.extract && <span className="chip extract">extract</span>}
          {i === 0 && step.blocked && <span className="chip blocked">blocked</span>}
        </div>
      ))}
      {step.result && <div className="step-result">{shorten(step.result, 260)}</div>}
    </div>
  );
}

const OTHER_ID = "__other__";
// Matches SKIP_SENTINEL in orchestrate.py: tells the agent the user has no preference, so it
// should explore broadly across all options instead of narrowing.
const SKIP_SENTINEL = "__no_preference__";

function AskModal({ ask, onSubmit }: { ask: NonNullable<Run["ask"]>; onSubmit: (a: string) => void }) {
  const multi = ask.multi_select ?? false;
  const allowOther = ask.allow_free_text ?? true;
  // Drop any model-provided "Other" option — we render our own (which reveals a write box).
  const options = (ask.options ?? []).filter(
    (o) => !["other", "others"].includes((o.label || o.id || "").trim().toLowerCase()));
  const [picked, setPicked] = useState<string[]>([]);
  const [other, setOther] = useState("");
  const dialogRef = useRef<HTMLDivElement>(null);
  const otherOn = picked.includes(OTHER_ID);

  useEffect(() => { dialogRef.current?.focus(); }, []);

  const choose = (id: string) =>
    setPicked((p) => (multi ? (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]) : [id]));

  const labelOf = (id: string) => options.find((o) => o.id === id)?.label ?? id;
  const chosen = picked.filter((id) => id !== OTHER_ID).map(labelOf);
  if (otherOn && other.trim()) chosen.push(other.trim());
  const canSend = chosen.length > 0;
  const send = () => { if (canSend) onSubmit(chosen.join(", ")); };

  const Mark = ({ on }: { on: boolean }) =>
    <span className={`mark ${multi ? "box" : "radio"} ${on ? "on" : ""}`} />;

  return (
    <div className="modal-overlay">
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="ask-title"
        tabIndex={-1} ref={dialogRef}>
        <div className="modal-eyebrow">{multi ? "Select all that apply" : "Select one"}</div>
        <h3 id="ask-title" className="modal-title">{ask.question}</h3>

        <div className="ask-options" role={multi ? "group" : "radiogroup"}>
          {options.map((o) => (
            <button key={o.id} type="button" role={multi ? "checkbox" : "radio"}
              aria-checked={picked.includes(o.id)}
              className={`ask-opt ${picked.includes(o.id) ? "on" : ""}`} onClick={() => choose(o.id)}>
              <Mark on={picked.includes(o.id)} />
              <span><b>{o.label}</b>{o.detail ? <span className="ask-detail"> — {o.detail}</span> : null}</span>
            </button>
          ))}
          {allowOther && (
            <button type="button" role={multi ? "checkbox" : "radio"} aria-checked={otherOn}
              className={`ask-opt ${otherOn ? "on" : ""}`} onClick={() => choose(OTHER_ID)}>
              <Mark on={otherOn} /><span><b>Other…</b><span className="ask-detail"> write your own</span></span>
            </button>
          )}
        </div>

        {otherOn && (
          <input className="ask-input modal-other" autoFocus value={other} placeholder="Type your answer"
            onChange={(e) => setOther(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") send(); }} />
        )}

        <div className="modal-actions">
          <button className="btn-skip" onClick={() => onSubmit(SKIP_SENTINEL)}
            title="No preference — let the agent explore all options and compare them">
            No preference — explore all
          </button>
          <button className="btn-approve" disabled={!canSend} onClick={send}>Send answer</button>
        </div>
      </div>
    </div>
  );
}

const Tag = ({ children }: { children: ReactNode }) => <span className="tag">{children}</span>;
const StatusTag = ({ status }: { status: Run["status"] }) => {
  const label: Record<Run["status"], string> = {
    running: "running", finished: "done", error: "error", cancelled: "cancelled",
  };
  return <span className={`tag t-${status}`}>{label[status]}</span>;
};

function shorten(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
