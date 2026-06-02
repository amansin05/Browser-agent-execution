// Events streamed by the backend (mirror of the agent's `emit` payloads).
export type AgentEvent =
  | { type: "ready"; groq_key: boolean }
  | { type: "run_started"; agent: string; mode: string; task: string }
  | { type: "plan"; subgoals: PlanSubgoal[] }
  | { type: "subgoal_start"; id: number; goal: string; success_condition: string; needs_approval: boolean; tier?: string; kind?: string }
  | { type: "step"; step: number; thought?: string; action: string; args: unknown; actions?: ActionItem[] }
  | { type: "extract"; step: number; chars?: number }
  | { type: "grounding"; text: string }
  | { type: "action_result"; action: string; outcome: string; blocked?: boolean; ok?: boolean }
  | { type: "verifier"; satisfied: boolean; reason: string }
  | { type: "ask_answer"; answer: string }
  | { type: "subgoal_end"; id: number; status: string; detail: string }
  | { type: "replan"; n: number; reason: string; subgoals: PlanSubgoal[] }
  | { type: "approval_request"; id: number; prompt: string }
  | { type: "ask_request"; id: number; question: string; options?: AskOption[]; allow_free_text?: boolean; multi_select?: boolean }
  | { type: "candidates"; id: number; count: number; total?: number; candidates: Record<string, unknown>[]; blocked?: boolean; source?: string }
  | { type: "exploit"; id: number; selected: Record<string, unknown> | null; top: Record<string, unknown>[]; near_tie: boolean }
  | { type: "present"; id: number; markdown: string }
  | { type: "loop_nudge"; step: number; action: string }
  | { type: "stagnation_nudge"; step: number }
  | { type: "budget_warning"; step: number; max_steps: number }
  | { type: "memory_update"; outcome?: string; preference_updates?: Record<string, number> }
  | { type: "tabs_parked"; urls: string[] }
  | { type: "tabs_reopened"; urls: string[] }
  | { type: "final_answer"; answer: string }
  | { type: "run_finished"; result: string | null }
  | { type: "run_error"; message: string }
  | { type: "run_cancelled" };

export interface ActionItem { action: string; args: unknown; }

export interface PlanSubgoal {
  id: number;
  goal: string;
  success_condition: string;
  needs_approval: boolean;
  tier?: string;
  type?: string;
}

export interface AskOption {
  id: string;
  label: string;
  detail?: string;
}

export interface Step {
  n: number;
  thought?: string;
  action: string;
  args: unknown;
  actions?: ActionItem[];   // multi-action turns (the reasoner may chain several actions)
  result?: string;
  blocked?: boolean;
  ok?: boolean;
  extract?: boolean;
}

export interface SubgoalCard {
  kind: "subgoal";
  id: number | string;
  goal: string;
  success?: string;
  needsApproval?: boolean;
  tier?: string;
  steps: Step[];
  verifiers: { satisfied: boolean; reason: string }[];
  status?: string;
  detail?: string;
}

export type TimelineItem =
  | { kind: "plan"; subgoals: PlanSubgoal[] }
  | SubgoalCard
  | { kind: "replan"; n: number; reason: string }
  | { kind: "tabs"; mode: "parked" | "reopened"; urls: string[] }
  | { kind: "grounding"; text: string }
  | { kind: "candidates"; count: number; total?: number; items: Record<string, unknown>[]; blocked?: boolean; source?: string }
  | { kind: "exploit"; selected: Record<string, unknown> | null; top: Record<string, unknown>[]; nearTie?: boolean }
  | { kind: "present"; markdown: string }
  | { kind: "note"; tone: "system" | "warn"; text: string };

// Who an item is "spoken by" — drives the actor avatar/colour in the stream (nanobrowser-style).
export type Actor = "system" | "planner" | "navigator" | "verifier" | "user";

export type RunStatus = "running" | "finished" | "error" | "cancelled";

export interface Run {
  task: string;
  agent: string;
  mode: string;
  status: RunStatus;
  items: TimelineItem[];
  result?: string | null;
  error?: string;
  approval?: { id: number; prompt: string } | null;
  ask?: { id: number; question: string; options?: AskOption[]; allow_free_text?: boolean; multi_select?: boolean } | null;
  pendingExtract?: boolean;
  grounding?: string;
}

export type ConnState = "connecting" | "open" | "closed";

export interface Settings {
  agent: "two-tier" | "flat";
  allow: string;
  maxSteps: number;
  extract: boolean;
  grounding: boolean;
  pickTab: boolean;
}
