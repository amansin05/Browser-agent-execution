# Browser Agent — Complete Execution Plan

A phased, build-it-incrementally plan for a personal browser-automation agent. Each phase
ends in something that *works*, so you're never debugging a huge unproven system at once.
The full planner/reasoner prompt templates and loop code are included in Appendix A.

---

## The stack at a glance

| Layer | Choice | Why |
|---|---|---|
| Hands (browser control) | **browser-use** (Python) on Playwright | Wraps perception + the loop; DOM-first; ~89% WebVoyager |
| Brain (thinking) | **Groq `llama-4-scout-17b-16e-instruct`** | Free-tier, very fast inference, multimodal, no local GPU |
| Brain structure | Planner + Reasoner (one model, two prompts) | Plan stays stable; reasoner picks one action per step |
| Profile control | Attach to your real Chrome over CDP | Acts as you, reuses logins — no re-auth |
| Perception | DOM-first, vision only as fallback | Cheaper/faster; Scout's multimodality covers the fallback |
| Safety | Human-approval gate + independent verify | Never auto-confirm consequential actions |

**The fork to decide early (Phase 0):**
- **Path A — browser-use + Groq (recommended for this project).** You write Python, build
  your own planner/reasoner brain, and Groq powers it. Profile control via CDP. Best if you
  want to *understand and own* the agent.
- **Path B — Playwright MCP + Playwright Extension.** The agent is an MCP client
  (Claude Desktop, Cursor, VS Code, or custom). Less code, but the "brain" is whatever the
  MCP client uses; using Scout as that brain means writing a custom MCP client. Best if you
  mainly want to drive your logged-in browser from an existing MCP app.

This plan follows **Path A** as the spine and notes Path B at the profile-control step.

---

## ✅ Implementation status (what we actually built)

> **We chose Path B** (Playwright MCP + a custom Scout MCP client), not Path A. The reasons and
> the full architecture are in [README.md](README.md). browser-use (Path A) survives only as a
> fallback hello-world. Below, each plan phase is mapped to the real code + its test.

| Phase | Status | Where it lives |
|---|---|---|
| 0 — env & prerequisites | ✅ done | Python 3.12, Node 24, Chrome, Groq key in `.env`, `.venv` |
| 1 — hello-world loop | ✅ done | `browser_agent.agent.flat` (Path B); `scripts/hello_world.py` (Path A fallback) |
| 2 — instrument & guardrails | ✅ done | per-step trace logging, `--max-steps`, loop detection |
| 3 — planner + reasoner + verify + re-plan | ✅ done | **`browser_agent.agent.main`** (this is the two-tier brain from Appendix A) |
| 4 — real logged-in profile | ✅ done | `browser_agent.agent.flat --extension` (proved by reading the live Google account) |
| 5 — safety / human-in-the-loop | ✅ done | subgoal `needs_approval` gate + `--allow` domain allowlist in `browser_agent.agent.main` |
| 6 — vision fallback & eval set | ✅ done | vision in `browser_agent.agent.main`; `scripts/eval_set.py` (6/6) |
| 7 — go local / scale | ⬜ optional | — |

**Tests:** `test_mcp_tools.py` (23/23 tools), `test_agent_loop.py` (12/12 flat-loop),
`test_planner_reasoner.py` (14/14 two-tier incl. vision), plus live `test_agent_e2e.py`,
`test_planner_reasoner_e2e.py`, `test_vision_e2e.py`, and `scripts/eval_set.py` (6/6 = 100%).

**Key deviations from the plan as written** (details in README):
- Path B uses **Playwright MCP tools** as the reasoner's action vocabulary (not the custom
  `click(index)` vocab in Appendix A) and **MCP's accessibility snapshot** as the observation
  (we don't hand-serialize the DOM — section 3 of Appendix A is handled for us).
- The reasoner is **single-shot per turn** (fresh snapshot + recent actions each step), honoring
  "never re-feed full history". `subgoal_complete` / `escalate` / `ask_human` are exposed as
  synthetic tools so the model signals them through the same tool-calling channel.
- The verifier is a **separate, focused Scout call** (sees only the success condition + a fresh
  snapshot) — the pragmatic realization of "don't trust the model's word" for NL conditions.

---

## Phase 0 — Decisions & prerequisites

- [x] Confirm Path A vs B (above). **Chose Path B.**
- [x] Install **Python 3.12+** and **Node.js** (Node 24; needed for `npx @playwright/mcp`).
- [x] Install **Google Chrome**.
- [x] Create a free **Groq API key** at the Groq console; set it as `GROQ_API_KEY` (in `.env`).
- [x] Make a project folder and a virtualenv (`.venv`).

```bash
mkdir browser-agent && cd browser-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install browser-use
playwright install chromium
export GROQ_API_KEY=...        # Windows: setx GROQ_API_KEY ...
```

**Exit criteria:** environment installed, `GROQ_API_KEY` set, Chrome present.

---

## Phase 1 — Hello-world agent

Prove the full observe → decide → act → verify loop on a **non-logged-in** site first.

> **⚠️ Superseded by Path B.** The snippet below is the original Path-A (browser-use) idea and
> is kept for reference only. What we actually run is **`browser_agent.agent.flat`** (custom Scout MCP
> client over Playwright MCP). The browser-use version lives in `scripts/hello_world.py` as a fallback;
> note that on the installed browser-use 0.12.x the class is `ChatGroq`, not `ChatOpenAI`, and
> no `base_url` is needed. See [README.md](README.md).

```python
# Path A reference only (we use browser_agent.agent.flat instead):
import os
from browser_use import Agent
from browser_use.llm import ChatGroq            # 0.12.x: ChatGroq, no base_url needed

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
               api_key=os.environ["GROQ_API_KEY"])
agent = Agent(task="Go to en.wikipedia.org, search 'Mars', return the first sentence.", llm=llm)
# asyncio.run(agent.run())
```

**Exit criteria:** a browser window opens, the agent searches, returns an answer. You've seen
the loop run end to end.

---

## Phase 2 — Understand & instrument

Before customizing, see what's actually happening.

- [x] Verbose per-step logging: each turn prints the chosen action + its result (and, in the
      two-tier agent, the reasoner's thought and the verifier's verdict).
- [x] Guardrails: `--max-steps` per task/subgoal + **loop detection** (repeated identical
      action ⇒ forced escalate). Plus the Scout `tool_use_failed` JSON-wobble retry.
- [x] Ran tasks of rising difficulty (Wikipedia search, the local multi-element test page,
      live Google account read). Stumbles noted in README (snapshot `filename`, modal cascades).

**Exit criteria:** you can read a run trace and explain why each action was chosen.

---

## Phase 3 — Add your custom brain (planner + reasoner)

Layer the two-tier decomposition on top. **The complete prompts, JSON schemas, observation
format, and loop code for this phase are in Appendix A at the end of this document.**

- [x] **Planner**: `plan_subgoals()` calls Scout → ordered subgoals JSON (`goal`,
      `success_condition`, `needs_approval`). Defensive JSON parsing via `extract_json()`.
- [x] **Reasoner**: `reasoner_decide()` — per step, fresh MCP snapshot + recent actions → one
      tool call (the action). Observation comes from Playwright MCP, not hand-serialized.
- [x] **Verifier**: `verify_success()` — independent Scout call on `subgoal_complete`; the
      live E2E showed it correctly rejecting a hallucinated success condition.
- [x] **Re-plan**: on `escalate`/exhaust, `plan_subgoals(prior_plan=...)` rewrites the remainder
      (budgeted by `--max-replans`). E2E recovered through two failed plans to "Done.".
- [x] State kept small: plan + current index + recent-actions ring buffer + observations
      scratchpad; each reasoner turn re-serializes the page rather than re-feeding history.

Use **one model (Scout) in two prompt modes** to start; split into two models only later.

**Exit criteria:** the agent plans subgoals up front, executes each as a verified loop, and
recovers from a deliberately-broken plan via re-plan.

---

## Phase 4 — Connect to your real logged-in profile

So the agent acts as *you*, reusing your sessions.

**✅ We did this via Path B.** Installed the **Playwright MCP Bridge** extension (Chrome Web
Store, ID `mmlmfjhmonkocbjadbfplnigmagldckm`, publisher Microsoft), put its token in `.env` as
`PLAYWRIGHT_MCP_EXTENSION_TOKEN`, and run `browser_agent.agent.flat --extension`. The controlled tab
lives in the real Chrome profile, so navigations reuse existing logins — **confirmed by reading
the live signed-in Google account** off google.com. (Tab-picker handling and the
"navigate-the-controlled-tab" model are documented in README.)

> Path A alternative (browser-use over CDP) — not used; kept as a note: launch Chrome with
> `--remote-debugging-port=9222 --user-data-dir=...`, then pass `cdp_url` to a `BrowserSession`.
> Requires closing/relaunching Chrome, which the extension avoids.

> First live runs were **read-only on low-risk pages** (example.com, google.com read), never
> email/banking, before trusting it — as advised.

**Exit criteria:** the agent completes a task on a site where you're already logged in,
without re-authenticating.

---

## Phase 5 — Safety & human-in-the-loop

Acting in your authenticated session raises the stakes — lock this down before going further.

- [x] **Approval gate**: a subgoal with `needs_approval=true` pauses for the user before its
      reasoner loop starts (`approve` callback; CLI prompts y/N). `ask_human` action also pauses.
- [x] **Domain allowlist**: `--allow github.com,google.com`. `domain_allowed()` blocks any
      action whose current/target host isn't listed (subdomain-aware). Tested deterministically.
- [x] **Prompt-injection awareness**: reasoner + verifier prompts state page text is untrusted
      DATA, never instructions. Consequential subgoals stay behind the approval gate.
- [x] **Never auto-confirm**: nothing consequential runs without the gate; the verifier won't
      take the model's word that a subgoal is done.

**Exit criteria:** the agent reliably stops and asks before anything consequential.

---

## Phase 6 — Vision fallback & robustness

- [x] **Vision fallback** — `snapshot_is_sufficient()` detects an empty/opaque DOM snapshot;
      `screenshot_data_url()` then attaches a PNG to the reasoner as multimodal `image_url`
      (Scout is multimodal). `--no-vision` disables it. Proven live: on a canvas page that
      paints "BANANA" (empty a11y tree), Scout read the word off the screenshot (`test_vision_e2e.py`).
- [x] **Retries / defensive parsing** — Groq SDK retries (`max_retries`), the `tool_use_failed`
      400 retry loop, and `extract_json()` tolerating fences/prose.
- [x] **Budgets exposed** — `--max-steps` (per subgoal) and `--max-replans`, tuned from runs.
- [x] **Eval set** — `scripts/eval_set.py`: 6 answer-checkable tasks over one shared browser, prints a
      success rate. Current: **6/6 (100%)** on the local test page.

**Exit criteria:** consistent success on your eval set; graceful failure (asks for help)
rather than silent wrong actions.

---

## Phase 7 — (Optional) go local or scale

- [ ] **Privacy / no rate limits**: swap the Groq endpoint for a **local Ollama** model
      (Qwen3-30B-A3B or GLM-4.5-Air) — same OpenAI-compatible interface, so only the
      `base_url`/`api_key`/`model` change. Move to **vLLM** for speed/concurrency.
- [ ] **Two-model split**: a stronger model for planning (runs rarely), a fast one for the
      reasoner (every step).
- [ ] **Cloud browsers** (Browserbase / Steel) only if you ever need many parallel sessions.

---

## Milestone checklist

- [x] M1 — Env ready, Groq key works (Phase 0)
- [x] M2 — Hello-world agent completes a task (Phase 1) — `browser_agent.agent.flat`
- [x] M3 — You can read and explain a run trace (Phase 2)
- [x] M4 — Custom planner/reasoner with verify + re-plan (Phase 3) — `browser_agent.agent.main`
- [x] M5 — Runs in your real logged-in profile (Phase 4) — `--extension`
- [x] M6 — Approval gate + allowlist enforced (Phase 5)
- [x] M7 — Passes your eval set (6/6) with vision fallback (Phase 6)

---

## Risk register

| Risk | Mitigation |
|---|---|
| Prompt injection via page content | Treat page text as untrusted; approval gate; domain allowlist |
| Acting wrongly in your logged-in accounts | Start on low-risk sites; never auto-confirm; human approval |
| Scout tool-call/JSON wobble | System-prompt rules (omit optional params, correct types); catch Groq `tool_use_failed` 400 and retry with the error fed back (capped); `extract_json()` tolerates fences/prose |
| Groq free-tier rate limits | Backoff/retry; Phase 7 local fallback |
| Page changes break a step | Re-index every loop; verify before advancing; re-plan on escalate |
| Runaway loops / cost | Step + re-plan budgets; loop detection |

---

## What to do right now

Phases 0–6 are **done** (see the status table up top) — M1–M7 all met. The agent plans,
executes a verified per-subgoal loop, recovers via re-plan, runs in the real logged-in profile,
is gated by approval + allowlist, falls back to vision on DOM-opaque pages, and passes its eval
set 6/6. Only the **optional Phase 7** (local model via Ollama/vLLM, two-model split, cloud
browsers) remains. Run any task with:

```powershell
.\.venv\Scripts\python.exe -m browser_agent.agent.main "your goal here"            # smoke (own browser)
.\.venv\Scripts\python.exe -m browser_agent.agent.main --extension --allow github.com "your goal"
```

---

## Appendix A — Planner + Reasoner reference (prompts, I/O, loop)

The full content Phase 3 builds on: both system prompts, their JSON schemas, the
observation format, and the loop wiring.


A working reference for the two-brain design of a browser agent. The planner sets the
agenda (semantic subgoals); the reasoner picks one atomic action per loop against the live
page. Both can be backed by the same model — what differs is the prompt and the inputs.

---

### 1. The planner

Runs rarely: once at the start, then only on a re-plan escalation. Reasons about the *task*,
never the raw page.

#### Inputs you feed it
- `goal` — the user's request, verbatim.
- `progress` — a short summary of completed subgoals (omit on first run).
- `observations` — a *compressed* note of what's been learned about the site so far
  (e.g. "site requires login", "no price slider, only a sort dropdown"). Never raw HTML.
- `capabilities` — the action types the reasoner can perform, so the planner only proposes
  things that are executable.

#### System prompt

```
You are the PLANNER for a browser automation agent. You decompose a user goal into an
ordered list of semantic subgoals. You do NOT interact with web pages and you do NOT
choose clicks — a separate reasoner executes each subgoal against the live page.

Rules:
- Each subgoal is a milestone a human would recognize ("log in", "search for the item",
  "filter results", "add to cart"), NOT an atomic click or a CSS selector.
- Order subgoals by dependency. Do not place a subgoal before its prerequisites.
- Give every subgoal a success_condition: an observable state that proves it is done.
- Set needs_approval=true for any subgoal that is irreversible or consequential
  (purchases, deletions, sending messages, changing account settings).
- Keep the plan short. Prefer 3-7 subgoals. Do not over-decompose.
- The reasoner can only perform these action types: {capabilities}.
  Never propose a subgoal that cannot be reached with them.

Respond with ONLY valid JSON, no prose, in this schema:
{
  "subgoals": [
    {
      "id": <int>,
      "goal": "<imperative description>",
      "success_condition": "<observable state>",
      "needs_approval": <bool>
    }
  ]
}
```

#### Re-plan variant (same system prompt, different user message)

When the reasoner escalates, send the planner the original goal, the plan so far, the
subgoal that failed, and the failure reason. Ask it to revise only the *remaining* subgoals.

```
The plan is failing at subgoal {id}: "{goal}".
Reason from the reasoner: "{escalation_reason}".
Here is what has been observed: {observations}.
Revise the remaining subgoals (keep completed ones untouched). Same JSON schema.
```

#### Example output

```json
{
  "subgoals": [
    {"id": 1, "goal": "Open the shop home page",
     "success_condition": "Home page loaded with a visible search box",
     "needs_approval": false},
    {"id": 2, "goal": "Search for a phone case",
     "success_condition": "A product results list is visible",
     "needs_approval": false},
    {"id": 3, "goal": "Restrict results to items under $20",
     "success_condition": "Visible results all show a price under $20",
     "needs_approval": false},
    {"id": 4, "goal": "Add a suitable case to the cart",
     "success_condition": "Cart count increased to 1",
     "needs_approval": false},
    {"id": 5, "goal": "Place the order",
     "success_condition": "Order confirmation page is shown",
     "needs_approval": true}
  ]
}
```

---

### 2. The reasoner

Runs every loop iteration. Reasons about the *page*, not the whole task. It sees exactly
one subgoal at a time and outputs exactly one action (ReAct style: a thought + an action).

#### Inputs you feed it (every loop)
- `subgoal` + `success_condition` — the current target only.
- `observation` — the serialized live page (see section 3).
- `recent_actions` — the last 3-5 actions and their outcomes, so it can self-correct.
- The action vocabulary, baked into the system prompt.

#### System prompt

```
You are the REASONER for a browser automation agent. Given the current subgoal and a
snapshot of the live page, you choose the SINGLE next action that makes progress.

You see the page as a numbered list of interactive elements. Refer to elements by their
[index]. Never invent an index that is not in the list.

Available actions:
- navigate(url)            go to a URL
- click(index)             click element [index]
- type(index, text)        type text into element [index]
- select(index, value)     choose an option in dropdown [index]
- scroll(direction)        "up" or "down" to reveal more elements
- extract(query)           read information from the page (returns text to you)
- wait(reason)             pause for the page to settle, then re-observe
- ask_human(question)      pause and ask the user (use for approvals / ambiguity)
- subgoal_complete(note)   declare the success_condition is met
- escalate(reason)         you are stuck; hand control back to the planner

Rules:
- Think first, then act. Your thought should reference what you see on THIS page.
- Output exactly one action. Do not chain actions.
- Before declaring subgoal_complete, confirm the success_condition is actually visible.
- If an element you expected is missing, an action had no effect twice, or you detect a
  loop, call escalate(reason) instead of guessing repeatedly.
- If the current subgoal is consequential and marked for approval, call ask_human first.

Respond with ONLY valid JSON in this schema:
{
  "thought": "<one or two sentences grounded in the current page>",
  "action": { "type": "<action>", ...args }
}
```

#### Example output

```json
{
  "thought": "The search box is element [1] and it's empty. I'll type the query, then submit next turn.",
  "action": { "type": "type", "index": 1, "text": "phone case" }
}
```

---

### 3. The observation format

This is the single most important thing to get right — it's what grounds the reasoner.
Serialize only *interactive and salient* elements, each with a stable index the action can
reference. Do not dump raw HTML. A compact, indexed list beats both raw DOM and screenshots
for token cost and reliability (add a screenshot only as a fallback for canvas/visual cases).

Build this fresh every loop and pass it as the reasoner's user message:

```
### Current subgoal
Search for a phone case
Success when: a product results list is visible

### Page
URL:   https://shop.example.com/
Title: Home — Example Shop

### Interactive elements
[0]  <button>  Accept cookies
[1]  <input>   placeholder="Search products..."
[2]  <button>  Search
[3]  <a>       Cart (0)
[4]  <a>       Deals
[5]  <select>  Sort by: Relevance

### Recent actions
1. navigate("https://shop.example.com")  -> ok, page loaded
2. click([0])                            -> ok, cookie banner dismissed

### Notes
- Refer to elements by their [index] only.
- Output exactly one action.
```

Practical tips for generating it:
- Walk the DOM (or accessibility tree) for clickable / typeable / selectable nodes; assign
  each a fresh integer index and keep an index→element map on your side to resolve actions.
- Include the element's role/tag, its visible text or label, and one disambiguating
  attribute (placeholder, aria-label). Drop everything else.
- Truncate long lists; if the target might be off-screen, that's what `scroll` is for.
- Re-index every loop — the page changes, so last turn's [3] may not be this turn's [3].

---

### 4. Wiring them into the loop

```python
def run_agent(goal, browser, planner_llm, reasoner_llm,
              max_subgoal_steps=15, max_replans=3):
    observations = []                      # compressed, cross-subgoal learnings
    plan = planner_llm.plan(goal, capabilities=ACTION_TYPES)
    replans = 0
    i = 0

    while i < len(plan.subgoals):
        sub = plan.subgoals[i]

        # Approval gate before consequential subgoals.
        if sub.needs_approval and not ask_human_approval(sub.goal):
            return "Stopped: user declined approval."

        completed = False
        for _ in range(max_subgoal_steps):
            obs = serialize_page(browser, sub, recent_actions=last_actions())
            step = reasoner_llm.decide(obs)        # -> {thought, action}
            action = step["action"]

            if action["type"] == "subgoal_complete":
                if verify(browser, sub.success_condition):   # double-check, don't trust the model
                    completed = True
                    break
                else:
                    record_action("subgoal_complete", "rejected: condition not met")
                    continue

            if action["type"] == "escalate":
                break                                # falls through to re-plan below

            if action["type"] == "ask_human":
                answer = ask_human(action["question"])
                record_action("ask_human", answer)
                continue

            result = execute(browser, action)       # click/type/navigate/...
            record_action(action, result)
            if action["type"] == "extract":
                observations.append(result)          # remember what we learned

        if completed:
            i += 1                                    # advance to next subgoal
            continue

        # Not completed: re-plan the remaining subgoals.
        if replans >= max_replans:
            return "Failed: exhausted re-plan budget."
        replans += 1
        reason = last_escalation_reason() or "subgoal did not complete in step budget"
        plan = planner_llm.replan(goal, plan, failed_id=sub.id,
                                  reason=reason, observations=observations)
        # re-plan returns a fresh list; keep i pointing at the first unfinished subgoal

    return "Done."
```

#### Where each role's inputs and outputs live (recap)
- Planner **in**: goal, progress, compressed observations, capabilities.
  Planner **out**: ordered subgoals + success conditions + approval flags.
- Reasoner **in**: one subgoal, the live serialized observation, recent actions.
  Reasoner **out**: one thought + one action.
- The **verifier** is deliberately not the model's say-so: `verify()` checks the page state
  independently before a subgoal is allowed to advance.
- **State you persist between loops**: the plan, the index of the current subgoal, the
  recent-action ring buffer, and the `observations` list. That's your defense against
  context bloat — you never re-feed the whole history, only these summaries plus the
  current page.

---

### 5. Knobs worth tuning
- **Step budget per subgoal** (`max_subgoal_steps`) and **re-plan budget** (`max_replans`):
  your guardrails against infinite loops and runaway cost.
- **Recent-action window**: 3-5 is usually enough for self-correction; more just adds tokens.
- **Same model vs two models**: a cheaper/faster model often suffices for the reasoner's
  per-step action choice, with a stronger model reserved for planning and re-planning.
- **Loop detection**: if the last N actions repeat with no page change, force an escalate.
```
