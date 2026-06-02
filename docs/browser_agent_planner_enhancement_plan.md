# Browser Agent — Planner Enhancement Plan

Turns a basic planner into a personalized, context-aware, learning planner. It still acts as
"the manager" — it never touches the page — but now plans using *who you are*, *where/when/how
you're working*, and *what you've preferred before*, and it knows when to explore options, when
to ask you, and when to stop for approval.

> Prerequisite: a working basic planner/reasoner loop (plan subgoals → execute each as a
> verified observe/reason/act loop → re-plan on failure). Build that first, then layer these on.

---

This upgrades the basic planner into a personalized, context-aware planner.
It is still "the manager" — it never touches the page — but now it plans using *who you are*,
*where/when/how you're working*, and *what you've preferred before*, and it knows when to
explore options, when to ask you, and when to stop and get approval.

Build this on top of a working basic planner. Do not start here.

## B0. The big design rule (from research)

**LLMs explore well but exploit poorly.** A model is good at *generating/gathering* candidate
options, but bad at reliably *picking the single best* from a known set (it underperforms a
plain weighted score). So:

- Use the **LLM to EXPLORE** — gather and propose candidate options.
- Use **explicit code + a preference-weighted rubric to EXPLOIT** — rank and choose.
- This is a *proposer–verifier* split. The planner proposes; a deterministic scorer verifies.

Everything below follows from separating those two jobs.

---

## B1. Context fed to the planner — two tiers

### Tier 1 — Basic context (ephemeral, rebuilt fresh every run)
Never stored long-term; gathered at task start and injected into the planner prompt.

```json
{
  "time":     { "now": "2026-06-02T18:30:00+05:30", "tz": "Asia/Kolkata", "day": "Tuesday" },
  "location": { "city": "Chennai", "country": "IN", "currency": "INR" },
  "device":   { "type": "desktop", "os": "Windows", "browser": "Chrome", "screen": "1920x1080" },
  "profile":  { "id": "default", "logged_in_sites": ["github.com", "amazon.in"] }
}
```

Why each matters to planning: **time** gates "is the store open / is same-day delivery
possible"; **location/currency** sets shipping and price context; **device** changes the UI
the reasoner will see (mobile vs desktop layouts); **profile** tells the planner which sites
it can act on without a login subgoal.

### Tier 2 — Personal memory (persistent, updated over time)
Stored in a small local store (start with a JSON file or SQLite; graduate to a memory lib like
Mem0/Letta later). Split by *lifecycle* — different facts change at different rates:

```json
{
  "persona": {                         // stable; changes rarely
    "name": "…", "risk_tolerance": "cautious",
    "values": ["prefers eco-friendly", "brand-loyal to X"]
  },
  "preferences": {                     // evolving; recency-weighted (see B2)
    "shopping": { "max_price_sensitivity": 0.8, "preferred_brands": ["Anker"],
                  "delivery_speed_weight": 0.6, "min_rating": 4.0 },
    "ui":       { "confirm_before_pay": true, "prefers_dark_sites": false }
  },
  "history": [                         // episodic, append-only, summarized periodically
    { "ts": "2026-05-30", "task": "bought USB-C cable", "outcome": "success",
      "on_time": true, "chosen": "Anker 120W", "rejected": ["generic 60W"] }
  ]
}
```

- **persona** and **preferences** are *core memory blocks* — always injected into the planner
  prompt (small, high-value).
- **history** is *archival/episodic* — do NOT dump it all in the prompt. Retrieve only the few
  records relevant to the current goal (e.g. past shopping tasks when the goal is a purchase).

---

## B2. Updating memory over time (the "update it with time" part)

Run a **memory-update step after each task completes** (a separate cheap LLM call + code):

1. **Append an episodic record** — task, outcome, what was chosen/rejected, whether it met any
   deadline (`on_time`), timestamp. This is append-only and never edited.
2. **Refine preferences with recency weighting.** Don't overwrite on a single observation and
   don't treat one action as ground truth. Blend short- and long-term signal:
   - keep a **sliding window** of recent choices (last N) → captures current behavior,
   - keep an **exponential moving average** per preference → captures lasting tendency,
   - new value = `α · recent + (1-α) · long_term` (start α ≈ 0.3).
   - Example: you pick the cheaper option 4 times running → `max_price_sensitivity` drifts up,
     but one splurge won't flip it.
3. **Guard against staleness & bad memories** (the #1 failure mode):
   - timestamp every preference; **decay confidence** in unused ones,
   - prefer recent evidence when memories conflict,
   - let the user **correct memory directly** ("no, I don't prefer that brand") — a correction
     outweighs many silent observations,
   - periodically **summarize & prune** history so it doesn't bloat.

> Keep persona edits behind explicit confirmation; preferences update automatically; history
> just accumulates. Different lifecycles, different update rules.

---

## B3. Explore → Exploit as planner behavior

When the goal is a **choice under options** (buy something, book something, pick a plan), the
planner emits a three-part structure instead of a single "do it" subgoal:

```json
{
  "subgoals": [
    { "id": 1, "type": "explore",
      "goal": "Gather candidate USB-C chargers matching the user's prefs",
      "explore_spec": { "target_count": 5,
        "must_have": ["120W", "in stock", "ships to Chennai"],
        "gather_fields": ["price", "rating", "brand", "delivery_days", "return_policy"] },
      "success_condition": "At least 3 candidates collected with all fields",
      "needs_approval": false },

    { "id": 2, "type": "exploit",
      "goal": "Score candidates and select the best",
      "scoring": {                       // deterministic — NOT the LLM's gut pick
        "weights": { "price": 0.30, "rating": 0.25, "delivery_days": 0.20,
                     "brand_affinity": 0.15, "return_policy": 0.10 },
        "rule": "normalize each field 0-1 (lower-is-better for price/delivery), weighted sum, max wins" },
      "success_condition": "One option ranked highest; top 3 retained to show user",
      "needs_approval": false },

    { "id": 3, "type": "act", "goal": "Add the selected item to cart",
      "success_condition": "Cart shows the chosen item", "needs_approval": false },

    { "id": 4, "type": "act", "goal": "Place the order",
      "success_condition": "Order confirmed", "needs_approval": true }   // secured — see B5
  ]
}
```

Division of labor:
- **explore** is an LLM/browser job — the reasoner browses, searches, collects candidates into
  a structured list (this is where LLMs are strong).
- **exploit** is a **code** job — a deterministic scorer reads the candidate list and the user's
  preference weights and computes the winner. The weights come straight from Tier-2 preferences,
  so the choice is personalized *and* explainable ("picked Anker: best price-rating-delivery
  blend for your weights").
- If the decision is consequential, the exploit step surfaces the **top 2–3** to the user
  rather than auto-committing (ties into disambiguation, B4).

Tuning knobs: `target_count` (explore breadth), the `weights` (personalization), and an
**explore budget** so it doesn't browse 40 products for a $5 item.

---

## B4. Disambiguation — asking you, with options + free text

The planner emits an `ask_user` step when the goal is **ambiguous** or a decision needs your
input. Always give selectable options **and** a free-text escape hatch (research: the "respond"
decision type — let the human type, don't trap them in buttons).

```json
{ "type": "ask_user",
  "question": "Which charger should I get?",
  "reason": "Two strong candidates, close on your criteria",
  "options": [
    { "id": "a", "label": "Anker 120W — ₹3,499 — 4.6★ — 2-day",  "detail": "best rating" },
    { "id": "b", "label": "Ugreen 100W — ₹2,899 — 4.4★ — 4-day", "detail": "cheapest" }
  ],
  "allow_free_text": true,
  "free_text_hint": "or tell me a different priority (e.g. 'cheapest regardless of brand')" }
```

When to disambiguate (don't over-ask — it's annoying):
- the goal itself is underspecified ("book a flight" — when? where from?),
- the exploit step produced a near-tie among top options,
- an action conflicts with a stored preference,
- a guess would be costly or hard to reverse.

The user's answer feeds back as context; if they pick an option, proceed; if they free-text,
re-plan that subgoal. Capture the choice into memory (B2) so the next similar task asks less.

---

## B5. HITL — secured actions & credentials

Two distinct pause types, both using the **interrupt-before-side-effect** pattern: pause,
persist state, get the human decision (approve / reject / edit / respond), resume from the
saved checkpoint. Approval must come **before** the action, never after.

### B5a. Secured actions — risk tiers
Tag every subgoal with a risk tier; the tier decides whether to pause.

| Tier | Examples | Behavior |
|---|---|---|
| **auto** | search, navigate, read, add-to-cart | proceed, no pause |
| **confirm** | submit a form, post, change a setting | pause, show what it will do, need approval |
| **secured** | **payments**, deletes, sending money/messages, anything irreversible | pause, show full detail, need explicit approval; never auto-confirm |

```json
{ "type": "approval_request",
  "tier": "secured",
  "action_summary": "Place order: Anker 120W charger, ₹3,499, ship to <address>, pay via <card>",
  "reversible": false,
  "allowed_decisions": ["approve", "reject", "edit"],
  "on_reject": "stop and report" }
```

### B5b. Credentials — never store, always ask in the moment
When a subgoal needs a login the active profile doesn't already satisfy:

```json
{ "type": "credential_request",
  "site": "example.com",
  "need": ["username", "password"],
  "note": "I'll use these only for this login and won't store them",
  "prefer": "If possible, log in yourself in the open browser tab and tell me when done" }
```

Best practice: **prefer the profile session you already attached (Phase 4)** so no credentials
are needed at all; if a login is genuinely required, ask the human to type it into the real
browser tab rather than handing secrets to the agent. Never persist credentials in memory.

> Add a **timeout/fallback**: if the human doesn't respond to an approval or credential request
> within a window, the safe default is to abort the subgoal, not to proceed.

---

## B6. The enhanced planner system prompt

Replaces the basic planner prompt once you reach this stage.

```
You are the ENHANCED PLANNER for a personal browser agent. You decompose a user goal into an
ordered list of subgoals, personalized to the user. You never interact with web pages.

You are given:
- BASIC CONTEXT: time, location, device, active profile (and which sites are already logged in).
- PERSONA & PREFERENCES: stable traits and evolving, recency-weighted preferences.
- RELEVANT HISTORY: a few past tasks similar to this goal (may be empty).

Use this context to plan. Examples: skip a login subgoal for already-logged-in sites; use the
user's currency/location for shipping; respect stored preferences when proposing options.

Subgoal types you may emit:
- "explore"  : gather candidate options (you/the reasoner browse). Give an explore_spec.
- "exploit"  : select among gathered candidates. Provide scoring weights derived from the
               user's preferences. (A deterministic scorer, NOT the reasoner, picks the winner.)
- "act"      : a concrete browser milestone (search, fill, add-to-cart, submit, ...).
- "ask_user" : disambiguate. Provide options AND allow_free_text. Use only when genuinely
               ambiguous, on near-ties, on preference conflicts, or when a wrong guess is costly.
- "approval_request"   : pause for human approval. Set tier = auto | confirm | secured.
- "credential_request" : ask for a login the profile can't satisfy; never store secrets.

Rules:
- For any goal that chooses among options (purchase, booking, plan selection), use
  explore -> exploit before the act that commits.
- Set tier="secured" for payments, deletions, sending money/messages, and anything
  irreversible. These ALWAYS require explicit approval before execution.
- Prefer the existing logged-in profile over asking for credentials.
- Don't over-ask: only emit ask_user when the value of clarifying beats the friction.
- Keep the plan short (3-8 subgoals). Each act/explore subgoal needs a success_condition.

Respond with ONLY valid JSON:
{ "subgoals": [ { "id": <int>, "type": "<type>", "goal": "...", ... type-specific fields ... ,
                  "success_condition": "...", "tier": "auto|confirm|secured",
                  "needs_approval": <bool> } ] }
```

---

## B7. How it wires into the loop

- **Before planning**: build Tier-1 basic context; load Tier-2 persona+preferences; retrieve a
  few relevant history records. Inject all into the planner call.
- **On an `explore` subgoal**: run the normal observe→reason→act loop, but the "done" output is
  a structured candidate list, not a page action.
- **On an `exploit` subgoal**: skip the LLM — run the deterministic scorer over the candidates
  with the preference weights; keep top-3.
- **On `ask_user` / `approval_request` / `credential_request`**: this is an `interrupt` — persist
  state, surface the request to the human, block, then resume from the checkpoint with their
  decision. Enforce a response timeout that aborts (not proceeds) on no answer.
- **After the task**: run the memory-update step (B2) — append episodic record, refine
  preferences (sliding-window + EMA), prune/summarize history.

## B8. Build order for these enhancements
1. Tier-1 basic context injection (cheap, immediate value).
2. Tier-2 persona/preferences as a static JSON injected into the planner.
3. HITL tiers + approval interrupts (do this before any real purchases).
4. Credential-request flow (prefer profile session).
5. explore → exploit with the deterministic scorer.
6. ask_user disambiguation with options + free text.
7. Memory-update step (recency-weighted) — the part that makes it improve over time.

Milestones to add to the checklist:
- [x] M8 — Planner uses basic context (time/device/profile) — `agent/context.py` injected into the planner preamble
- [x] M9 — Persona/preferences injected (`memory/profile.py`); HITL tiers enforced in `orchestrate` (confirm/secured need approval, with a no-answer timeout abort)
- [x] M10 — explore→exploit with deterministic scorer (`agent/scoring.py`); ask_user disambiguation (options + free text)
- [x] M11 — Memory updates after each task (`agent/reflect.py` → episode append + EMA preference refine, tested for drift-not-flip)

**Implementation notes (Path B):**
- The enhanced planner prompt (`prompts.ENHANCED_PLANNER`) replaces the basic one; `plan_subgoals`
  normalizes `type`/`tier`/`explore_spec`/`scoring`/`options` (back-compatible defaults).
- `AgentSession` builds the planner preamble = basic context + persona/preferences + relevant
  history + this session's rolling context, and runs the reflect (memory-update) after each task.
- Tier-2 memory (persona/preferences/history) lives in the same SQLite store keyed by a profile
  id ("default"), separate from per-session rolling memory.
- Deterministic tests in `test/test_enhanced.py`: scorer, context, profile EMA/correction/history,
  and subgoal normalization. Full suite: 48 passed.
