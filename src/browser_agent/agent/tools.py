"""Tool vocabulary for the two-tier reasoner.

The reasoner acts with a fixed, curated action set — the INDEX-ADDRESSED page actions
(INDEX_ACTION_TOOLS, executed in-page via agent/dom_index.py) plus three synthetic control actions
(subgoal_complete / escalate / ask_human). It does NOT get the raw Playwright-MCP element tools:
referring to elements by a stable [index] (resolved to the element's data-ba-id) replaces the
fragile snapshot-ref clicks, and curating the set out of existence also removes the old "free no-op"
traps the reasoner used to loop on (browser_snapshot/screenshot/wait_for/evaluate/hover — each one
burned the step budget without progress on the Amazon/Flipkart runs). The flat agent still gets the
full MCP tool set (minus EXCLUDED_TOOLS).
"""

from browser_agent.services.mcp_client import mcp_tools_to_groq

# Never expose these to the agent. browser_close would close tabs (we keep them open across a
# session — see AgentSession); browser_run_code_unsafe is gated off by default anyway.
EXCLUDED_TOOLS = {"browser_close", "browser_run_code_unsafe"}

SYNTHETIC_TOOLS = [
    {"type": "function", "function": {
        "name": "subgoal_complete",
        "description": "Call when the current subgoal's success condition is visibly satisfied.",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Call when stuck, looping, or the page does not match the plan; hands control back to the planner.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "ask_human",
        "description": "Pause and ask the user a question (for approvals or ambiguity).",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]
SYNTHETIC_NAMES = {"subgoal_complete", "escalate", "ask_human"}

# The reasoner's INDEX-ADDRESSED action set. These REPLACE the raw Playwright-MCP element tools
# (browser_click / browser_type / browser_select_option with snapshot refs). The reasoner now refers
# to elements by the [index] shown in the indexed-DOM view (agent/dom_index.py); the orchestrator
# resolves that index back to the exact element via its data-ba-id, so a click never depends on a
# stale ref or a guessed CSS selector. Navigation / key presses are dispatched to MCP internally
# (agent/dom_index.execute_action), so the reasoner never touches the raw MCP tool list — which also
# removes the old "free no-op trap" tools (snapshot/screenshot/wait/evaluate/hover) entirely.
INDEX_ACTION_TOOLS = [
    {"type": "function", "function": {
        "name": "click_element",
        "description": "Click the interactive element with the given [index] from the current page view.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer", "description": "The [index] of the element to click."}},
            "required": ["index"]}}},
    {"type": "function", "function": {
        "name": "input_text",
        "description": ("Type text into the input / textarea / contenteditable with the given "
                        "[index]. Set submit=true to press Enter afterward (e.g. to run a search)."),
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"},
            "text": {"type": "string"},
            "submit": {"type": "boolean", "description": "Press Enter after typing."}},
            "required": ["index", "text"]}}},
    {"type": "function", "function": {
        "name": "select_option",
        "description": "Select an option (by value or visible label) in the <select> with the given [index].",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}, "value": {"type": "string"}},
            "required": ["index", "value"]}}},
    {"type": "function", "function": {
        "name": "scroll_page",
        "description": ("Scroll the page by ~one screen to reveal more elements (off-screen elements "
                        "are flagged in the view)."),
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["down", "up"]}}, "required": []}}},
    {"type": "function", "function": {
        "name": "navigate",
        "description": ("Navigate directly to a URL. PREFER this to reach a known page (a category "
                        "or search-results URL) instead of clicking through menus; also use it to go "
                        "back by navigating to the previous URL."),
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
]
# Deliberately small (5 page actions + 3 synthetic controls): a tight, high-frequency set so the
# reasoner picks reliably and emits fewer malformed tool calls. Dropped press_key (input_text's
# submit=Enter covers it) and go_back (navigate to the previous URL covers it).
INDEX_ACTION_NAMES = {t["function"]["name"] for t in INDEX_ACTION_TOOLS}

# An ask_user tool the FLAT agent can call to pop the MCQ modal when the request is ambiguous
# (e.g. "buy a phone" -> which brand?), instead of replying with a clarifying question.
ASK_USER_TOOL = {"type": "function", "function": {
    "name": "ask_user",
    "description": ("Ask the user a multiple-choice clarifying question when the request is "
                    "ambiguous or missing a key detail (brand, model, budget, dates). Provide "
                    "concrete options and allow_free_text so they can type their own answer. "
                    "Prefer this over replying with a question."),
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "label": {"type": "string"}, "detail": {"type": "string"}}}},
        "multi_select": {"type": "boolean"},
        "allow_free_text": {"type": "boolean"},
    }, "required": ["question"]}}}


def flat_tools(listed_tools) -> list[dict]:
    """Groq tool schema for the flat agent (MCP tools minus excluded, plus the ask_user MCQ)."""
    return [t for t in mcp_tools_to_groq(listed_tools)
            if t["function"]["name"] not in EXCLUDED_TOOLS] + [ASK_USER_TOOL]


def build_reasoner_tools(listed_tools=None) -> tuple[list[dict], list[str]]:
    """The reasoner's fixed, curated action set: the index-addressed page actions
    (INDEX_ACTION_TOOLS, executed via agent/dom_index.py) plus the synthetic controls, and the
    capability-name list the planner is told about. This REPLACES the old "MCP tools minus the
    no-op traps" scheme — the reasoner no longer drives raw MCP element tools by snapshot ref, so
    `listed_tools` is accepted only for call-site compatibility and is unused (navigation and key
    presses are dispatched to MCP internally by dom_index.execute_action)."""
    reasoner = INDEX_ACTION_TOOLS + SYNTHETIC_TOOLS
    capabilities = sorted(INDEX_ACTION_NAMES) + sorted(SYNTHETIC_NAMES)
    return reasoner, capabilities
