"""Tool vocabulary for the two-tier reasoner.

The reasoner acts with the real Playwright MCP tools PLUS three synthetic control actions
(subgoal_complete / escalate / ask_human) that it signals through the same tool-calling channel.
"""

from browser_agent.services.mcp_client import mcp_tools_to_groq

# Never expose these to the agent. browser_close would close tabs (we keep them open across a
# session — see AgentSession); browser_run_code_unsafe is gated off by default anyway.
EXCLUDED_TOOLS = {"browser_close", "browser_run_code_unsafe"}

# Additionally hidden from the TWO-TIER REASONER (but kept for the flat agent): the orchestrator
# already feeds a fresh snapshot to the reasoner every turn (observe()) and, when the DOM is sparse,
# appends clean extracted page text (Readability.js -> trafilatura; see agent/reader.py). Re-exposing
# these lets the reasoner pick them as a "free" no-op action — which it then repeats and trips the
# loop detector, dying at the first subgoal. So the reasoner never calls them itself.
# browser_wait_for is the same trap but worse: when its text never appears it blocks for the full
# 30s MCP timeout AND burns a step, so a few bad waits exhaust the whole subgoal budget (the
# Amazon "Levi's 501 retailers" run died this way). The reasoner gets a fresh snapshot every turn,
# so it never needs to wait — withhold it too.
# browser_evaluate is the SAME trap once more: the reasoner treats it as a "free" way to inspect
# the page, hallucinates a CSS selector (e.g. '.s-result-item.s-asin Pipeline'), gets 0/None back,
# re-runs the identical call, and trips the loop detector — dying at the first explore subgoal
# without ever taking a real action. Candidates are read off the snapshot by extract_candidates,
# not by the reasoner running JS, so the reasoner never needs it — withhold it too. (The
# orchestrator itself still drives browser_evaluate directly for the content extractor and the
# web-grounding step — see agent/reader.py and agent/grounding.py — just never the reasoner.)
# browser_hover is the SAME trap AGAIN: on heavy retail pages (Amazon/Flipkart) the reasoner hovered
# the same element 5+ times to "reveal" content — sometimes even passing a URL as the target (a CSS
# parse error) — burning the whole step budget without ever reading listings (the "i want to buy a
# phone" run died this way on every source). A fresh snapshot already shows whatever a hover would
# reveal on the next turn, and the reasoner can navigate to a category URL or click directly, so it
# never needs to hover — withhold it.
REASONER_EXCLUDED = EXCLUDED_TOOLS | {
    "browser_snapshot", "browser_take_screenshot", "browser_wait_for", "browser_evaluate",
    "browser_hover"}

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


def build_reasoner_tools(listed_tools) -> tuple[list[dict], list[str]]:
    """Reasoner tool schema (MCP minus reasoner-excluded, plus synthetic controls) and the
    capability name list the planner is told about. browser_snapshot/screenshot are withheld —
    see REASONER_EXCLUDED."""
    mcp_names = [t.name for t in listed_tools if t.name not in REASONER_EXCLUDED]
    mcp = [t for t in mcp_tools_to_groq(listed_tools)
           if t["function"]["name"] not in REASONER_EXCLUDED]
    reasoner = mcp + SYNTHETIC_TOOLS
    capabilities = mcp_names + sorted(SYNTHETIC_NAMES)
    return reasoner, capabilities
