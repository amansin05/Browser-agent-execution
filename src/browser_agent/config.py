"""Shared configuration constants."""

# Groq models, split by ROLE:
# - PLANNER: a thinking model — it reasons about how to decompose the goal into subgoals.
PLANNER_MODEL = "qwen/qwen3-32b"
# - REASONING / STRUCTURED: fast, tool-call-capable. Used for the reasoner's per-step tool calls
#   and every structured-JSON brain (verifier, candidate extractor, reflection) + the flat loop.
REASONING_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# - COMPOSITION: plain-text / final prose (the markdown shortlist a user reads). A larger model
#   writes better.
COMPOSITION_MODEL = "openai/gpt-oss-120b"

# Back-compat alias: the generic "reasoning" model that most role functions default to. Existing
# signatures say `model=MODEL`; only the planner and the composer override to their own model.
MODEL = REASONING_MODEL

# Cap on how much of a single tool result (e.g. a huge page snapshot) we feed back to the
# model — protects the context window and Groq's free-tier token limits.
MAX_TOOL_RESULT_CHARS = 12_000
