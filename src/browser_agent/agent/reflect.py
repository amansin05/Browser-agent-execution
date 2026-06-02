"""Memory-update reflection (B2/M11): summarize a finished task into an episode + preference
signals, run as a cheap LLM call after the task completes."""

from browser_agent.agent.prompts import REFLECT_SYSTEM
from browser_agent.config import MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


async def reflect(groq, task, result, observations, *, model=MODEL) -> dict:
    notes = "\n".join(f"- {o}" for o in (observations or [])[-8:]) or "(none)"
    user = (f"Task: {task}\n"
            f"Result: {result}\n"
            f"Notes gathered:\n{notes}\n\n"
            "Write the memory record as JSON.")
    try:
        resp = await groq.chat.completions.create(
            model=model, temperature=0.0, max_completion_tokens=400,
            messages=[{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": user}],
        )
        data = extract_json(resp.choices[0].message.content)
    except Exception as e:
        log.warning("reflect failed (%r); recording 'unknown' outcome", e)
        return {"outcome": "unknown", "preference_updates": {}}
    if not isinstance(data, dict):
        return {"outcome": "unknown", "preference_updates": {}}
    data.setdefault("outcome", "unknown")
    pu = data.get("preference_updates")
    data["preference_updates"] = pu if isinstance(pu, dict) else {}
    return data
