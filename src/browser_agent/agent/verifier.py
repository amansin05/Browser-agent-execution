"""Verifier — an independent Scout call that rules whether a success condition holds."""

from browser_agent.agent.prompts import VERIFIER_SYSTEM
from browser_agent.config import MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


async def verify_success(groq, snapshot_text, success_condition, *, model=MODEL) -> tuple[bool, str]:
    user = f"Success condition: {success_condition}\n\nCurrent page snapshot:\n{snapshot_text[:8000]}"
    resp = await groq.chat.completions.create(
        model=model, temperature=0.0, max_completion_tokens=300,
        messages=[{"role": "system", "content": VERIFIER_SYSTEM}, {"role": "user", "content": user}],
    )
    try:
        data = extract_json(resp.choices[0].message.content)
        return bool(data.get("satisfied", False)), str(data.get("reason", ""))
    except Exception as e:
        log.warning("verifier could not parse response (%r): %.120r", e, resp.choices[0].message.content)
        return False, f"verifier parse error: {e}"
