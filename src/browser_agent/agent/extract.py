"""Explore-step extraction (B3): turn a page snapshot into a structured candidate list."""

import json

from browser_agent.agent.prompts import EXTRACTOR_SYSTEM, SYNTHESIZER_SYSTEM
from browser_agent.config import COMPOSITION_MODEL, MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


async def extract_candidates(groq, snapshot_text, explore_spec, *, model=MODEL) -> list[dict]:
    spec = explore_spec or {}
    # Read deep into the snapshot: on retail pages the product grid sits BELOW a large nav/filter
    # sidebar, so a tight head-slice cut the actual listings and yielded 0 candidates. The explore
    # path passes the FULL untruncated snapshot (full_snapshot), so allow a generous window — Scout
    # has a large context and products on Amazon/Flipkart appear well past the first 12k chars.
    user = (f"Spec: {json.dumps(spec)}\n\n"
            f"Page snapshot:\n{snapshot_text[:40000]}\n\n"
            "Return the candidates as JSON.")
    resp = await groq.chat.completions.create(
        model=model, temperature=0.0, max_completion_tokens=1200,
        messages=[{"role": "system", "content": EXTRACTOR_SYSTEM}, {"role": "user", "content": user}],
    )
    try:
        data = extract_json(resp.choices[0].message.content)
    except Exception as e:
        log.warning("extractor could not parse response (%r); 0 candidates", e)
        return []
    cands = data.get("candidates", data) if isinstance(data, dict) else data
    result = [c for c in cands if isinstance(c, dict)] if isinstance(cands, list) else []
    log.debug("extractor parsed %d candidate(s)", len(result))
    return result


async def synthesize_options(groq, goal, candidates, selected=None, *, model=COMPOSITION_MODEL) -> str:
    """Turn gathered candidates into a structured markdown shortlist (pros/cons/offers + a
    recommendation) for the user. Returns plain markdown — this becomes the run's final answer."""
    if not candidates:
        return "I couldn't gather any options to compare. The sources may have blocked browsing " \
               "or returned no matching results."
    user = (f"User goal: {goal}\n\n"
            f"Candidate options (JSON, across sources):\n{json.dumps(candidates[:12])[:6000]}\n\n"
            f"Scorer's top pick: {json.dumps(selected) if selected else '(none — you decide)'}\n\n"
            "Write the markdown shortlist now.")
    resp = await groq.chat.completions.create(
        model=model, temperature=0.2, max_completion_tokens=1300,
        messages=[{"role": "system", "content": SYNTHESIZER_SYSTEM}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()
