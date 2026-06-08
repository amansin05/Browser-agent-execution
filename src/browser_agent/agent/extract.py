"""Explore-step extraction (B3): turn a page snapshot into a structured candidate list."""

import json
import re

from browser_agent.agent.prompts import EXTRACTOR_SYSTEM, SELECTOR_SYSTEM, SYNTHESIZER_SYSTEM
from browser_agent.config import COMPOSITION_MODEL, MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)

# Stopwords stripped from the goal when matching a candidate's title (fix 6 relevance filter).
_GOAL_STOP = {"buy", "order", "get", "find", "me", "a", "an", "the", "and", "or", "of", "for", "to",
              "in", "on", "online", "please", "want", "need", "book", "new", "best", "cheap",
              "cheapest", "under", "with", "my", "some", "from", "at", "is", "it"}
# Markers that a listing is a multi-item BUNDLE, not the single title the user asked for.
_BUNDLE_MARKERS = ("set of", "pack of", "combo", "bundle", "(set", "books)", "2 books", "3 books",
                   "4 books", "collection of")

# A spending ceiling stated in the goal — "under 6000", "below ₹5999", "less than 5000", "within
# 6000", "upto 6000", "budget of 6000", "< 5000", "max 6000". We drop candidates priced over it
# BEFORE ranking so "under 6000" actually means under 6000 (the scorer otherwise only down-weights
# price, letting an over-budget item still win).
_CEILING_RE = re.compile(
    r"(?:under|below|less than|cheaper than|within|up\s?to|upto|max(?:imum)?|budget(?:\s+of)?|<=?|≤)"
    r"\s*(?:rs\.?|inr|₹|\$|usd|eur|€|gbp|£)?\s*([\d][\d,]*(?:\.\d+)?)", re.I)


def _price_ceiling(goal: str):
    """The numeric spending ceiling stated in `goal` (e.g. 6000 for 'asics under ₹6000'), or None."""
    m = _CEILING_RE.search(goal or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _price_num(price):
    """Parse a price string/number to a float, ignoring currency symbols + thousands separators
    (e.g. '₹6,000' -> 6000.0, 'Rs.5,499' -> 5499.0). None when no number is present."""
    if price is None or price == "":
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(price))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _goal_terms(goal: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (goal or "").lower())
            if len(w) > 2 and w not in _GOAL_STOP]


def _name(c: dict) -> str:
    return str(c.get("name") or c.get("title") or "").lower()


def relevance_filter(goal: str, candidates: list[dict]) -> list[dict]:
    """Drop candidates that don't match the goal BEFORE ranking (fix 6): wrong-language / unrelated
    titles (too few of the goal's title words present — e.g. the Portuguese 'Quem pensa enriquece'
    for 'Think and Grow Rich'), multi-book BUNDLES when a single title was asked for, and no-price
    listings when priced options exist. Falls back to the full set if a filter would empty it, so we
    never zero out the candidate pool."""
    terms = _goal_terms(goal)
    if not terms or not candidates:
        return candidates
    need = 1 if len(terms) <= 2 else 2
    kept = [c for c in candidates if sum(1 for t in terms if t in _name(c)) >= need]
    if not kept:
        return candidates                       # goal terms matched nothing — don't over-filter
    singular = not any(w in (goal or "").lower() for w in ("set", "bundle", "combo", "pack", "books"))
    if singular:
        no_bundle = [c for c in kept if not any(b in _name(c) for b in _BUNDLE_MARKERS)]
        kept = no_bundle or kept
    priced = [c for c in kept if c.get("price") not in (None, "")]
    pool = priced or kept
    # Budget ceiling from the goal ("under ₹6000"): drop priced candidates ABOVE it. Unpriced rows
    # pass (price unknown, not "over budget"). Falls back to the full pool if this would empty it.
    ceiling = _price_ceiling(goal)
    if ceiling is not None:
        within = [c for c in pool
                  if _price_num(c.get("price")) is None or _price_num(c.get("price")) <= ceiling]
        if len(within) != len(pool):
            log.debug("budget filter (<= %.0f): %d -> %d candidates", ceiling, len(pool), len(within))
        pool = within or pool
    return pool


async def _chat_json(groq, system, user, *, model, max_completion_tokens, attempts=3):
    """Call Scout for a JSON reply and parse it, RETRYING on a parse wobble — feed the bad reply +
    the error back and ask for clean JSON, capped at `attempts`. Mirrors the planner's retry and the
    flat agent's tool_use_failed recovery so ONE malformed reply never silently becomes zero results
    (the old `except: return []` was exactly the "0 candidates" failure). Raises the last parse error
    if every attempt fails; the caller decides the fallback."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for attempt in range(attempts):
        resp = await groq.chat.completions.create(
            model=model, temperature=0.0, max_completion_tokens=max_completion_tokens, messages=messages)
        content = resp.choices[0].message.content
        try:
            return extract_json(content)
        except (ValueError, TypeError) as e:
            log.warning("JSON parse failed (attempt %d/%d): %r", attempt + 1, attempts, e)
            last_err = e
            messages.append({"role": "assistant", "content": content or ""})
            messages.append({"role": "user", "content":
                             "That could not be parsed as the required JSON (it may have been "
                             "truncated or wrapped in prose). Reply with ONLY the JSON for the "
                             "schema — no prose, no code fence, no <think>."})
    raise last_err


async def extract_candidates(groq, snapshot_text, explore_spec, *, model=MODEL) -> list[dict]:
    spec = explore_spec or {}
    # Read deep into the snapshot: on retail pages the product grid sits BELOW a large nav/filter
    # sidebar, so a tight head-slice cut the actual listings and yielded 0 candidates. The explore
    # path passes the FULL untruncated snapshot (full_snapshot), so allow a generous window — Scout
    # has a large context and products on Amazon/Flipkart appear well past the first 12k chars.
    # NOTE: this is now the FALLBACK reader — the explore path tries DOM extraction (agent/dom_extract)
    # first; this LLM/a11y pass only runs when the deterministic DOM reader found nothing.
    user = (f"Spec: {json.dumps(spec)}\n\n"
            f"Page snapshot:\n{snapshot_text[:40000]}\n\n"
            "Return the candidates as JSON.")
    try:
        data = await _chat_json(groq, EXTRACTOR_SYSTEM, user, model=model, max_completion_tokens=1200)
    except Exception as e:  # only after the bounded retries — never on the first wobble
        log.warning("extractor could not parse response after retries (%r); 0 candidates", e)
        return []
    cands = data.get("candidates", data) if isinstance(data, dict) else data
    result = [c for c in cands if isinstance(c, dict)] if isinstance(cands, list) else []
    log.debug("extractor parsed %d candidate(s)", len(result))
    return result


async def select_candidates(groq, goal, candidates, *, top_n=3, model=MODEL) -> dict:
    """LLM-based ranking/selection (replaces the deterministic scorer as the decider). Returns
    {"selected": <best dict | None>, "top": [<dicts>], "reason": str}. The model picks by index from
    the gathered set; on any failure we fall back to the rating-dominant deterministic scorer so a
    selection is always produced."""
    if not candidates:
        return {"selected": None, "top": [], "reason": "no candidates"}
    # Fix 6: filter to goal-relevant candidates (right title/language/format, has a price) BEFORE
    # ranking, so the selector can't pick a wrong-language edition or a bundle just because it has the
    # most reviews. Both the LLM selector and the deterministic fallback operate on this pool.
    pool = relevance_filter(goal, candidates)
    if len(pool) != len(candidates):
        log.debug("relevance filter: %d -> %d candidates for %r", len(candidates), len(pool), goal[:60])
    user = (f"Goal: {goal}\n\nCandidates (0-based index):\n{json.dumps(pool[:20])[:6000]}\n\n"
            f"Pick up to {top_n} best, best first.")
    try:
        data = await _chat_json(groq, SELECTOR_SYSTEM, user, model=model, max_completion_tokens=300)
        idxs = [i for i in (data.get("top") or []) if isinstance(i, int) and 0 <= i < len(pool)]
        if idxs:
            top = [pool[i] for i in idxs][:top_n]
            return {"selected": top[0], "top": top, "reason": str(data.get("reason", "")).strip()}
        log.debug("selector returned no usable indices; falling back to scorer")
    except Exception as e:  # only after bounded retries
        log.warning("LLM selection failed after retries (%r); falling back to the deterministic scorer", e)
    # Fallback: deterministic, rating-dominant scorer over the SAME relevance-filtered pool.
    from browser_agent.agent.scoring import score_candidates
    ranked = score_candidates(pool, {"rating": 0.6, "review_count": 0.2, "price": 0.2})
    return {"selected": ranked[0] if ranked else None, "top": ranked[:top_n],
            "reason": "ranked by rating (deterministic fallback)"}


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
