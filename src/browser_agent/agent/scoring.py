"""Deterministic candidate scorer (B3 exploit).

The LLM EXPLORES (gathers candidates); this code EXPLOITS (ranks + picks) with a
preference-weighted rubric, because models pick poorly from a known set. Pure + testable.
"""

# Fields where a smaller value is better (so we invert them when normalizing).
LOWER_IS_BETTER = {"price", "delivery_days", "cost", "eta_days", "distance"}

# Fields where a value of 0 (or negative) means MISSING, not "best". A 0.00 price is almost always
# an unparsed/sponsored/Kindle row — treating it as the cheapest is what made an "INR 0.00" book win
# the ranking. We map such zeros to None so they score neutrally instead of best.
ZERO_IS_MISSING = {"price", "cost", "mrp"}


def _to_number(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        digits = "".join(c for c in v if c.isdigit() or c in ".-")
        try:
            return float(digits) if digits not in ("", "-", ".") else None
        except ValueError:
            return None
    return None


def _normalize(values: list[float], lower_is_better: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)  # all equal -> neutral
    norm = [(v - lo) / (hi - lo) for v in values]
    return [1 - n for n in norm] if lower_is_better else norm


def score_candidates(candidates: list[dict], weights: dict[str, float],
                     lower_is_better: set[str] | None = None) -> list[dict]:
    """Return candidates annotated with a 0-1 `_score`, ranked best-first. Each weighted field
    is min-max normalized across the set (inverting lower-is-better fields). Non-numeric or
    missing fields contribute a neutral 0.5 for that candidate."""
    if not candidates:
        return []
    lower = (lower_is_better or set()) | LOWER_IS_BETTER
    fields = [f for f in weights if any(_to_number(c.get(f)) is not None for c in candidates)]

    # Pre-normalize each scored field across the candidate set.
    norm_by_field: dict[str, list[float]] = {}
    for f in fields:
        nums = [_to_number(c.get(f)) for c in candidates]
        if f in ZERO_IS_MISSING:  # a 0/negative price means "unknown", not "free/best"
            nums = [None if (n is not None and n <= 0) else n for n in nums]
        present = [n for n in nums if n is not None]
        if not present:
            continue
        filled = [n if n is not None else sum(present) / len(present) for n in nums]
        norm_by_field[f] = _normalize(filled, f in lower)

    total_w = sum(weights[f] for f in norm_by_field) or 1.0
    ranked = []
    for idx, c in enumerate(candidates):
        score = sum(weights[f] * norm_by_field[f][idx] for f in norm_by_field) / total_w
        ranked.append({**c, "_score": round(score, 4)})
    ranked.sort(key=lambda c: c["_score"], reverse=True)
    return ranked


def is_near_tie(ranked: list[dict], threshold: float = 0.05) -> bool:
    """True if the top two are within `threshold` — a cue to ask the user rather than auto-pick."""
    return len(ranked) >= 2 and (ranked[0]["_score"] - ranked[1]["_score"]) < threshold
