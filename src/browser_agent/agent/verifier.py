"""Verifier — an independent Scout call that rules whether a success condition holds."""

import re

from browser_agent.agent.prompts import VERIFIER_SYSTEM
from browser_agent.config import MODEL
from browser_agent.log import get_logger
from browser_agent.utils.text import extract_json

log = get_logger(__name__)

# Amazon product-id (ASIN) in a URL path: /dp/<ASIN>/, /gp/product/<ASIN>, /gp/aw/d/<ASIN>.
_ASIN_RE = re.compile(r"/(?:dp|gp/product|gp/aw/d|gp/offer-listing)/([A-Za-z0-9]{10})")


def _asin(u: str) -> str | None:
    m = _ASIN_RE.search(u or "")
    return m.group(1).upper() if m else None


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


def verify_candidates(candidates, explore_spec=None) -> tuple[bool, str]:
    """Verify a GATHER/EXPLORE subgoal against the structured candidates already read off the live
    DOM — NOT the a11y snapshot. The snapshot-based verifier is blind to the product grid (the
    extractor reads the DOM), so it kept rejecting "gather" steps that had already succeeded, which
    sent the reasoner into a scroll-storm until the step budget. Here, enough real candidates = done.

    Deterministic (no LLM): pass when at least `need` named candidates exist (need is the spec's
    target_count, capped to a modest floor so we never demand more than a real list), with a real
    product signal (a price or rating on at least one) so nav-link noise doesn't count."""
    spec = explore_spec or {}
    rows = [c for c in (candidates or []) if isinstance(c, dict) and (c.get("name") or c.get("title"))]
    if not rows:
        return False, "no candidates gathered yet"
    target = spec.get("target_count")
    need = min(max(int(target), 1), 3) if isinstance(target, (int, float)) else 3
    if len(rows) < need:
        return False, f"only {len(rows)} candidate(s) gathered; need at least {need}"
    has_signal = any(c.get("price") not in (None, "") or c.get("rating") not in (None, "") for c in rows)
    if not has_signal:
        return False, f"{len(rows)} rows but none has a price/rating — not a real product list yet"
    return True, f"{len(rows)} candidates gathered (price/rating present)"


def verify_product_match(current_url, page_title, selected) -> tuple[bool, str]:
    """Confirm the loaded page is the SELECTED product — not merely 'a product page'. Compares the
    ASIN in the current URL with the selected candidate's (its `asin`, or the ASIN in its `url`); a
    clear mismatch REJECTS (the 'loaded a OnePlus page for a book' bug a generic check let through).
    Falls back to a title-token overlap when ASINs aren't both available. Returns (matches, reason).
    Non-blocking (True) when there's nothing to compare against."""
    if not (isinstance(selected, dict) and (selected.get("url") or selected.get("asin"))):
        return True, "no selected product to match against"
    want = (str(selected.get("asin") or "").upper() or _asin(selected.get("url") or ""))
    have = _asin(current_url or "")
    if want and have:
        return (True, f"on the selected product (ASIN {have})") if want == have \
            else (False, f"WRONG product: page ASIN {have} != selected {want}")
    # No comparable ASIN -> match the selected title against the page title (token overlap).
    title = str(selected.get("name") or selected.get("title") or "").lower()
    terms = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 2][:6]
    pt = (page_title or "").lower()
    if terms and pt:
        hits = sum(1 for t in terms if t in pt)
        if hits >= max(1, len(terms) // 2):
            return True, f"page title matches the selected product ({hits}/{len(terms)} terms)"
        return False, f"page title {page_title[:50]!r} doesn't match selected {title[:50]!r}"
    return True, "could not compare identity (no ASIN/title) — not blocking"
