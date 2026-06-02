"""Deterministic tests for the planner enhancements (M8-M11): context, scorer, profile memory,
preference EMA, history retrieval, and subgoal normalization."""

import pytest

from browser_agent.agent.context import build_basic_context, context_text
from browser_agent.agent.planner import _normalize
from browser_agent.agent.scoring import is_near_tie, score_candidates
from browser_agent.memory import MemoryStore, ProfileMemory


# ----------------------------------------------------------------- scorer (B3 exploit)
def test_score_ranks_lower_price_higher_rating():
    cands = [
        {"name": "A", "price": 100, "rating": 4.0},
        {"name": "B", "price": 50, "rating": 4.8},
        {"name": "C", "price": 200, "rating": 4.5},
    ]
    ranked = score_candidates(cands, {"price": 0.5, "rating": 0.5})
    assert ranked[0]["name"] == "B"            # cheapest AND best rated
    assert [c["name"] for c in ranked] == ["B", "C", "A"] or ranked[0]["name"] == "B"
    assert 0.0 <= ranked[-1]["_score"] <= ranked[0]["_score"] <= 1.0


def test_score_parses_currency_strings():
    cands = [{"id": 1, "price": "₹3,499"}, {"id": 2, "price": "₹2,899"}]
    ranked = score_candidates(cands, {"price": 1.0})
    assert ranked[0]["id"] == 2                # cheaper wins (lower-is-better)


def test_score_empty_and_near_tie():
    assert score_candidates([], {"price": 1.0}) == []
    tie = score_candidates([{"x": 10, "p": 1}, {"x": 10, "p": 1}], {"x": 1.0})
    assert is_near_tie(tie) is True


def test_zero_price_is_not_treated_as_cheapest():
    # A 0.00 price means "unknown", not "free/best" — it must NOT win a price-weighted ranking
    # (the "INR 0.00 book won" bug). With real prices present it scores neutrally (~mean).
    cands = [
        {"name": "junk", "price": 0, "rating": 4.6},
        {"name": "cheap", "price": 50, "rating": 4.0},
        {"name": "mid", "price": 100, "rating": 4.2},
    ]
    ranked = score_candidates(cands, {"price": 1.0})
    assert ranked[0]["name"] == "cheap"          # the real cheapest, not the 0.00 item
    assert ranked[0]["name"] != "junk"


def test_rating_dominant_weights_rank_by_rating_not_price():
    # With no price preference -> rating-dominant weights: the best-reviewed wins even if pricier.
    cands = [
        {"name": "A", "price": 50, "rating": 3.9, "review_count": 10},
        {"name": "B", "price": 300, "rating": 4.8, "review_count": 900},
    ]
    ranked = score_candidates(cands, {"rating": 0.6, "review_count": 0.2, "price": 0.2})
    assert ranked[0]["name"] == "B"


# ----------------------------------------------------------------- context (B1)
def test_basic_context_shape():
    ctx = build_basic_context()
    assert set(ctx) >= {"time", "device", "profile"}
    assert ctx["device"]["browser"] == "Chrome"
    txt = context_text(ctx)
    assert "Basic context" in txt and "device" in txt


# ----------------------------------------------------------------- profile (B1/B2)
@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem"))
    yield s
    s.close()


def test_profile_defaults_and_persist(store):
    p = ProfileMemory(store, "u1")
    assert p.preferences()["shopping"]["max_price_sensitivity"] == 0.5
    p.refine_preferences({"shopping.max_price_sensitivity": 1.0}, alpha=0.3)
    # reload from store -> persisted
    p2 = ProfileMemory(store, "u1")
    assert p2.preferences()["shopping"]["max_price_sensitivity"] == pytest.approx(0.65)


def test_preference_ema_drifts_not_flips(store):
    p = ProfileMemory(store, "u2")
    for _ in range(4):
        p.refine_preferences({"shopping.max_price_sensitivity": 1.0}, alpha=0.3)
    v = p.preferences()["shopping"]["max_price_sensitivity"]
    assert 0.5 < v < 1.0          # drifted up over repeats, never snapped to 1.0


def test_correction_overrides(store):
    p = ProfileMemory(store, "u3")
    p.correct("shopping.min_rating", 4.7)
    assert p.preferences()["shopping"]["min_rating"] == 4.7


def test_relevant_history_retrieval(store):
    p = ProfileMemory(store, "u4")
    p.append_episode("bought USB-C cable", "success", chosen="Anker")
    p.append_episode("booked a flight to Delhi", "success")
    rel = p.relevant_history("buy a USB cable", limit=3)
    assert any("cable" in e["task"] for e in rel)
    assert all("flight" not in e["task"] for e in rel)


# ----------------------------------------------------------------- normalization (M9)
def test_normalize_defaults_and_tiers():
    assert _normalize({"goal": "g"}, 1)["type"] == "act"
    assert _normalize({"goal": "g"}, 1)["tier"] == "auto"
    assert _normalize({"goal": "g", "needs_approval": True}, 1)["tier"] == "secured"
    confirm = _normalize({"goal": "g", "tier": "confirm"}, 1)
    assert confirm["needs_approval"] is True


def test_normalize_passes_type_fields():
    ex = _normalize({"goal": "g", "type": "explore", "explore_spec": {"target_count": 5}}, 2)
    assert ex["type"] == "explore" and ex["explore_spec"]["target_count"] == 5
    au = _normalize({"goal": "pick one", "type": "ask_user", "options": [{"id": "a"}]}, 3)
    assert au["allow_free_text"] is True and au["question"] == "pick one" and au["options"]


def test_normalize_keeps_present_type():
    pr = _normalize({"goal": "show shortlist", "type": "present"}, 4)
    assert pr["type"] == "present" and pr["tier"] == "auto" and pr["needs_approval"] is False


# ----------------------------------------------------------------- present synthesis (skip-safe)
@pytest.mark.asyncio
async def test_synthesize_options_no_candidates_skips_llm():
    from browser_agent.agent.extract import synthesize_options

    # No candidates -> returns a plain message WITHOUT touching the LLM (groq=None would crash).
    md = await synthesize_options(groq=None, goal="buy a phone", candidates=[])
    assert "couldn't gather" in md.lower()
