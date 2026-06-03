"""Deterministic tests for the LLM extraction fallback's JSON RETRY (agent/extract.py, fix 2).

A single malformed Scout reply must NOT silently become zero candidates (the old
`except: return []`). The retry feeds the error back and tries again, capped — so one wobble is
recovered. Uses a fake Groq that scripts replies in order.
"""

import copy
from types import SimpleNamespace as NS

from browser_agent.agent.extract import extract_candidates, relevance_filter, select_candidates


def content_resp(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))])


class FakeGroq:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

        async def create(**kwargs):
            self.calls.append(copy.deepcopy({k: v for k, v in kwargs.items()}))
            return self.scripted.pop(0)

        self.chat = NS(completions=NS(create=create))


# ----------------------------------------------------------------- extract_candidates retry
async def test_extract_retries_then_succeeds():
    # First reply is unparseable; the retry returns valid JSON -> candidates, NOT [].
    groq = FakeGroq([content_resp("sorry, here are the results:"),       # no JSON -> parse fails
                     content_resp('{"candidates": [{"name": "Think and Grow Rich", "price": "₹139"}]}')])
    out = await extract_candidates(groq, "snapshot text", {"sources": ["amazon"]})
    assert len(out) == 1 and out[0]["name"] == "Think and Grow Rich"
    assert len(groq.calls) == 2                                          # retried, didn't return [] on #1


async def test_extract_returns_empty_only_after_retries_exhausted():
    groq = FakeGroq([content_resp("nope"), content_resp("still nope"), content_resp("nada")])
    out = await extract_candidates(groq, "snapshot", {})
    assert out == [] and len(groq.calls) == 3                            # 3 attempts, then [] (not on #1)


async def test_extract_bare_list_reply():
    # The model may return a bare JSON array instead of {"candidates": [...]}.
    groq = FakeGroq([content_resp('[{"name": "A"}, {"name": "B"}]')])
    out = await extract_candidates(groq, "snapshot", {})
    assert [c["name"] for c in out] == ["A", "B"] and len(groq.calls) == 1


# ----------------------------------------------------------------- select_candidates retry
async def test_select_retries_then_succeeds():
    cands = [{"name": "A", "rating": 4.1}, {"name": "B", "rating": 4.8}, {"name": "C", "rating": 4.5}]
    groq = FakeGroq([content_resp("the best is index..."),               # unparseable
                     content_resp('{"top": [1, 2], "reason": "highest rated"}')])
    sel = await select_candidates(groq, "best book", cands)
    assert sel["selected"]["name"] == "B" and [c["name"] for c in sel["top"]] == ["B", "C"]
    assert len(groq.calls) == 2


async def test_select_falls_back_to_scorer_after_retries():
    # Every reply is unparseable -> deterministic rating-dominant scorer picks the top-rated.
    cands = [{"name": "A", "rating": 4.1, "review_count": 10},
             {"name": "B", "rating": 4.9, "review_count": 9000}]
    groq = FakeGroq([content_resp("x"), content_resp("y"), content_resp("z")])
    sel = await select_candidates(groq, "best", cands)
    assert sel["selected"]["name"] == "B"                                # scorer fallback still picks
    assert len(groq.calls) == 3


# ----------------------------------------------------------------- fix 6: relevance filter
def test_relevance_filter_drops_language_bundle_and_noprice():
    from browser_agent.agent.extract import relevance_filter
    cands = [
        {"name": "Think and Grow Rich (English)", "price": "₹139", "rating": 4.5},
        {"name": "Quem pensa enriquece", "price": "₹200", "rating": 4.9},                  # Portuguese
        {"name": "How to Win Friends + Think & Grow Rich (Set of 2 Books)", "price": "₹250"},  # bundle
        {"name": "Think and Grow Rich Hardcover", "price": None, "rating": 4.4},            # no price
    ]
    out = relevance_filter("buy Think and Grow Rich", cands)
    names = [c["name"] for c in out]
    assert any("English" in n for n in names)
    assert not any("Quem pensa" in n for n in names)      # wrong language (no title overlap) dropped
    assert not any("Set of 2" in n for n in names)        # bundle dropped (single title asked)
    assert all(c.get("price") for c in out)               # no-price dropped since priced options exist


def test_relevance_filter_keeps_all_when_nothing_matches():
    cands = [{"name": "Random Widget", "price": "₹10"}, {"name": "Another Thing", "price": "₹20"}]
    out = relevance_filter("buy Think and Grow Rich", cands)
    assert out == cands                                   # never zero out the pool


async def test_select_candidates_filters_before_ranking():
    from browser_agent.agent.extract import select_candidates
    cands = [
        {"name": "Quem pensa enriquece", "rating": 4.9, "review_count": 99999},   # most reviews, wrong lang
        {"name": "Think and Grow Rich", "price": "₹139", "rating": 4.5, "review_count": 6060},
    ]
    # LLM selector fails on every attempt -> deterministic scorer over the FILTERED pool, which has
    # dropped the Portuguese edition despite its higher review count.
    groq = FakeGroq([content_resp("not json"), content_resp("nope"), content_resp("nada")])
    sel = await select_candidates(groq, "buy Think and Grow Rich", cands)
    assert sel["selected"]["name"] == "Think and Grow Rich"
