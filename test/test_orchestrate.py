"""orchestrate() flow: clarification cap and salvage-extraction on a non-completing explore,
plus the role->model wiring (planner=qwen, composer=gpt-oss)."""

from types import SimpleNamespace as NS

import browser_agent.agent.orchestrate as orch
import browser_agent.config as cfg
from browser_agent.agent.extract import synthesize_options
from browser_agent.agent.planner import plan_subgoals


class RecordingGroq:
    """Captures the `model` of the last create() call and returns scripted content."""
    def __init__(self, content):
        self.model = None

        async def create(**kw):
            self.model = kw.get("model")
            return NS(choices=[NS(message=NS(content=content))])

        self.chat = NS(completions=NS(create=create))


def _explore():
    return {"id": 1, "type": "explore", "goal": "find phones", "success_condition": "5+",
            "tier": "auto", "needs_approval": False, "explore_spec": {"sources": ["amazon"]}}


def _present():
    return {"id": 2, "type": "present", "goal": "show", "success_condition": "shown",
            "tier": "auto", "needs_approval": False}


def _ask():
    return {"id": 1, "type": "ask_user", "goal": "clarify", "question": "Budget?",
            "success_condition": "", "tier": "auto", "needs_approval": False,
            "allow_free_text": True, "multi_select": False}


async def _orchestrate(**over):
    # ground=False: grounding needs a live browser; these tests run with session=None.
    kw = dict(allowlist=set(), approve=lambda p: True, ask=lambda *a, **k: "",
              max_steps=5, max_replans=3, read_content=False, ground=False)
    kw.update(over)
    return await orch.orchestrate(None, None, [], [], "buy a phone", **kw)


async def test_orchestrate_salvages_candidates_when_explore_escalates(monkeypatch):
    seen = {}

    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("escalate", "loop detected")          # reasoner never called subgoal_complete

    async def fake_observe(session):
        return ("a page full of phone listings", "http://amazon.in/s")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return [{"name": "Phone A", "source": "amazon", "url": "/dp/B0PHONE"}]  # relative url

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        seen["cands"] = cands
        return "## Shortlist\n- Phone A"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    async def _no_dom(session, spec=None, **k):
        return []
    monkeypatch.setattr(orch, "extract_with_scroll", _no_dom)
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate()
    # The explore escalated, but candidates on the page were salvaged -> present ran -> we get a
    # shortlist instead of "Failed: exhausted re-plan budget".
    assert result == "## Shortlist\n- Phone A"
    # the relative product url was resolved to absolute against the page (fix 1)
    assert seen["cands"] == [{"name": "Phone A", "source": "amazon", "url": "http://amazon.in/dp/B0PHONE"}]


async def test_orchestrate_salvages_via_content_extractor(monkeypatch):
    # Heavy retail SPA: the a11y snapshot yields 0 candidates, but the content extractor
    # (readable_text) recovers the product text and a re-extract finds them. The run should proceed
    # to present instead of burning the re-plan budget (the "i want to buy a phone" failure).
    seen = {}
    calls = {"extract": 0}

    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")

    async def fake_observe(session):
        return ("nav chrome only, grid not in a11y tree", "http://flipkart.com/mobiles/pr")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        calls["extract"] += 1
        # first call (raw snapshot) -> nothing; second call (readable text) -> candidates
        return [] if calls["extract"] == 1 else [
            {"name": "Galaxy M14", "source": "flipkart", "url": "https://www.flipkart.com/m14/p/itm1"}]

    async def fake_readable(session):
        return "# Mobiles\nGalaxy M14 5G Rs 12,999\nRedmi 13C Rs 9,499"

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        seen["cands"] = cands
        return "## Shortlist\n- Galaxy M14"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    async def _no_dom(session, spec=None, **k):
        return []
    monkeypatch.setattr(orch, "extract_with_scroll", _no_dom)
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)
    monkeypatch.setattr(orch, "readable_text", fake_readable)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate(read_content=True)
    assert result == "## Shortlist\n- Galaxy M14"
    assert seen["cands"] == [{"name": "Galaxy M14", "source": "flipkart",
                              "url": "https://www.flipkart.com/m14/p/itm1"}]
    assert calls["extract"] == 2          # raw snapshot, then content-extractor re-pass


async def test_orchestrate_does_not_present_empty_when_explore_finds_nothing(monkeypatch):
    plans = {"n": 0}

    async def fake_plan(*a, **k):
        plans["n"] += 1
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")          # reasoner CLAIMS done...

    async def fake_observe(session):
        return ("a page with no listings", "http://x")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return []                          # ...but nothing extractable -> not really done

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    async def _no_dom(session, spec=None, **k):
        return []
    monkeypatch.setattr(orch, "extract_with_scroll", _no_dom)
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)

    result = await _orchestrate(max_replans=1)
    # It must re-plan (not march on to present an empty shortlist), then fail honestly.
    assert plans["n"] >= 2 and result.startswith("Failed")


def test_page_looks_blocked_detects_walls_and_passes_real_pages():
    from browser_agent.utils.text import page_looks_blocked
    assert page_looks_blocked("Enter the characters you see below")
    assert page_looks_blocked("Our systems have detected unusual traffic from your network")
    assert page_looks_blocked("Please complete the CAPTCHA to continue")
    # Real listings and an empty string are NOT walls.
    assert page_looks_blocked("Galaxy M33 5G  4.2 stars  Rs 13999  Add to cart") is None
    assert page_looks_blocked("") is None


async def test_orchestrate_steers_replan_away_from_blocked_source(monkeypatch):
    captured = {}

    async def fake_plan(*a, **k):
        if k.get("failed_id") is not None:        # the re-plan call
            captured["reason"] = k.get("reason")
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("escalate", "loop detected")

    async def fake_observe(session):
        return ("Enter the characters you see below — we just need to make sure you're not a robot",
                "https://www.amazon.in/s?k=phones")

    async def fake_extract(groq, snapshot, spec, *, model=None):
        return []                                  # blocked page -> nothing extractable

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "observe", fake_observe)
    monkeypatch.setattr(orch, "full_snapshot", fake_observe)  # explore extraction reads the full snapshot
    async def _no_dom(session, spec=None, **k):
        return []
    monkeypatch.setattr(orch, "extract_with_scroll", _no_dom)
    monkeypatch.setattr(orch, "extract_candidates", fake_extract)

    await _orchestrate(max_replans=1)
    # The re-plan reason must name the dead source, flag it as a wall, and steer elsewhere.
    reason = captured["reason"]
    assert "amazon.in" in reason
    assert "DIFFERENT" in reason
    assert "bot-check" in reason
    assert "avoid these" in reason.lower()


async def test_orchestrate_caps_clarifications(monkeypatch):
    calls = {"ask": 0}

    async def fake_plan(*a, **k):
        return [_ask()]                                # planner keeps re-asking a clarification

    async def fake_run(*a, **k):
        return ("complete", "ok")

    def fake_ask(q, **k):
        calls["ask"] += 1
        return "Budget"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)

    result = await _orchestrate(ask=fake_ask)
    # Without the cap this loops forever (every re-plan re-asks). With it, we ask at most the cap
    # then skip and finish.
    assert calls["ask"] == orch.MAX_CLARIFICATIONS
    assert result == "Done."


async def test_planner_uses_planner_model():
    g = RecordingGroq('{"subgoals": [{"id": 1, "goal": "g", "success_condition": "s"}]}')
    await plan_subgoals(g, "buy a phone", ["browser_navigate"])
    assert g.model == cfg.PLANNER_MODEL


async def test_synthesize_uses_composition_model():
    g = RecordingGroq("## Shortlist")
    await synthesize_options(g, "buy a phone", [{"name": "P", "source": "x"}])
    assert g.model == cfg.COMPOSITION_MODEL


# ----------------------------------------------------------------- LLM-based selection (exploit)
async def test_select_candidates_uses_llm_pick():
    from browser_agent.agent.extract import select_candidates
    g = RecordingGroq('{"top": [2, 0], "reason": "best rated"}')
    cands = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    out = await select_candidates(g, "buy a book", cands)
    assert out["selected"] == {"name": "C"}                       # index 2, best-first
    assert [c["name"] for c in out["top"]] == ["C", "A"]
    assert out["reason"] == "best rated"


async def test_select_candidates_falls_back_to_scorer_on_bad_json():
    from browser_agent.agent.extract import select_candidates
    g = RecordingGroq("not json at all")                          # LLM reply unparseable
    cands = [{"name": "A", "rating": 4.0}, {"name": "B", "rating": 4.9}]
    out = await select_candidates(g, "buy a book", cands)
    assert out["selected"]["name"] == "B"                         # rating-dominant fallback picks B


async def test_select_candidates_empty():
    from browser_agent.agent.extract import select_candidates
    out = await select_candidates(None, "x", [])
    assert out["selected"] is None and out["top"] == []


# ----------------------------------------------------------------- parallel fan-out
def test_expand_explore_sources_fans_out_per_source():
    sg = {"id": 2, "type": "explore", "goal": "gather books", "success_condition": "5+",
          "explore_spec": {"sources": ["goodreads", "bookbub", "lithub"], "target_count": 5}}
    out = orch._expand_explore_sources(sg)
    assert [s["explore_spec"]["sources"] for s in out] == [["goodreads"], ["bookbub"], ["lithub"]]
    assert [s["id"] for s in out] == ["2.0", "2.1", "2.2"]
    assert "goodreads" in out[0]["goal"]
    # <2 sources -> unchanged
    assert orch._expand_explore_sources({"id": 1, "explore_spec": {"sources": ["x"]}})[0]["id"] == 1
    assert len(orch._expand_explore_sources({"id": 1, "explore_spec": {}})) == 1


async def test_orchestrate_fans_single_multisource_explore_into_parallel(monkeypatch):
    captured = {}

    async def fake_plan(*a, **k):
        return [{"id": 1, "type": "explore", "goal": "gather", "success_condition": "5+",
                 "tier": "auto", "needs_approval": False,
                 "explore_spec": {"sources": ["goodreads", "bookbub"]}},
                _present()]

    async def fake_gather(groq, tools, subgoals, **k):
        captured["subgoals"] = subgoals
        return {s["id"]: [{"name": "B", "source": s["explore_spec"]["sources"][0]}] for s in subgoals}

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        captured["cands"] = cands
        return "## Shortlist"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "gather_parallel", fake_gather)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate(parallel=True)
    # The single 2-source explore was fanned out into 2 parallel workers, both aggregated.
    assert len(captured["subgoals"]) == 2
    assert len(captured["cands"]) == 2
    assert result == "## Shortlist"


async def test_orchestrate_uses_dom_extractor_first(monkeypatch):
    # Fix 1: the explore salvage reads candidates from the live DOM FIRST (deterministic). When that
    # yields rows, the LLM a11y extractor must NOT be called at all.
    seen = {"llm_extract": 0, "cands": None}

    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")

    async def fake_dom_scroll(session, spec=None, **k):
        return [{"name": "Think and Grow Rich", "price": "₹139", "rating": 4.5, "source": "amazon.in",
                 "url": "https://www.amazon.in/dp/9389931525"}]

    async def fake_llm_extract(groq, snapshot, spec, *, model=None):
        seen["llm_extract"] += 1
        return [{"name": "SHOULD_NOT_BE_USED"}]

    async def fake_synth(groq, goal, cands, selected=None, *, model=None):
        seen["cands"] = cands
        return "## Shortlist\n- Think and Grow Rich"

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "extract_with_scroll", fake_dom_scroll)
    monkeypatch.setattr(orch, "extract_candidates", fake_llm_extract)
    monkeypatch.setattr(orch, "synthesize_options", fake_synth)

    result = await _orchestrate()
    assert result.startswith("## Shortlist")
    assert seen["cands"] == [{"name": "Think and Grow Rich", "price": "₹139", "rating": 4.5,
                              "source": "amazon.in", "url": "https://www.amazon.in/dp/9389931525"}]
    assert seen["llm_extract"] == 0          # DOM-first: the LLM a11y extractor was never invoked


# ----------------------------------------------------------------- fix 3: cart-confirmation gate
def _checkout_sg():
    return {"id": 1, "type": "act", "goal": "Proceed to checkout and pay",
            "success_condition": "order placed", "tier": "auto", "needs_approval": False}


async def test_checkout_gated_when_cart_empty(monkeypatch):
    # A checkout subgoal must NOT run with an empty/unconfirmed cart — it escalates -> re-plan.
    plans = {"n": 0}
    ran = {"run": 0}

    async def fake_plan(*a, **k):
        plans["n"] += 1
        return [_checkout_sg()]

    async def fake_run(*a, **k):
        ran["run"] += 1
        return ("complete", "ok")

    async def fake_cart(session):
        return 0                                  # confirmed empty

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "cart_count", fake_cart)
    result = await _orchestrate(max_replans=1)
    assert ran["run"] == 0                         # checkout never executed on an empty cart
    assert plans["n"] >= 2 and result.startswith("Failed")   # gated -> re-plan -> exhausted


async def test_checkout_proceeds_when_cart_confirmed(monkeypatch):
    ran = {"run": 0}

    async def fake_plan(*a, **k):
        return [_checkout_sg()]

    async def fake_run(*a, **k):
        ran["run"] += 1
        return ("complete", "ok")

    async def fake_cart(session):
        return 2                                   # item(s) in the cart

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "cart_count", fake_cart)
    result = await _orchestrate()
    assert ran["run"] == 1 and result == "Done."   # cart confirmed -> checkout runs


# ----------------------------------------------------------------- fix 1/2: url finalize + selected note
def test_finalize_candidates_resolves_and_flags():
    cands = [{"name": "abs", "url": "https://www.amazon.in/dp/9389931525"},
             {"name": "rel", "url": "/dp/B0X"},
             {"name": "none", "url": None},
             {"name": "bad", "url": "https://gp/x"}]
    out = orch._finalize_candidates(cands, "https://www.amazon.in/s?k=x")
    assert out[0]["url"] == "https://www.amazon.in/dp/9389931525"   # absolute passes through
    assert out[1]["url"] == "https://www.amazon.in/dp/B0X"          # relative resolved to origin
    assert out[2]["url"] is None                                    # missing -> flagged None
    assert out[3]["url"] is None                                    # malformed (dotless host) rejected


def test_size_hint_parsing():
    assert orch._size_hint("Add the Asics shoe to the cart, size 9") == "9"
    assert orch._size_hint("buy uk 8 trainers") == "8"
    assert orch._size_hint("get me a medium tee, size: M") == "M"
    assert orch._size_hint("eu 42 boots") == "42"
    assert orch._size_hint("Open https://shop/dp/B09ABC123 and add to cart") is None   # ASIN digits != size


def test_drop_junk_candidates_removes_search_engine_rows():
    cands = [
        {"name": "Real shoe", "url": "https://www.superkicks.in/products/asics-gel", "source": "superkicks.in"},
        {"name": "Up To ₹6000 - Asics ...Amazon.in", "url": "https://www.google.com/search?q=asics",
         "source": "google.com"},                                          # SERP result block
        {"name": "store rating", "price": "₹7,999", "source": "google.com"},  # SERP snippet, no real url
        {"name": "No-url retailer row", "url": None, "source": "myntra.com"},  # url-less but real source
    ]
    out = orch._drop_junk_candidates(cands)
    names = [c["name"] for c in out]
    assert "Real shoe" in names
    assert "No-url retailer row" in names                  # real retailer kept even without a url
    assert not any("google" in (c.get("source") or "") for c in out)   # both google rows dropped
    assert len(out) == 2


async def test_orchestrate_skips_extraction_on_search_engine_page(monkeypatch):
    # The reasoner ended on a Google results page. We must NOT scrape it as products — with no other
    # candidates the explore re-plans onto a real source (instead of presenting SERP junk).
    async def fake_plan(*a, **k):
        return [_explore(), _present()]

    async def fake_run(*a, **k):
        return ("complete", "ok")

    async def fake_dom(session, spec=None, **k):
        raise AssertionError("must not extract from a search-engine page")

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "current_url", lambda s: _async("https://www.google.com/search?q=asics"))
    monkeypatch.setattr(orch, "extract_with_scroll", fake_dom)   # asserts it is never reached
    monkeypatch.setattr(orch, "extract_candidates", fake_dom)

    # Only one source -> when it yields nothing and there are no candidates, the re-plan budget is
    # spent and we get a Failed result naming the dead source. The key assertion is no scrape happened.
    result = await _orchestrate(max_replans=0)
    assert result.startswith("Failed:")
    assert "google.com" in result


def _async(value):
    async def _coro():
        return value
    return _coro()


def test_selected_note_carries_url_and_title():
    note = orch._selected_note({"name": "Think and Grow Rich",
                                "url": "https://www.amazon.in/dp/9389931525", "price": "₹139"})
    assert "Selected product" in note and "https://www.amazon.in/dp/9389931525" in note
    assert "Think and Grow Rich" in note
    assert orch._selected_note({"name": "no url"}) == "" and orch._selected_note(None) == ""


async def test_selected_url_threaded_into_act_subgoal(monkeypatch):
    captured = {}

    async def fake_plan(*a, **k):
        return [{"id": 1, "type": "exploit", "goal": "rank", "success_condition": "",
                 "tier": "auto", "needs_approval": False},
                {"id": 2, "type": "act", "goal": "Open the selected book and add to cart",
                 "success_condition": "in cart", "tier": "auto", "needs_approval": False}]

    async def fake_select(groq, goal, cands, **k):
        return {"selected": {"name": "Think and Grow Rich",
                             "url": "https://www.amazon.in/dp/9389931525"}, "top": [], "reason": "r"}

    async def fake_run(groq, session, tools, sg, **k):
        if sg.get("type") == "act":
            captured["plan_context"] = k.get("plan_context", "")
            captured["selected"] = k.get("selected")
        return ("complete", "ok")

    async def fake_cart(session):
        return 1                                          # add-to-cart subgoal isn't checkout-gated, but be safe

    monkeypatch.setattr(orch, "plan_subgoals", fake_plan)
    monkeypatch.setattr(orch, "select_candidates", fake_select)
    monkeypatch.setattr(orch, "run_subgoal", fake_run)
    monkeypatch.setattr(orch, "cart_count", fake_cart)
    await _orchestrate()
    assert "https://www.amazon.in/dp/9389931525" in captured["plan_context"]   # fed the exact URL
    assert captured["selected"]["url"].endswith("9389931525")                  # and the selected dict
