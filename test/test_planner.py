"""Deterministic regression tests for the PLANNER role (agent/planner.py + ENHANCED_PLANNER).

Covers JSON parsing/normalization + retry, the re-plan prompt shape, and the B-series source
behaviour the enhancements target: web-grounding reaches the prompt, and the prompt instructs the
planner to match source category to the goal (the "jeans -> electronics store" wild-planning bug).
Reasoner-side behaviour lives in test_reasoner.py. (`_normalize` is covered in test_enhanced.py.)
"""

import copy
from types import SimpleNamespace as NS

from browser_agent.agent.planner import plan_subgoals
from browser_agent.agent.prompts import ENHANCED_PLANNER
from browser_agent.utils.text import extract_json


# ----------------------------------------------------------------- fakes
def content_resp(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))])


class FakeGroq:
    """Records every create() call (sans tools) so tests can inspect the planner's prompt."""
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

        async def create(**kwargs):
            self.calls.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != "tools"}))
            return self.scripted.pop(0)

        self.chat = NS(completions=NS(create=create))


def _user_msg(groq, call=0) -> str:
    return groq.calls[call]["messages"][1]["content"]


# ----------------------------------------------------------------- JSON parsing / normalization
def test_extract_json_tolerates_fences_think_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<think>reasoning…</think>\n{"a": 1}') == {"a": 1}
    assert extract_json('here you go: {"a": 1} done') == {"a": 1}


async def test_plan_subgoals_parses_and_normalizes():
    groq = FakeGroq([content_resp(
        '```json\n{"subgoals": ['
        '{"id": 1, "goal": "open page", "success_condition": "loaded", "needs_approval": false},'
        '{"goal": "place order", "needs_approval": true}]}\n```')])
    plan = await plan_subgoals(groq, "buy a thing", ["browser_navigate"])
    assert len(plan) == 2
    assert plan[0]["goal"] == "open page" and plan[0]["needs_approval"] is False
    assert plan[1]["id"] == 2 and plan[1]["needs_approval"] is True
    assert plan[1]["success_condition"] == ""           # default filled in


async def test_plan_subgoals_retries_on_truncated_json():
    # A thinking model can return truncated JSON; the planner retries once before giving up.
    groq = FakeGroq([
        content_resp('{"subgoals": [{"id": 1, "goal": "a"'),   # truncated -> parse error
        content_resp('{"subgoals": [{"id": 1, "goal": "a", "success_condition": "s", "needs_approval": false}]}'),
    ])
    plan = await plan_subgoals(groq, "g", ["browser_navigate"])
    assert len(plan) == 1 and plan[0]["goal"] == "a"
    assert len(groq.calls) == 2                          # retried exactly once


async def test_plan_subgoals_raises_after_retry_fails():
    # The planner gives a thinking model 3 chances at clean JSON before giving up.
    groq = FakeGroq([content_resp("not json"), content_resp("still not json"), content_resp("nope")])
    raised = False
    try:
        await plan_subgoals(groq, "g", ["browser_navigate"])
    except ValueError:
        raised = True
    assert raised and len(groq.calls) == 3


# ----------------------------------------------------------------- prompt content / grounding
def test_enhanced_planner_prompt_has_grounding_and_category_rules():
    p = ENHANCED_PLANNER.lower()
    assert "web grounding" in p                          # uses the pre-planning search block
    assert "category" in p                               # match source category to the goal
    assert "electronics" in p                            # the explicit "not electronics for clothing" guard
    # source discovery is the agent's job, never a user-facing question
    assert "which website" in p and "your job" in p


async def test_grounding_block_reaches_planner_prompt():
    # The orchestrator prepends the web-grounding block to `preamble`; planner puts the preamble at
    # the head of the user turn. Assert the grounded domains actually reach the model.
    groq = FakeGroq([content_resp('{"subgoals": [{"id": 1, "goal": "g", "success_condition": "s"}]}')])
    grounding = ('### Web grounding (live search for \'buy levis 501 jeans\')\n'
                 '1. Levi\'s 501 Original - Myntra — myntra.com\n'
                 '2. Levi\'s 501 - Amazon Fashion — amazon.in')
    await plan_subgoals(groq, "buy levis 501 jeans", ["browser_navigate"], preamble=grounding)
    user = _user_msg(groq)
    assert "Web grounding" in user and "myntra.com" in user
    assert "User goal: buy levis 501 jeans" in user


async def test_replan_prompt_carries_failure_context():
    groq = FakeGroq([content_resp('{"subgoals": [{"id": 1, "goal": "g", "success_condition": "s"}]}')])
    await plan_subgoals(
        groq, "buy a phone", ["browser_navigate"],
        prior_plan=[{"id": 1, "goal": "search amazon"}], failed_id=1,
        reason="source 'amazon.in' yielded no candidates (bot-check). Already tried (avoid these): amazon.in",
        observations="blocked at amazon.in")
    user = _user_msg(groq)
    assert "failing at subgoal 1" in user
    assert "amazon.in" in user and "avoid these" in user
    assert "blocked at amazon.in" in user
