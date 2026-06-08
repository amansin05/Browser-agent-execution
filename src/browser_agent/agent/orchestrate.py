"""The two-tier orchestration: per-subgoal reasoner loop + the plan -> verify -> re-plan flow.

Both `run_subgoal` and `orchestrate` operate on an already-open MCP session (the browser is
opened/closed by the caller — AgentSession), so a session can run many tasks against the same
tabs.
"""

import asyncio
import json
import re
from collections import deque

from browser_agent.agent.dom_extract import (
    at_checkout, cart_confirmed, cart_count, click_add_to_cart, current_url, extract_candidates_dom,
    extract_with_scroll, proceed_to_checkout, select_cod, select_required_variant,
)
from browser_agent.agent.dom_index import execute_action, index_dom, is_terminating
from browser_agent.agent.extract import extract_candidates, select_candidates, synthesize_options
from browser_agent.agent.gather import gather_parallel
from browser_agent.agent.grounding import web_grounding
from browser_agent.agent.planner import plan_subgoals
from browser_agent.agent.reader import readable_text
from browser_agent.agent.reasoner import reasoner_decide
from browser_agent.agent.verifier import verify_candidates, verify_product_match, verify_success
from browser_agent.config import COMPOSITION_MODEL, MODEL, PLANNER_MODEL
from browser_agent.log import get_logger, trace_enabled
from browser_agent.services.mcp_client import (
    full_snapshot, list_tabs_state, newest_new_tab, observe, open_session, resolve_url, select_tab,
)
from browser_agent.utils.domains import domain_allowed, domain_of, is_search_engine
from browser_agent.utils.io import maybe_await
from browser_agent.utils.text import page_looks_blocked, page_looks_like_login, snapshot_is_sufficient

log = get_logger(__name__)


def _log_plan(label: str, plan: list[dict]) -> None:
    log.info("[%s] %s", label, " | ".join(f"{sg['id']}. {sg['goal']}" for sg in plan))

# A confirm/secured subgoal that gets no human answer within this window is ABORTED (never
# auto-approved) — the safe default from B5.
APPROVAL_TIMEOUT = 300.0

# Sentinel an ask_user answer carries when the user clicked "No preference — explore all".
# The UI (AskModal) submits this exact string; we then steer the re-plan to broaden, not narrow.
SKIP_SENTINEL = "__no_preference__"

# Stop the planner from pestering the user with one clarification after another (each forces a full
# re-plan). After this many answered clarifications, further ask_user subgoals are skipped and we
# explore broadly instead.
MAX_CLARIFICATIONS = 2

# Control-flow philosophy: the LLM makes the decisions (when stuck, when done, which way to go). The
# guards below DON'T force a give-up on first sight any more — they INFORM the reasoner (a nudge it
# reads next turn) and let it decide. Code only steps in as a HIGH anti-runaway BACKSTOP, so a truly
# stuck agent can't spin to the step budget. (max_steps + approval gates remain the only hard rails.)

# Verifier keeps refusing a subgoal_complete: each refusal is fed back to the reasoner to retry
# differently; only after THIS many refusals do we hand back to the planner (high backstop).
MAX_COMPLETE_REJECTS = 6

# Anti-runaway backstop: the reasoner choosing the EXACT same action this many times in a row is a
# genuine infinite loop — force an escalate. (A developing loop just nudges; see below.)
LOOP_HARD_STOP = 6

# Below this many interactive elements, the indexed-DOM view alone is too thin to reason over (a
# content blob, an article, a canvas) — append the cleaned page text (Readability -> trafilatura ->
# innerText) so the reasoner has something to read, the same salvage the a11y path used.
MIN_INTERACTIVE_FOR_TEXT = 3

# The reasoner may emit several actions in one turn; execute at most this many (the page-change
# guard usually stops a batch sooner). Mirrors browser-use's max_actions_per_step.
MAX_ACTIONS_PER_STEP = 5

# Page-stagnation signal: if the page fingerprint is unchanged this many steps running DESPITE the
# agent acting, its actions are having no effect — we NUDGE the reasoner (inform-only; it decides
# whether to change tactics or escalate). Catches DIFFERENT actions that still change nothing, which
# the exact-loop backstop wouldn't.
STAGNATION_LIMIT = 3

# Once the subgoal has burned this fraction of its step budget, warn the reasoner to land it or
# escalate rather than fritter the tail away (browser-use's 75% budget nudge).
BUDGET_WARN_FRACTION = 0.75

# Ping-pong guard (fix 4a/5): a `navigate` to a URL the agent is currently on or has been on within
# the last few steps is the click->product->navigate-back-to-search bounce. We BLOCK that redundant
# navigation and nudge; after this many such revisits in a subgoal we escalate (anti-runaway).
URL_REVISIT_WINDOW = 6
MAX_REVISIT_NUDGES = 3

# Fix 4b: on an EXPLORE subgoal, once the live DOM yields at least this many candidate rows the
# listing has loaded — we drop `navigate`/`click_element` from that turn's tools so the reasoner
# completes instead of clicking into a product (which starts the click<->navigate ping-pong). The
# candidates are read off the page automatically on completion.
EXPLORE_LISTING_MIN = 5


# A subgoal that moves to checkout / payment (fix 3). We gate these on a non-empty cart so the agent
# never navigates to /checkout having never confirmed the item was added (the blind-checkout bug).
_CHECKOUT_PHRASES = ("checkout", "check out", "proceed to pay", "proceed to buy", "place order",
                     "place the order", "make payment", "payment", "pay now")


def _is_checkout(sg: dict) -> bool:
    if sg.get("type", "act") != "act":
        return False
    goal = (sg.get("goal") or "").lower()
    return any(p in goal for p in _CHECKOUT_PHRASES)


_ADD_TO_CART_PHRASES = ("add to cart", "add to bag", "add to basket", "add it to the cart",
                        "add the", "put in the cart")


def _is_add_to_cart(sg: dict) -> bool:
    if sg.get("type", "act") != "act":
        return False
    goal = (sg.get("goal") or "").lower()
    return ("cart" in goal or "bag" in goal or "basket" in goal) and not _is_checkout(sg)


def _is_cart_stage(sg: dict) -> bool:
    """A subgoal at the CART/CHECKOUT stage — the item is expected to be in the cart ALREADY, so the
    agent must NOT navigate back to the product page (the post-add product<->cart ping-pong). True for
    checkout subgoals and for cart/bag subgoals that aren't an 'add' (e.g. 'go to cart', 'checkout')."""
    if sg.get("type", "act") != "act":
        return False
    goal = (sg.get("goal") or "").lower()
    if _is_checkout(sg):
        return True
    return ("cart" in goal or "bag" in goal or "basket" in goal) and "add" not in goal


def _selected_note(selected, sg=None) -> str:
    """A `### Selected product` block telling the reasoner the EXACT product it already chose, with
    its absolute URL — so a post-selection act subgoal navigates straight to it (fix 2) instead of
    re-searching and clicking a stray result by an unstable index (the OnePlus-page bug). At the
    CART/CHECKOUT stage the item is already in the cart, so the note instead FORBIDS navigating back
    to the product page (the post-add product<->cart ping-pong)."""
    if not (isinstance(selected, dict) and selected.get("url")):
        return ""
    title = selected.get("name") or selected.get("title") or "the selected item"
    price = f" (price {selected['price']})" if selected.get("price") else ""
    if isinstance(sg, dict) and _is_cart_stage(sg):
        return ("\n### Selected product\n"
                f"The selected item \"{title}\"{price} should ALREADY be in the cart. Do NOT navigate "
                f"back to its product page ({selected['url']}) — work with the cart / checkout on the "
                f"current page; if the cart looks empty, escalate rather than re-opening the product.")
    return ("\n### Selected product\n"
            f"You have ALREADY chosen \"{title}\"{price}. Navigate to its URL ONCE to open it: "
            f"{selected['url']} — do NOT search again or pick a different item. Do what the subgoal "
            f"asks (e.g. click 'Add to Cart') on THAT page; the on-page 'Added to Cart' confirmation "
            f"or the cart badge is success — then call subgoal_complete. Do NOT then re-navigate to "
            f"the product or open the cart page.")


# A size stated in the goal/clarifications — "size 9", "uk 8", "us 10", "eu 42", "size: M". Requires a
# size keyword so a stray digit (an ASIN, a price) isn't mistaken for a size. Used to pick the right
# variant before add-to-cart; falls back to the first in-stock size when absent.
_SIZE_RE = re.compile(
    r"\b(?:size|uk|us|eu|eur)\s*[:=]?\s*((?:xxs|xs|s|m|l|xl|xxl|xxxl)|\d{1,2}(?:\.\d)?)\b", re.I)


def _size_hint(text: str) -> str | None:
    m = _SIZE_RE.search(text or "")
    return m.group(1) if m else None


def _norm_url(u: str) -> str:
    """Normalize a URL for revisit/loop comparison: drop the query + fragment, lowercase, no trailing
    slash. So `/s?k=x&qid=1` and `/s?k=x&qid=2` (the index-varying search bounce) compare equal."""
    if not u:
        return ""
    return u.split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()


def _finalize_candidates(cands: list, base: str = "") -> list:
    """Resolve each candidate's `url` to an absolute http(s) URL (reusing the navigate resolver),
    flagging an unresolvable one as None. DOM rows already carry an absolute a.href (passes through);
    LLM/a11y rows may be relative -> resolved against the page. The buy flow navigates by this url."""
    for c in cands:
        if isinstance(c, dict):
            c["url"] = resolve_url(c.get("url") or "", base)
    return cands


def _drop_junk_candidates(cands: list) -> list:
    """Drop candidates that came off a SEARCH ENGINE — a SERP result block scrapes into a snippet
    "candidate" (junk title, a number lifted from the snippet, a google.com/bing link) that is never
    a buyable product (the SERP-scrape bug). Keyed on the per-row url/source; the page-level guard in
    the explore salvage handles a11y rows that carry neither. Real-retailer rows (incl. url-less ones,
    kept for research) pass through unchanged."""
    out = [c for c in cands if isinstance(c, dict)
           and not is_search_engine(c.get("url") or "") and not is_search_engine(c.get("source") or "")]
    if len(out) != len(cands):
        log.info("dropped %d search-engine (SERP) candidate(s) — not buyable products",
                 len(cands) - len(out))
    return out


def _expand_explore_sources(sg: dict) -> list[dict]:
    """Split one explore subgoal into one-subgoal-per-source so they can run in parallel. A planner
    often emits a SINGLE 'gather from these sites' explore with several sources in its explore_spec;
    fanning it out is what lets the parallel gather actually engage (otherwise a lone explore flails
    across the sites sequentially in one tab). Returns [sg] unchanged when there are <2 sources."""
    spec = sg.get("explore_spec") or {}
    sources = [s for s in (spec.get("sources") or []) if s]
    if len(sources) < 2:
        return [sg]
    out = []
    for k, src in enumerate(sources):
        out.append({**sg,
                    "id": f"{sg.get('id', 'e')}.{k}",
                    "goal": f"{sg.get('goal', 'Gather candidates')} — from {src}",
                    "explore_spec": {**spec, "sources": [src]}})
    return out


def _page_fingerprint(dom, url: str, snapshot_text: str):
    """A cheap identity of the current page used to detect stagnation. From the indexed view when we
    have one (url + scroll position + the set of interactive elements), else the a11y snapshot."""
    if dom is not None and dom.elements:
        return ("dom", dom.url, dom.scroll_y, tuple(sorted(str(e.key()) for e in dom.elements)))
    return ("a11y", url, hash(snapshot_text))


async def _verify_complete(groq, session, subgoal, snapshot_text, model, *, selected=None,
                           url="", page_title=""):
    """Decide whether a subgoal_complete is genuine:
      - GATHER/EXPLORE  -> against the DOM candidate list (snapshot is blind to the grid);
      - any ACT with a SELECTED product -> IDENTITY is AUTHORITATIVE: the loaded page must BE that
        product (ASIN/title). A match PASSES outright; a mismatch REJECTS. We do NOT fall through to
        the snapshot verifier, which is fuzzy enough to wave a home/search page through (fix 4 — the
        OnePlus-for-a-book bug + the "verifier passed on the home page" regression);
      - ADD-TO-CART -> the cart itself (badge / post-add panel), not the snapshot;
      - only WITHOUT a selection to match against -> the independent snapshot verifier."""
    if subgoal.get("type") == "explore":
        cands = await extract_candidates_dom(session)
        return verify_candidates(cands, subgoal.get("explore_spec"))
    identity = None
    if selected and (selected.get("url") or selected.get("asin")):
        ok, reason = verify_product_match(url, page_title, selected)
        if not ok:
            return False, reason            # wrong product loaded — reject regardless of page type
        identity = reason                   # the loaded page IS the selected product (ASIN/title)
    if _is_add_to_cart(subgoal):
        # Confirm on the CURRENT page (badge / post-add panel / 'added to cart'), NOT by going to
        # /cart (Amazon shows a smart-wagon interstitial, not the cart contents). On failure return a
        # SPECIFIC reason that tells the reasoner NOT to re-navigate (fix 1/2). cart_confirmed is the
        # authority here — identity (above) only guards that we're confirming the RIGHT product.
        title = (selected or {}).get("name") or (selected or {}).get("title") if isinstance(selected, dict) else None
        ok, signal = await cart_confirmed(session, title)
        if ok:
            return True, f"item added to cart ({signal})"
        return False, (f"add-to-cart not confirmed: {signal}. Do NOT navigate to the product page or "
                       "the cart — click 'Add to Cart' on the product page and let the on-page "
                       "confirmation appear, then complete.")
    if identity is not None:
        # Identity matched and this isn't add-to-cart: the page IS the selected product, so the
        # subgoal (open/inspect it) is genuinely complete. Authoritative — skip the snapshot verifier
        # (`identity` already reads as a reason, e.g. "on the selected product (ASIN …)").
        return True, identity
    return await verify_success(groq, snapshot_text, subgoal["success_condition"], model=model)


async def _approve_with_timeout(approve, prompt: str, timeout: float = APPROVAL_TIMEOUT) -> bool:
    """Await an approval, but treat no-answer-in-time as a denial. Sync (CLI) callbacks aren't
    timed (a human is at the terminal); async (web) ones are."""
    value = approve(prompt)
    if asyncio.iscoroutine(value):
        try:
            return bool(await asyncio.wait_for(value, timeout))
        except asyncio.TimeoutError:
            return False
    return bool(value)


async def run_subgoal(groq, session, reasoner_tools, subgoal, *, allowlist, approve, ask,
                      observations, max_steps, model=MODEL, read_content=True, emit=None,
                      plan_context="", selected=None) -> tuple[str, str]:
    """Drive one subgoal. Returns (status, detail) where status in
    {'complete','escalate','exhausted','denied'}. `emit(event_dict)` (optional) streams progress
    to a UI; prints are kept for the CLI. When `read_content` is set and a page's snapshot is too
    sparse to reason over, the content extractor (Readability.js -> trafilatura) turns the live page
    into clean text appended to the observation — this replaced the old screenshot/vision fallback."""
    emit = emit or (lambda e: None)
    trace = trace_enabled()     # DEBUG/TRACE -> emit the deep per-turn record (obs summary + raw args)
    recent_actions: list[str] = []
    last_sig = None
    repeat = 0
    last_action = None
    name_repeat = 0
    last_sem_sig = None         # (action, url-no-query) from the prior step (semantic loop, fix 5)
    sem_repeat = 0
    sem_loop_hits = 0           # times a semantic loop was blocked -> escalate when it persists
    complete_rejects = 0
    last_rejection = ""         # the most recent verifier rejection reason, surfaced to the reasoner (5b)
    working_tab = None  # the tab the agent drives; we keep focus pinned here across new-tab popups
    previous_keys: set = set()  # interactive-element keys from the last step -> mark NEW ones (*)
    dom = None                  # the current indexed-DOM view (None when we fell back to a11y)
    last_thought = ""           # the reasoner's note from the prior turn, echoed back for continuity
    last_fp = None              # page fingerprint from the prior step (stagnation detection)
    stagnant = 0                # consecutive steps the page hasn't changed despite acting
    budget_warned = False
    acted = False               # did the PRIOR step execute a page action? (gates stagnation)
    visited_urls: deque = deque(maxlen=URL_REVISIT_WINDOW)  # normalized URLs the agent has been ON
    revisit_nudges = 0          # times we blocked a navigate-back-to-a-recent-URL (ping-pong guard)
    is_explore = subgoal.get("type") == "explore"
    # At the cart/checkout stage the item is already added, so navigating BACK to the selected
    # product page is the post-add ping-pong — block it (fix 2).
    cart_stage = _is_cart_stage(subgoal)
    selected_path = _norm_url(selected["url"]) if isinstance(selected, dict) and selected.get("url") else ""
    is_add_to_cart = _is_add_to_cart(subgoal)
    sel_title = ((selected or {}).get("name") or (selected or {}).get("title")) if isinstance(selected, dict) else None
    atc_clicked = False         # add-to-cart: clicked the button deterministically (fix 2)?
    atc_confirm_polls = 0       # how many re-perceives we've waited for the cart to confirm
    atc_offpage_nudged = False  # nudged the reasoner once that we're not on the selected product yet?
    variant_done = False        # add-to-cart: already selected a required size/variant (no re-select)?
    # A size stated in the goal/clarifications steers variant selection; else first in-stock is picked.
    preferred_size = _size_hint(f"{subgoal.get('goal', '')} {plan_context}")
    is_checkout = _is_checkout(subgoal)   # a checkout/payment subgoal -> deterministic proceed + stop
    ckout_polls = 0             # how many re-perceives we've waited to land on the checkout page
    cod_tried = False           # already attempted the best-effort COD pre-selection?
    cod_wanted = any(p in f"{subgoal.get('goal', '')} {subgoal.get('success_condition', '')} {plan_context}".lower()
                     for p in ("cash on delivery", "cod", "cash-on-delivery"))
    listing_ready = False       # explore: a product grid is on the page -> restrict tools, complete
    listing_announced = False   # so we tell the reasoner "listing is ready" only once

    for step in range(1, max_steps + 1):
        # Keep the agent and the user on the SAME tab. `working_tab` follows the tab a click opened
        # (set after the action below). Before observing, make sure that tab is the active one — so
        # the snapshot (and any in-page prompt overlay) target the tab you're looking at, not the
        # opener. The first step establishes the working tab.
        cur, idxs = await list_tabs_state(session)
        if working_tab is None:
            working_tab = cur
        elif cur is not None and cur != working_tab and working_tab in idxs:
            await select_tab(session, working_tab)

        # PERCEPTION: build the indexed-DOM view (numbered interactive elements + NEW marks) as the
        # reasoner's primary observation — stable addresses instead of a11y refs. Fall back to the
        # accessibility snapshot only if the in-page build fails or finds nothing to act on.
        dom = await index_dom(session)
        if dom is not None and dom.elements:
            snapshot_text = dom.render(previous_keys)
            url = dom.url or url
            previous_keys = dom.keys
        else:
            snapshot_text, url = await observe(session)
            previous_keys = set()
        page_title = dom.title if dom is not None else ""   # for the product-identity verifier (fix 3)
        content_appended = False    # did the readable-text extractor add to this step's observation?
        cand_count = None           # DOM candidate count this step (set below when computed; for trace)

        # Append cleaned page text when there's little to act on (a content page / opaque app): the
        # reasoner needs something to read. Gate on the indexed view when we have one, else on the
        # legacy a11y sufficiency check — so a normal interactive page never triggers it.
        if read_content and (
                (dom is not None and len(dom.elements) < MIN_INTERACTIVE_FOR_TEXT)
                or (dom is None or not dom.elements) and not snapshot_is_sufficient(snapshot_text)):
            extra = await readable_text(session)
            if extra:
                log.debug("step %d: page thin to act on — appended %d chars of extracted content",
                          step, len(extra))
                emit({"type": "extract", "step": step, "chars": len(extra)})
                snapshot_text = f"{snapshot_text}\n\n### Extracted readable content\n{extra[:6000]}"
                content_appended = True

        # --- stagnation guard: did the page change since last step despite us acting? ---
        # `acted` still holds whether the PRIOR step ran a page action; use it before resetting.
        fp = _page_fingerprint(dom, url, snapshot_text)
        if fp != last_fp:
            stagnant = 0
        elif step > 1 and acted:
            stagnant += 1
        last_fp = fp
        if stagnant >= STAGNATION_LIMIT:
            # Inform-only: tell the reasoner its actions aren't changing the page and let IT decide
            # (change tactics or escalate). Reset so we nudge again only after another stagnant run.
            stagnant = 0
            recent_actions.append(
                "STAGNANT: the page has NOT changed despite your last few actions — they are having "
                "no effect. Do NOT repeat them. Try a DIFFERENT element, navigate directly to a URL, "
                "or scroll; if you genuinely cannot make progress, call escalate.")
            emit({"type": "stagnation_nudge", "step": step})

        # --- step-budget warning: near the end, push the reasoner to land it or escalate ---
        if not budget_warned and step >= max(2, int(max_steps * BUDGET_WARN_FRACTION)):
            budget_warned = True
            recent_actions.append(
                f"BUDGET: you have used {step}/{max_steps} steps. If the success condition is "
                f"satisfied, call subgoal_complete now; otherwise make your single most decisive "
                f"move (navigate directly to the target), or escalate if you are stuck.")
            emit({"type": "budget_warning", "step": step, "max_steps": max_steps})

        # Record the page we're on, for the navigate-back ping-pong guard below.
        if url:
            visited_urls.append(_norm_url(url))

        # DETERMINISTIC CHECKOUT: on a checkout/payment subgoal, click "Proceed to checkout" and STOP
        # the instant we reach the payment / order-review page — that page is the user's SECURED step
        # (they place the order). The reasoner must NOT drive the payment pipeline (the COD-selection
        # loop: it clicked payment radios + "Use this method" over and over until the budget ran out).
        # We never click an order-COMMIT control; if the user asked for COD we pre-select it best-effort.
        if is_checkout:
            reached, signal = await at_checkout(session)
            if reached:
                if cod_wanted and not cod_tried:
                    cod_tried = True
                    cod = await select_cod(session)
                    recent_actions.append(f"COD pre-select -> {'ok' if cod.get('ok') else cod.get('reason', 'not found')}")
                    emit({"type": "cod_select", "ok": bool(cod.get("ok")), "detail": cod.get("reason", "")})
                    await asyncio.sleep(0.4)
                detail = ("reached the checkout/payment page — ready for you to "
                          + ("confirm Cash on Delivery and place the order"
                             if cod_wanted else "select payment and place the order")
                          + f" ({signal})")
                emit({"type": "verifier", "satisfied": True, "reason": detail,
                      "condition": subgoal.get("success_condition", "")})
                return "complete", detail
            res = await proceed_to_checkout(session)
            if res.get("ok"):
                recent_actions.append(f"clicked Proceed to checkout (deterministic) -> {res.get('text', 'ok')}")
                emit({"type": "action_result", "action": "proceed_to_checkout", "ok": True,
                      "outcome": res.get("text", "ok")})
                await asyncio.sleep(0.8)
                continue                            # re-perceive: the checkout page should load
            ckout_polls += 1
            if ckout_polls >= 6:
                return "escalate", ("could not reach the checkout page — no 'Proceed to checkout' "
                                    "control found (the cart may be empty or a sign-in is required)")
            if ckout_polls == 1:
                # Steer the reasoner to the cart instead of fiddling with the payment pipeline — once a
                # cart page is shown WE click Proceed to checkout deterministically.
                recent_actions.append(
                    "CHECKOUT: navigate to the cart page first. Once it's shown I will click 'Proceed "
                    "to checkout' automatically — do NOT click payment options, 'Use this payment "
                    "method', or 'Place your order' yourself.")
            # No proceed control here yet -> let the reasoner navigate to the cart, then we take over.

        # DETERMINISTIC ADD-TO-CART (fix 2): on an add-to-cart subgoal, find + click the Add-to-Cart
        # control via the DOM the moment it's present on the page — the reasoner must NOT scroll-hunt
        # for it (the regression where it scrolled 4x and hit the loop hard-stop). After clicking,
        # CONFIRM on the current page (cart_confirmed) and complete; if it never confirms, escalate
        # with a specific reason rather than scrolling/re-navigating. The reasoner is used only to
        # NAVIGATE to the product page first (the selected-product note drives that).
        if is_add_to_cart:
            if atc_clicked:
                ok, signal = await cart_confirmed(session, sel_title)
                if ok:
                    emit({"type": "verifier", "satisfied": True, "reason": signal,
                          "condition": subgoal.get("success_condition", "")})
                    return "complete", f"added to cart ({signal})"
                atc_confirm_polls += 1
                # The add didn't register. The most common cause on fashion/footwear PDPs is an
                # unselected REQUIRED size/variant — pick one (preferred size, else first in-stock) and
                # retry the click ONCE before giving up (the asics/superkicks add-never-confirms bug).
                # We do this only AFTER a clicked add failed, so we're definitely on a product page (no
                # false positives from size FILTERS on a listing/category page).
                if not variant_done:
                    var = await select_required_variant(session, preferred_size)
                    if var.get("ok"):
                        variant_done = True
                        recent_actions.append(
                            f"selected required {var.get('kind', 'variant')} {var.get('selected')!r}; "
                            f"retrying add to cart")
                        emit({"type": "variant_selected", "kind": var.get("kind"),
                              "selected": var.get("selected")})
                        atc_clicked = False        # re-click add-to-cart now that a size is chosen
                        atc_confirm_polls = 0
                        await asyncio.sleep(0.5)   # let the variant's stock/button state update
                        continue
                    if var.get("kind") == "size":
                        # A size is required but none is selectable (e.g. every size sold out) — a clear
                        # dead end; escalate with the reason instead of blindly re-clicking.
                        return "escalate", (f"can't add to cart — {var.get('reason', 'no size available')} "
                                            f"for the selected product")
                    variant_done = True            # 'none'/'already' — nothing to select; don't re-probe
                if atc_confirm_polls >= 3:
                    return "escalate", ("clicked Add to Cart but the cart never confirmed — the add "
                                        "may have failed (sign-in / out of stock); do not retry blindly")
                await asyncio.sleep(0.8)            # let the AJAX / smart-wagon interstitial render
                continue
            # ADD ONLY FROM THE SELECTED PRODUCT'S OWN PAGE. On the cart or search-results page a
            # generic "Add to cart" button belongs to a RECOMMENDED item (e.g. "Rich Dad Poor Dad"
            # shown next to "Think and Grow Rich") — clicking it adds the WRONG book. If we know the
            # selected product, require its identity before the click; otherwise let the reasoner
            # navigate to it first (the selected-product note drives that).
            on_selected = True
            if isinstance(selected, dict) and (selected.get("url") or selected.get("asin")):
                on_selected, _id_reason = verify_product_match(url, page_title, selected)
            if not on_selected:
                if not atc_offpage_nudged:
                    atc_offpage_nudged = True
                    recent_actions.append(
                        "ADD-TO-CART: you are NOT on the selected product's page yet. Navigate to the "
                        "selected product URL FIRST — do NOT click an 'Add to cart' here (it would add a "
                        "recommended/related item, not the one chosen).")
                # fall through to the reasoner so it navigates to the selected product page
            else:
                res = await click_add_to_cart(session)
                if res.get("ok"):
                    atc_clicked = True
                    recent_actions.append(f"clicked Add to Cart (deterministic) -> {res.get('text', 'ok')}")
                    emit({"type": "action_result", "action": "add_to_cart", "ok": True,
                          "outcome": res.get("text", "ok")})
                    await asyncio.sleep(0.8)
                    continue                        # re-perceive: the page may go to the confirmation
                # button not on this page yet -> fall through; the reasoner navigates to the product page.

        # SIGN-IN WALL (fix 4): if we hit an auth page that ISN'T a planned login step, hand off to
        # the human (ask_human routes to the in-page overlay / chat in extension mode) — never let the
        # reasoner type fabricated credentials. Inform-only nudge; the no-fabrication prompt rule + the
        # reasoner choosing ask_human do the rest.
        goal_is_login = any(w in subgoal.get("goal", "").lower()
                            for w in ("sign in", "log in", "login", "sign-in", "authenticate"))
        if not goal_is_login and page_looks_like_login(snapshot_text, url):
            recent_actions.append(
                "SIGN-IN WALL: this is a login/authentication page. Do NOT type any email, password, "
                "or other credentials, and do NOT invent values. Call ask_human so the user can sign "
                "in, then continue.")
            emit({"type": "login_wall", "step": step, "url": url})

        # Fix 4b: on an EXPLORE subgoal, once a product grid is readable on the live DOM, drop
        # `navigate` and `click_element` from this turn's tools so the reasoner completes here
        # instead of clicking into a product (which starts the click<->navigate ping-pong). The
        # candidates are read off the page automatically on completion (the DOM extractor salvage).
        turn_tools = reasoner_tools
        if trace or (is_explore and not listing_ready):
            try:
                cand_count = len(await extract_candidates_dom(session))
            except Exception:  # best-effort — never block the turn on perception extras
                cand_count = None
        if is_explore and not listing_ready and cand_count is not None and cand_count >= EXPLORE_LISTING_MIN:
            listing_ready = True
        if listing_ready:
            turn_tools = [t for t in reasoner_tools
                          if t["function"]["name"] not in ("navigate", "click_element")]
            if not listing_announced:
                listing_announced = True
                recent_actions.append(
                    "LISTING READY: a product list is on this page and its candidates are read off "
                    "automatically — call subgoal_complete now. Do NOT navigate away or click into a "
                    "product.")

        acted = False  # reset for THIS step; set True below only if a page action runs
        thought, actions = await reasoner_decide(
            groq, turn_tools, subgoal, snapshot_text, recent_actions, last_thought,
            plan_context=plan_context, model=model, emit=emit, last_rejection=last_rejection)
        last_thought = thought
        actions = actions[:MAX_ACTIONS_PER_STEP]
        if not actions:                       # reasoner_decide always returns at least one
            return "escalate", "model returned no action"
        first_action = actions[0][0]
        log.info("step %d: %s%s — %s", step, first_action,
                 f" (+{len(actions) - 1} more)" if len(actions) > 1 else "", thought[:100])
        emit({"type": "step", "step": step, "thought": thought,
              "action": first_action, "args": actions[0][1],
              "actions": [{"action": n, "args": a} for n, a in actions]})
        # DEEP TRACE (fix 5a): a complete per-turn record to disk for debugging a failed run —
        # subgoal id/text, the observation summary the reasoner saw, its one-line rationale, and the
        # RAW args of every action it chose. Gated on BROWSER_AGENT_LOG_LEVEL=DEBUG/TRACE so a normal
        # run's JSONL stays small; the EventRecorder wired to `emit` persists it. Best-effort.
        if trace:
            emit({"type": "trace_turn", "step": step,
                  "subgoal_id": subgoal.get("id"), "subgoal_text": subgoal.get("goal"),
                  "observation": {"snapshot_chars": len(snapshot_text),
                                  "dom_extracted": bool(dom is not None and dom.elements),
                                  "dom_elements": (len(dom.elements) if dom is not None else 0),
                                  "content_appended": content_appended,
                                  "candidate_count": cand_count, "url": url},
                  "rationale": (thought or "").strip()[:300], "action": first_action,
                  "actions": [{"action": n, "args": a} for n, a in actions]})

        # --- a LEADING control action is the whole decision (any trailing actions are ignored) ---
        # Handled before loop detection so a legitimately repeated complete/ask isn't mistaken for a
        # navigation loop (the complete-reject standoff guard handles repeated completes instead).
        if first_action in ("subgoal_complete", "escalate", "ask_human"):
            _, args = actions[0]
            if first_action == "escalate":
                return "escalate", args.get("reason", "escalated")
            if first_action == "ask_human":
                answer = await maybe_await(ask(args.get("question", "(no question)")))
                recent_actions.append(f"ask_human -> {answer!r}")
                emit({"type": "ask_answer", "answer": answer})
                continue
            ok, reason = await _verify_complete(groq, session, subgoal, snapshot_text, model,
                                                selected=selected, url=url, page_title=page_title)
            log.info("verifier: satisfied=%s (%s)", ok, reason[:80])
            emit({"type": "verifier", "satisfied": ok, "reason": reason,
                  "condition": subgoal.get("success_condition", "")})
            if ok:
                return "complete", reason
            complete_rejects += 1
            if complete_rejects >= MAX_COMPLETE_REJECTS:
                # Standoff: stop letting the reasoner re-declare done forever. Hand back to the
                # planner — for an explore subgoal, orchestrate still salvages whatever is on the page.
                return "escalate", f"verifier refused completion {complete_rejects}x: {reason}"
            last_rejection = reason
            recent_actions.append(f"subgoal_complete -> REJECTED by verifier: {reason}")
            continue

        # --- loop handling: INFORM the reasoner, with a HIGH anti-runaway BACKSTOP ---
        # `repeat` counts consecutive IDENTICAL batches (exact same action+args); `name_repeat`
        # counts the same leading action NAME even as args wobble.
        sig = json.dumps([[n, a] for n, a in actions], sort_keys=True)
        repeat = repeat + 1 if sig == last_sig else 0
        last_sig = sig
        name_repeat = name_repeat + 1 if first_action == last_action else 0
        last_action = first_action
        # SEMANTIC signature (fix 5): the same action TYPE on the same page (URL with query + element
        # index stripped). Catches index-varying repeats the exact-arg `repeat` misses — clicking
        # result after result on one listing page, or re-issuing navigate to the same URL.
        sem_sig = (first_action, _norm_url(url))
        sem_repeat = sem_repeat + 1 if sem_sig == last_sem_sig else 0
        last_sem_sig = sem_sig

        # BACKSTOP: the exact same action LOOP_HARD_STOP times running is a genuine infinite loop —
        # force an escalate so a stuck agent can't burn the whole budget. (repeat is 0 on the 1st.)
        if repeat + 1 >= LOOP_HARD_STOP:
            emit({"type": "loop_hard_stop", "step": step, "kind": "exact", "signature": sig[:120]})
            return "escalate", f"loop backstop: repeated the exact same action {repeat + 1}x"

        # LOOP HANDLING. An EXACT repeat (3rd+ identical batch) is HARD-STOPPED (fix 5): block the
        # redundant action and re-decide. A SEMANTIC loop — the same NON-ADVANCING action on the same
        # page — is hard-stopped and escalated when it persists. A repeated NAVIGATE to the same page
        # is a real loop (3rd); a few exploratory SCROLLs are legitimate, so scroll is NOT early-blocked
        # and only trips the semantic stop after ~5 in a row (fix 3) — the exact-repeat backstop above
        # still ends an endless identical scroll. click/input/select on one page only nudge (a legit
        # sequence: pick size -> colour -> add to cart).
        sem_loop = (sem_repeat >= 2 and first_action == "navigate") or \
                   (sem_repeat >= 4 and first_action == "scroll_page")
        if repeat >= 2 or name_repeat >= 3 or sem_loop:
            recent_actions.append(
                f"LOOP: you keep choosing {first_action} with no new progress. Reach the goal a "
                f"DIFFERENT way (a different element, or navigate to a different URL), or call "
                f"subgoal_complete / escalate.")
            emit({"type": "loop_nudge", "step": step, "action": first_action})
            if sem_loop:
                sem_loop_hits += 1
                if sem_loop_hits >= 2:
                    emit({"type": "loop_hard_stop", "step": step, "kind": "semantic",
                          "signature": f"{first_action}@{_norm_url(url)}"})
                    return "escalate", (f"semantic loop hard-stop: repeated {first_action} on "
                                        f"{_norm_url(url)} despite nudges")
            # Block the redundant action (re-decide) for exact repeats and navigate loops; let a few
            # scrolls through (the backstop/semantic stop above still bound endless scrolling).
            if (repeat >= 2 or sem_loop) and first_action != "scroll_page":
                continue

        # --- execute the batch in order, behind the page-change guard (see dom_index) ---
        before_idxs = idxs
        for action, args in actions:
            # A control action reached mid-batch (e.g. [scroll, subgoal_complete]) ends the turn.
            if action == "escalate":
                return "escalate", args.get("reason", "escalated")
            if action == "ask_human":
                answer = await maybe_await(ask(args.get("question", "(no question)")))
                recent_actions.append(f"ask_human -> {answer!r}")
                emit({"type": "ask_answer", "answer": answer})
                break
            if action == "subgoal_complete":
                ok, reason = await _verify_complete(groq, session, subgoal, snapshot_text, model,
                                                    selected=selected, url=url, page_title=page_title)
                emit({"type": "verifier", "satisfied": ok, "reason": reason,
                      "condition": subgoal.get("success_condition", "")})
                if ok:
                    return "complete", reason
                complete_rejects += 1
                if complete_rejects >= MAX_COMPLETE_REJECTS:
                    return "escalate", f"verifier refused completion {complete_rejects}x: {reason}"
                last_rejection = reason
                recent_actions.append(f"subgoal_complete -> REJECTED by verifier: {reason}")
                break

            # RESOLVE the navigate target first (fix 2): a scraped href is often relative
            # (`/gp/product/…`), so origin-prefix it against the current page; reject a malformed /
            # unresolvable target (`https://gp/…`, `javascript:`) so we never navigate somewhere broken.
            if action == "navigate":
                resolved = resolve_url(args.get("url", ""), url)
                if not resolved:
                    bad = args.get("url", "")
                    recent_actions.append(f"navigate -> REJECTED: {bad!r} is not a usable URL "
                                          "(relative links are resolved automatically; give a real path).")
                    emit({"type": "action_result", "action": "navigate", "blocked": True,
                          "outcome": f"rejected malformed url {bad!r}"})
                    break  # re-decide next step
                args = {**args, "url": resolved}

                # POST-ADD GUARD (fix 2): at the cart/checkout stage, never navigate BACK to the
                # selected product page — the item is already added; doing so is the product<->cart
                # bounce. Block it with a specific reason; persistent attempts escalate (revisit cap).
                if cart_stage and selected_path and _norm_url(resolved) == selected_path:
                    revisit_nudges += 1
                    msg = ("the item is already in the cart — do NOT go back to the product page. "
                           "Confirm the cart / proceed to checkout here, or escalate if the cart is empty.")
                    log.info("step %d: blocked navigate-back-to-product (cart stage)", step)
                    recent_actions.append(f"navigate -> BLOCKED: {msg}")
                    emit({"type": "loop_nudge", "step": step, "action": "navigate", "blocked": True})
                    if revisit_nudges >= MAX_REVISIT_NUDGES:
                        return "escalate", ("cart not confirmed and the agent kept returning to the "
                                            "product page; the add-to-cart likely didn't register")
                    break

            # PING-PONG GUARD (fix 4a/5): a navigate to the page we're already on, or one we were on
            # within the last few steps, is the click->product->navigate-back-to-search bounce. Block
            # the redundant navigation, nudge, and re-decide; escalate only after repeated revisits.
            if action == "navigate":
                nt = _norm_url(args.get("url", ""))
                if nt and nt in visited_urls:
                    revisit_nudges += 1
                    msg = (f"already on / recently visited {nt} — NOT re-navigating there. The listing "
                           f"is here: read it and call subgoal_complete, or click a DIFFERENT element.")
                    log.info("step %d: blocked navigate-back to %s (revisit #%d)", step, nt, revisit_nudges)
                    recent_actions.append(f"navigate -> BLOCKED: {msg}")
                    emit({"type": "loop_nudge", "step": step, "action": "navigate", "blocked": True})
                    if revisit_nudges >= MAX_REVISIT_NUDGES:
                        return "escalate", f"ping-pong: navigated back to a recent URL {revisit_nudges}x"
                    break  # end the batch; re-perceive + re-decide next step

            # enforce the domain allowlist (only navigate / a link click can change origin)
            target_url = None
            if action == "navigate":
                target_url = args.get("url", "")
            elif action == "click_element" and dom is not None:
                try:
                    el = dom.selector_map.get(int(args.get("index")))
                except (TypeError, ValueError):
                    el = None
                if el and el.href.startswith(("http://", "https://")):
                    target_url = el.href
            if target_url and not domain_allowed(target_url, allowlist):
                msg = f"blocked by allowlist: {domain_of(target_url)!r} not in {sorted(allowlist)}"
                log.warning("%s", msg)
                recent_actions.append(f"{action} -> {msg}")
                emit({"type": "action_result", "action": action, "blocked": True, "outcome": msg})
                break  # a blocked navigation ends the batch

            res = await execute_action(session, action, args)
            acted = True  # a page action ran -> next step's stagnation check is meaningful
            rtext = res["outcome"]
            # FOLLOW a new tab the action opened (adopt + activate) so the agent works on the tab the
            # user is now looking at instead of being stranded on the opener.
            cur2, idxs2 = await list_tabs_state(session)
            adopt = newest_new_tab(before_idxs, idxs2)
            new_tab_opened = adopt is not None
            if new_tab_opened:
                working_tab = adopt
                if cur2 != adopt:
                    await select_tab(session, adopt)
                rtext += " (note: a new tab opened — followed it)"
            before_idxs = idxs2
            outcome = rtext[:150].replace("\n", " ")
            log.debug("step %d result: %s(%s) -> %s", step, action, json.dumps(args)[:60], outcome)
            recent_actions.append(f"{action}({json.dumps(args)[:60]}) -> {outcome}")
            emit({"type": "action_result", "action": action, "outcome": outcome, "ok": res["ok"]})

            # PAGE GUARD: stop after any action that may have changed the page (navigation, click,
            # submit) or that opened a new tab — re-perceive next step rather than act on a stale view.
            if new_tab_opened or is_terminating(action, args):
                break

    return "exhausted", "step budget exhausted"


async def orchestrate(groq, session, reasoner_tools, capabilities, goal, *, allowlist,
                      approve, ask, max_steps, max_replans, read_content, ground=True, emit=None,
                      preamble="", observations=None, model=MODEL,
                      parallel=False, browser="chrome") -> str:
    """Plan the goal into subgoals, run each as a verified reasoner loop, re-plan on escalation.
    `preamble` (session memory) is fed to the planner; `observations` (if provided) accumulates
    cross-subgoal readouts so the caller can persist them to the scratchpad. When `ground` is set,
    a quick live web search of the goal runs FIRST and its top results are prepended to the planner
    preamble, so the planner anchors on real, goal-appropriate sources (see agent/grounding.py)."""
    emit = emit or (lambda e: None)
    observations = observations if observations is not None else []

    working_goal = goal  # grows as the user clarifies via ask_user, so the rest re-plans around it

    # Ground the planner in real search results BEFORE planning. Runs in a HEADLESS throwaway browser
    # (offscreen) so the search engine is never shown in your visible tab — grounding is plumbing,
    # not the task. Computed once and reused across all plan/re-plan calls. Best-effort: "" if it
    # can't run (no search step shown, planning proceeds).
    grounding = ""
    if ground and working_goal and "http://" not in working_goal and "https://" not in working_goal:
        try:
            async with open_session(use_extension=False, browser=browser, headless=True) as (gsess, _gp):
                grounding = await web_grounding(gsess, working_goal)
        except Exception as e:  # best-effort: planning must proceed even if grounding can't launch
            log.debug("headless grounding failed (%r); proceeding without it", e)
    if grounding:
        emit({"type": "grounding", "text": grounding})
    plan_preamble = f"{grounding}\n\n{preamble}".strip() if grounding else preamble

    plan = await plan_subgoals(groq, working_goal, capabilities, preamble=plan_preamble, model=PLANNER_MODEL)
    _log_plan("plan", plan)
    emit({"type": "plan", "subgoals": plan})

    state: dict = {"candidates": [], "selected": None, "answers": [], "failed_sources": [],
                   "added_to_cart": False}
    replans = 0
    clarifications = 0
    i = 0
    while i < len(plan):
        sg = plan[i]
        sg_type = sg.get("type", "act")
        tier = sg.get("tier", "auto")
        log.info("subgoal %s (%s/%s): %s", sg["id"], sg_type, tier, sg["goal"])

        # --- PARALLEL EXPLORE: dispatch a run of consecutive explore subgoals concurrently, each on
        #     its own headless session, instead of one-at-a-time on the live browser (agent/gather).
        #     gather_parallel emits its own per-source subgoal_start/candidates/subgoal_end, so we
        #     skip the normal single-subgoal emit + loop below for these.
        if parallel and sg_type == "explore":
            run = [sg]
            j = i + 1
            while j < len(plan) and plan[j].get("type", "act") == "explore":
                run.append(plan[j])
                j += 1
            # Fan EACH explore out into one subgoal per source (a planner often emits a single
            # "gather from these sites" explore with several sources) so the parallel gather actually
            # engages instead of one tab flailing across the sites sequentially.
            expanded = [sub for e in run for sub in _expand_explore_sources(e)]
            if len(expanded) >= 2:
                log.info("dispatching %d explore source(s) in parallel", len(expanded))
                results = await gather_parallel(
                    groq, reasoner_tools, expanded, allowlist=allowlist, max_steps=max_steps + 6,
                    model=model, read_content=read_content, browser=browser, emit=emit)
                for sub in expanded:
                    cands = results.get(sub.get("id"), [])
                    if cands:
                        _finalize_candidates(cands)   # DOM rows are absolute; just validate/flag
                        state["candidates"].extend(cands)
                        observations.append(
                            f"gathered {len(cands)} candidates from source (parallel)")
                log.info("parallel explore -> %d candidates total", len(state["candidates"]))
                i = j
                continue
            # a lone explore on a single source -> fall through to the normal sequential path below

        emit({"type": "subgoal_start", "id": sg["id"], "goal": sg["goal"], "kind": sg_type,
              "tier": tier, "success_condition": sg["success_condition"],
              "needs_approval": sg["needs_approval"]})

        # --- HITL: confirm/secured subgoals pause for approval BEFORE any side effect ---
        if tier in ("confirm", "secured") or sg["needs_approval"]:
            prompt = f"[{tier}] Subgoal {sg['id']}: {sg['goal']}"
            if not await _approve_with_timeout(approve, prompt):
                emit({"type": "subgoal_end", "id": sg["id"], "status": "denied", "detail": "declined/timeout"})
                return "Stopped: user declined approval."

        # --- ask_user: disambiguate (single/multi choice + "other"), then re-plan around the
        #     answer so the choice actually steers the remaining work ---
        if sg_type == "ask_user":
            question = sg.get("question", sg["goal"])
            # Cap clarifications: a weak planner tends to ask one preference at a time, re-planning
            # after each. Once we've asked enough, stop asking — skip this subgoal and let the rest
            # of the plan (explore/rank/present) run, exploring broadly.
            if clarifications >= MAX_CLARIFICATIONS:
                log.info("subgoal %s ask_user skipped — clarification cap (%d) reached; exploring broadly",
                         sg["id"], MAX_CLARIFICATIONS)
                observations.append(f"skipped further clarification '{question}' (cap reached)")
                emit({"type": "subgoal_end", "id": sg["id"], "status": "skipped",
                      "detail": "clarification cap reached"})
                i += 1
                continue
            clarifications += 1
            answer = await maybe_await(ask(question, options=sg.get("options"),
                                           allow_free_text=sg.get("allow_free_text", True),
                                           multi_select=sg.get("multi_select", False)))
            # "No preference / skip": don't narrow — tell the planner to explore broadly instead.
            if (answer or "").strip() in (SKIP_SENTINEL, ""):
                answer = "(no preference — explore all options)"
                clarification = (f"(User has NO preference about — {question}. Do NOT narrow on "
                                 f"this; explore broadly across all reasonable options and compare them.)")
            else:
                clarification = f"(User clarified — {question}: {answer})"
            state["answers"].append(answer)
            observations.append(f"user answered '{question}': {answer}")
            emit({"type": "subgoal_end", "id": sg["id"], "status": "answered", "detail": str(answer)})
            working_goal = f"{working_goal}\n{clarification}"
            plan = await plan_subgoals(groq, working_goal, capabilities, preamble=plan_preamble,
                                       observations="; ".join(observations[-5:]), model=PLANNER_MODEL)
            _log_plan(f"plan re-shaped around: {answer}", plan)
            emit({"type": "replan", "n": replans, "reason": f"clarified: {answer}", "subgoals": plan})
            i = 0
            continue

        # --- exploit: the LLM ranks/selects the best candidates (decision kept with the model;
        #     falls back to the deterministic rating scorer only if the LLM call fails) ---
        if sg_type == "exploit":
            sel = await select_candidates(groq, working_goal, state["candidates"], model=model)
            state["selected"] = sel["selected"]
            top3 = sel["top"][:3]
            detail = (f"selected {state['selected']}" if state["selected"] else "no candidates to select")
            log.info("subgoal %s exploit -> %s (%s)", sg["id"], detail, sel.get("reason", "")[:60])
            emit({"type": "exploit", "id": sg["id"], "selected": state["selected"],
                  "top": top3, "near_tie": False, "reason": sel.get("reason", "")})
            emit({"type": "subgoal_end", "id": sg["id"], "status": "complete", "detail": detail})
            observations.append(detail)
            i += 1
            continue

        # --- present: synthesize gathered candidates into a structured markdown shortlist (no LLM
        #     browsing). Its markdown becomes the run's final answer shown to the user. ---
        if sg_type == "present":
            md = await synthesize_options(groq, working_goal, state["candidates"],
                                          state.get("selected"), model=COMPOSITION_MODEL)
            state["summary"] = md
            log.info("subgoal %s present -> %d options synthesized", sg["id"], len(state["candidates"]))
            emit({"type": "present", "id": sg["id"], "markdown": md})
            emit({"type": "subgoal_end", "id": sg["id"], "status": "complete",
                  "detail": f"presented {len(state['candidates'])} options"})
            observations.append("presented a shortlist of options to the user")
            i += 1
            continue

        # --- explore: reasoner gathers, then we extract a structured candidate list ---
        # Explore is the heaviest subgoal kind — it has to search, route to one or more sources,
        # and read enough to gather candidates — so give it more headroom than an act subgoal.
        sg_steps = max_steps + 6 if sg_type == "explore" else max_steps
        prior_goals = [p["goal"] for p in plan[:i]]
        plan_context = (f"Subgoal {i + 1} of {len(plan)} in the overall plan."
                        + (f" Earlier subgoals already handled: {'; '.join(prior_goals[-4:])}."
                           if prior_goals else "")
                        # fix 2: once a product is selected, later act subgoals get its exact URL
                        # (or, at the cart/checkout stage, a "don't go back to the product" note)
                        + (_selected_note(state.get("selected"), sg) if sg_type != "explore" else ""))

        # --- RE-ADD GUARD: once the chosen item is in the cart, a re-plan that re-emits "add to cart"
        #     must NOT add it again — that puts a 2nd copy (or, off the product page, a RECOMMENDED
        #     item like "Rich Dad Poor Dad") in the cart. If we already added and the cart is non-empty,
        #     skip straight past this subgoal. ---
        if _is_add_to_cart(sg) and state.get("added_to_cart"):
            count = await cart_count(session)
            if isinstance(count, int) and count > 0:
                detail = f"already added to cart (count={count}) — not adding again"
                log.info("subgoal %s add-to-cart skipped: %s", sg["id"], detail)
                emit({"type": "subgoal_end", "id": sg["id"], "status": "complete", "detail": detail})
                observations.append(detail)
                i += 1
                continue

        # --- CART-CONFIRMATION GATE (fix 3): never run a checkout/payment subgoal until the item is
        #     actually in the cart. The agent used to click an unstable "Proceed" index and navigate
        #     straight to /checkout on an empty/unrecognized cart -> sign-in wall -> loop. If the cart
        #     can't be confirmed non-empty, escalate so the planner re-routes to add-to-cart first. ---
        if _is_checkout(sg):
            count = await cart_count(session)
            if not (isinstance(count, int) and count > 0):
                detail = (f"cart not confirmed (count={count}) before checkout — the item isn't in the "
                          f"cart yet. Add it to the cart and verify before proceeding to checkout.")
                log.info("subgoal %s checkout gated: %s", sg["id"], detail)
                emit({"type": "cart_gate", "id": sg["id"], "count": count})
                emit({"type": "subgoal_end", "id": sg["id"], "status": "escalate", "detail": detail})
                if replans >= max_replans:
                    return f"Failed: exhausted re-plan budget at subgoal {sg['id']} ({detail})."
                replans += 1
                observations.append(detail)
                plan = await plan_subgoals(groq, working_goal, capabilities, prior_plan=plan,
                                           failed_id=sg["id"], reason=detail,
                                           observations="; ".join(observations[-5:]),
                                           preamble=preamble, model=PLANNER_MODEL)
                _log_plan("plan revised (cart gate)", plan)
                emit({"type": "replan", "n": replans, "reason": detail, "subgoals": plan})
                i = 0
                continue

        status, detail = await run_subgoal(
            groq, session, reasoner_tools, sg, allowlist=allowlist, approve=approve, ask=ask,
            observations=observations, max_steps=sg_steps, model=model, read_content=read_content,
            emit=emit, plan_context=plan_context, selected=state.get("selected"))

        if sg_type == "explore" and status != "denied":
            # Extract whatever listings are on the current page — even when the reasoner escalated
            # or exhausted. In practice it almost always still landed on a results page (it just
            # never called subgoal_complete), so SALVAGING those candidates beats throwing the whole
            # attempt away and re-planning from scratch (the failure mode that exhausted the budget
            # while real listings sat on the page).
            # DOM-FIRST extraction: read structured rows straight from the live product grid
            # (agent/dom_extract) — deterministic, no LLM, no a11y-tree blindness (the Amazon
            # "0 candidates despite a full results page" bug). Scroll-to-reveal handles lazy grids.
            snapshot_text = ""
            page_url = await current_url(session)
            if is_search_engine(page_url):
                # The agent ended on a SEARCH-ENGINE results page (e.g. it fell back to Google). Its
                # "listings" are snippet junk, not buyable products — don't extract; force a re-plan
                # onto a real retailer (the SERP-scrape bug). page-level guard for a11y rows that
                # carry no per-row url/source for _drop_junk_candidates to key on.
                cands = []
                log.info("subgoal %s explore: on a search-engine page (%s) — skipping extraction",
                         sg["id"], domain_of(page_url) or page_url)
            else:
                cands = _drop_junk_candidates(await extract_with_scroll(session, sg.get("explore_spec")))
            if cands:
                log.info("subgoal %s explore -> %d candidates via DOM extractor", sg["id"], len(cands))
            elif not is_search_engine(page_url):
                # Fall back to the LLM a11y extractor (FULL untruncated snapshot — products sit past
                # observe()'s 12k cap), then the Readability/innerText salvage, only if the DOM
                # reader found nothing (an unknown site whose grid the generic heuristic missed).
                snapshot_text, page_url = await full_snapshot(session)
                cands = _drop_junk_candidates(
                    await extract_candidates(groq, snapshot_text, sg.get("explore_spec"), model=model))
                if not cands and read_content and not page_looks_blocked(snapshot_text):
                    extra = await readable_text(session)
                    if extra:
                        cands = _drop_junk_candidates(
                            await extract_candidates(groq, extra, sg.get("explore_spec"), model=model))
                if cands:
                    log.info("subgoal %s explore -> %d candidates via a11y/text fallback",
                             sg["id"], len(cands))
            if cands:
                # Resolve every candidate's product url to an absolute http(s) URL (fix 1): DOM rows
                # are already absolute (a.href); LLM/a11y rows may be relative -> resolve against the
                # page. The buy flow navigates by this url, so an unresolvable one is flagged None.
                _finalize_candidates(cands, page_url)
                state["candidates"].extend(cands)  # accumulate across multiple explore subgoals
                salvaged = " [salvaged]" if status != "complete" else ""
                log.info("subgoal %s explore -> +%d candidates (%d total)%s",
                         sg["id"], len(cands), len(state["candidates"]), salvaged)
                emit({"type": "candidates", "id": sg["id"], "count": len(cands),
                      "total": len(state["candidates"]), "candidates": cands[:10]})
                observations.append(f"gathered {len(cands)} candidates ({len(state['candidates'])} total)")
                status, detail = "complete", f"gathered {len(cands)} candidates"  # proceed, don't re-plan
            else:
                # Nothing extracted here. Work out WHY so the re-plan can steer to a DIFFERENT
                # source instead of retrying the same dead end: record this source as failed and,
                # if it is a bot-check / captcha wall, say so. (A blocked or empty page often still
                # has refs, so it looked "sufficient" to the reasoner — but there is nothing to read.)
                src = domain_of(page_url)
                blocked = page_looks_blocked(snapshot_text)
                if src and src not in state["failed_sources"]:
                    state["failed_sources"].append(src)
                wall = f"looks like a {blocked}" if blocked else "no listings on the page"
                avoid = (f" Already tried (avoid these): {', '.join(state['failed_sources'])}."
                         if state["failed_sources"] else "")
                log.info("subgoal %s explore -> 0 candidates from %s%s",
                         sg["id"], src or "page", f" [{blocked}]" if blocked else "")
                emit({"type": "candidates", "id": sg["id"], "count": 0,
                      "total": len(state["candidates"]), "blocked": bool(blocked), "source": src})
                if state["candidates"]:
                    # Earlier explores already gathered options — one dead source is not fatal.
                    # Proceed to rank/present what we have instead of burning re-plans.
                    status, detail = "complete", (
                        f"no new candidates from {src or 'this page'} ({wall}); proceeding with "
                        f"{len(state['candidates'])} already gathered")
                else:
                    # No candidates anywhere yet — force a re-plan that routes to another source.
                    status, detail = "escalate", (
                        f"source {src or 'page'!r} yielded no candidates ({wall}). Route the explore "
                        f"step to a DIFFERENT retailer/source.{avoid}")
                observations.append(detail)

        log.info("subgoal %s -> %s: %s", sg["id"], status, detail)
        emit({"type": "subgoal_end", "id": sg["id"], "status": status, "detail": detail})

        if status == "complete":
            # Remember a successful add-to-cart so a later re-plan doesn't add the item a SECOND time
            # (the re-add guard above reads this).
            if _is_add_to_cart(sg):
                state["added_to_cart"] = True
            i += 1
            continue
        if status == "denied":
            return "Stopped: user declined."

        if replans >= max_replans:
            return f"Failed: exhausted re-plan budget at subgoal {sg['id']} ({detail})."
        replans += 1
        log.info("replan %d/%d: %s", replans, max_replans, detail)
        plan = await plan_subgoals(groq, working_goal, capabilities, prior_plan=plan,
                                   failed_id=sg["id"], reason=detail,
                                   observations="; ".join(observations[-5:]),
                                   preamble=preamble, model=PLANNER_MODEL)
        _log_plan("plan revised", plan)
        emit({"type": "replan", "n": replans, "reason": detail, "subgoals": plan})
        i = 0

    # A present subgoal's markdown is the user-facing answer; otherwise a plain completion.
    return state.get("summary") or "Done."
