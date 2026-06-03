"""Deterministic tests for the candidate-aware verifier (fix 1), the navigate URL resolver (fix 2),
and the sign-in-wall detector (fix 4). All pure functions — no Groq, no browser."""

from browser_agent.agent.verifier import verify_candidates
from browser_agent.services.mcp_client import resolve_url
from browser_agent.utils.text import page_looks_like_login


# ----------------------------------------------------------------- fix 1: verify_candidates
def test_verify_candidates_passes_on_enough_priced_rows():
    rows = [{"name": "Think and Grow Rich", "price": "₹139", "rating": 4.5},
            {"name": "Think & Grow Rich (Hindi)", "price": "₹120", "rating": 4.3},
            {"name": "Think and Grow Rich 21st Century", "price": "₹388", "rating": 4.5}]
    ok, reason = verify_candidates(rows, {"target_count": 5})   # target 5 but 3 real rows is enough
    assert ok is True and "3 candidates" in reason


def test_verify_candidates_rejects_when_empty():
    ok, reason = verify_candidates([], {"target_count": 5})
    assert ok is False and "no candidates" in reason


def test_verify_candidates_rejects_too_few():
    ok, reason = verify_candidates([{"name": "Only one", "price": "₹10"}], {"target_count": 5})
    assert ok is False and "only 1" in reason


def test_verify_candidates_rejects_without_price_or_rating():
    rows = [{"name": "Nav Link A"}, {"name": "Nav Link B"}, {"name": "Nav Link C"}]
    ok, reason = verify_candidates(rows, {})
    assert ok is False and "price/rating" in reason


def test_verify_candidates_no_target_defaults_to_three():
    rows = [{"name": f"Book {i}", "rating": 4.0} for i in range(3)]
    assert verify_candidates(rows, None)[0] is True


# ----------------------------------------------------------------- fix 2: resolve_url
def test_resolve_url_origin_prefixes_relative():
    assert resolve_url("/gp/product/B0GVY", "https://www.amazon.in/s?k=x") == "https://www.amazon.in/gp/product/B0GVY"


def test_resolve_url_resolves_relative_without_leading_slash():
    # urljoin against the current path -> absolute on the same origin.
    assert resolve_url("gp/product/B0", "https://www.amazon.in/s?k=x") == "https://www.amazon.in/gp/product/B0"


def test_resolve_url_passes_absolute_through():
    assert resolve_url("https://www.amazon.in/dp/123", "https://x.com/") == "https://www.amazon.in/dp/123"


def test_resolve_url_rejects_dotless_host():
    # The "https://gp/product/…" ERR_NAME_NOT_RESOLVED case.
    assert resolve_url("https://gp/product/B0", "https://www.amazon.in/") is None


def test_resolve_url_rejects_non_http_scheme():
    assert resolve_url("javascript:void(0)", "https://x.com/") is None
    assert resolve_url("mailto:a@b.com", "https://x.com/") is None


def test_resolve_url_rejects_empty_and_baseless_relative():
    assert resolve_url("", "https://x.com/") is None
    assert resolve_url("/p", "") is None        # relative with no base to resolve against


# ----------------------------------------------------------------- fix 4: login-wall detection
def test_page_looks_like_login_by_url():
    assert page_looks_like_login("", "https://www.amazon.in/ap/signin?openid=...") is True
    assert page_looks_like_login("", "https://accounts.example.com/login") is True


def test_page_looks_like_login_by_body():
    assert page_looks_like_login("Sign in\nEmail or mobile\nPassword\nContinue", "https://x/account") is True


def test_page_not_login_on_ordinary_page():
    # A "Login" nav link alone must not trip it (needs a password cue + a sign-in phrase).
    assert page_looks_like_login("Home  Shop  Login  Cart", "https://shop/home") is False
    assert page_looks_like_login("Think and Grow Rich  ₹139  4.5 stars  Add to cart", "https://shop/p") is False


# ----------------------------------------------------------------- fix 3: product-identity match
from browser_agent.agent.verifier import verify_product_match


def test_verify_product_match_rejects_asin_mismatch():
    sel = {"name": "Think and Grow Rich", "url": "https://www.amazon.in/dp/9389931525", "asin": "9389931525"}
    ok, reason = verify_product_match("https://www.amazon.in/OnePlus-Nord/dp/B0ONEPLUS1/ref=x",
                                      "OnePlus Nord CE6 Lite", sel)
    assert ok is False and "WRONG product" in reason          # the OnePlus-for-a-book bug


def test_verify_product_match_passes_on_asin_match():
    sel = {"url": "https://www.amazon.in/x/dp/9389931525"}
    ok, _ = verify_product_match("https://www.amazon.in/Think/dp/9389931525/ref=sr_1_1",
                                 "Think and Grow Rich Book", sel)
    assert ok is True


def test_verify_product_match_title_fallback_when_no_asin():
    sel = {"name": "Think and Grow Rich", "url": "https://www.flipkart.com/think/p/itm1", "asin": "9788172345648"}
    ok_good, _ = verify_product_match("https://www.flipkart.com/think/p/itm1", "Think & Grow Rich - Buy Online", sel)
    ok_bad, _ = verify_product_match("https://www.flipkart.com/phone/p/itm2", "OnePlus Nord CE6", sel)
    assert ok_good is True and ok_bad is False


def test_verify_product_match_noop_without_selection():
    assert verify_product_match("https://x/dp/AAAAAAAAAA", "t", None)[0] is True
    assert verify_product_match("https://x/dp/AAAAAAAAAA", "t", {"name": "no url"})[0] is True
