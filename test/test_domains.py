"""Domain helpers — the allowlist matcher and the search-engine detector (utils/domains.py)."""

from browser_agent.utils.domains import domain_allowed, domain_of, is_search_engine


def test_domain_of():
    assert domain_of("https://www.amazon.in/s?k=x") == "www.amazon.in"
    assert domain_of("not a url") == ""


def test_domain_allowed_subdomain_aware():
    assert domain_allowed("https://m.github.com/x", {"github.com"})       # subdomain matches
    assert domain_allowed("https://github.com/x", {"github.com"})         # exact matches
    assert not domain_allowed("https://evil.com/x", {"github.com"})       # unrelated blocked
    assert domain_allowed("https://anything.com", set())                  # empty = unrestricted


def test_is_search_engine_flags_serps():
    for u in ("https://www.google.com/search?q=asics",
              "https://google.co.in/search?q=x",
              "https://www.bing.com/search?q=x",
              "https://duckduckgo.com/?q=x",
              "https://search.yahoo.com/search?p=x",
              "yandex.com",                       # bare host
              "google.com"):
        assert is_search_engine(u), u


def test_is_search_engine_passes_retailers():
    for u in ("https://www.amazon.in/s?k=asics",
              "https://www.flipkart.com/search?q=asics",
              "https://www.superkicks.in/collections/asics",
              "https://www.myntra.com/asics",
              "https://asics.co.in",
              "shop.com",
              ""):
        assert not is_search_engine(u), u
