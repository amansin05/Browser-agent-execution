"""Domain allowlist helpers (Phase 5 safety)."""

import re
from urllib.parse import urlparse

# General web search engines. A SERP is NOT a retailer/content source: its result blocks scrape into
# snippet "candidates" with junk titles + prices and no buyable product URL (the SERP-scrape bug).
# Mirrors the SKIP list the grounding extractor already uses (agent/grounding.py); shared so both the
# candidate pipeline and tab-parking can drop search engines.
_SEARCH_ENGINE_RE = re.compile(
    r"(^|\.)(google|bing|duckduckgo|yahoo|baidu|yandex|ecosia|startpage|ask|aol|qwant|brave)\.", re.I)


def domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_search_engine(url_or_host: str) -> bool:
    """True if the URL (or bare host) is a general web search engine — google.com, bing.com,
    duckduckgo, etc. Candidates harvested off a search-results page are snippet junk, never real
    products, so the pipeline drops them and a parked SERP tab is never reopened."""
    host = domain_of(url_or_host) or (url_or_host or "").strip().lower()
    return bool(_SEARCH_ENGINE_RE.search(host))


def domain_allowed(url: str, allowlist: set[str]) -> bool:
    """True if the URL's host equals or is a subdomain of any allowed domain. An empty
    allowlist (no --allow given) means unrestricted."""
    if not allowlist:
        return True
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in allowlist)
