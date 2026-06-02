"""Domain allowlist helpers (Phase 5 safety)."""

from urllib.parse import urlparse


def domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def domain_allowed(url: str, allowlist: set[str]) -> bool:
    """True if the URL's host equals or is a subdomain of any allowed domain. An empty
    allowlist (no --allow given) means unrestricted."""
    if not allowlist:
        return True
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in allowlist)
