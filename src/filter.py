"""
Filter stage. Drops noise before classification.

v1 strategy:
- Hard relevance check: text must contain at least one configured brand term
- Error-page detector: drops mentions whose enriched text is clearly a 404 or
  maintenance page rather than real content
- Sidebar-noise detector: drops mentions where the brand only appears inside
  a category-listing/review-count snippet on a page that's actually about
  another brand (e.g. Trustpilot's "related companies" sidebar)
- Length floor: very short mentions are kept but not specially flagged
- Dedup: exact-text dedup per source for v1; v1.1 swaps in embedding-based

The relevance check is intentionally cheap. Anything that passes here goes
to the LLM, which can mark it UNRELATED in the topic classifier as the
proper second pass.
"""

from __future__ import annotations
import re
from typing import Iterable, Iterator

from .schema import Mention


# Phrases that indicate the fetched page was an error/maintenance page rather
# than real content. Conservative on purpose — false positives would drop real
# mentions, so each phrase needs to be unambiguous.
_ERROR_PAGE_MARKERS = (
    "we'll be right back",
    "we will be right back",
    "site is currently unavailable",
    "service temporarily unavailable",
    "503 service",
    "504 gateway",
    "page not found",
    "404 not found",
    "access denied",
    "error 403",
    "this page isn't available",
    "rate limit exceeded",
    "checking your browser before",  # cloudflare interstitials
    "please enable javascript",  # JS-rendered pages we couldn't fetch
)


def is_error_page(mention: Mention) -> bool:
    """Return True if the mention text looks like an error/maintenance page.

    Only fires on short text where the error markers dominate the content;
    a real article that mentions '404 not found' in a quote should pass.
    """
    text_lower = (mention.text or "").lower()
    if len(text_lower) > 600:
        # Long text — error markers are unlikely to be the primary content
        return False
    return any(marker in text_lower for marker in _ERROR_PAGE_MARKERS)


# Pattern that recognizes a directory-listing entry: brand name followed
# (within a short distance) by a domain + review count or rating.
# This is a *consequent* pattern — we look for it AFTER finding the brand,
# rather than trying to match the brand inside it.
# Examples this catches when applied right after a brand mention:
#   ". www.navyfederal.org•49K reviews. 4.4"
#   " | navyfederal.org · 1.3K reviews"
#   " www.example.com · 4.9 stars"
_POST_BRAND_LISTING = re.compile(
    r"\s*[\.\|·•,]?\s*"
    r"(?:www\.)?[a-z0-9\-]+\.(?:com|org|net|edu)"
    r"\s*[•·\|,]?\s*"
    r"(?:\d[\d,\.kKlLmM]*\s*(?:reviews?|stars?|rating)|\d+\.\d+\s*(?:stars?)?)",
    re.IGNORECASE,
)


def is_sidebar_noise(mention: Mention, brand_terms: list[str]) -> bool:
    """Drop mentions where the brand only appears as a directory-listing
    sidebar entry (Trustpilot-style "related companies", Yelp-style "similar
    businesses"). These pages aren't about the brand; the brand is just a
    sidebar link.

    Heuristic:
      1. Brand term is NOT in the title (the title is the page subject).
      2. Every body occurrence of a brand term is immediately followed
         (within a few characters) by a domain + reviews/rating pattern.
         That means the brand is being rendered as a directory entry.

    If both conditions hold, the page subject is something else and the
    brand mention is incidental.

    Conservative: short text only (<800 chars). Long enriched articles can
    have a sidebar plus real content; we only drop when the sidebar IS the
    content.
    """
    text = mention.text or ""
    title = (mention.title or "")
    text_lower = text.lower()
    title_lower = title.lower()

    if len(text) > 800:
        return False

    if any(term.lower() in title_lower for term in brand_terms):
        return False

    # Every brand-term occurrence in the body must be immediately followed
    # by a directory-listing pattern (domain + reviews/rating). If even one
    # mention sits in regular prose, the page is plausibly about the brand.
    LOOKAHEAD = 60  # chars after brand name to check for listing pattern

    found_any_brand = False
    for term in brand_terms:
        term_lower = term.lower()
        start = 0
        while True:
            idx = text_lower.find(term_lower, start)
            if idx < 0:
                break
            # Skip occurrences inside URLs — e.g. "navyfederal" inside
            # "www.navyfederal.org". The URL itself is already counted by
            # the preceding brand-name occurrence.
            preceding = text_lower[max(0, idx - 8):idx]
            if "www." in preceding or "://" in preceding or preceding.endswith("/"):
                start = idx + len(term_lower)
                continue
            found_any_brand = True
            after = text_lower[idx + len(term_lower):idx + len(term_lower) + LOOKAHEAD]
            # Allow optional "Credit Union" suffix between brand and domain
            after = re.sub(r"^\s*(?:credit union|federal credit union|fcu)?\s*",
                            "", after, count=1)
            if not _POST_BRAND_LISTING.match(after):
                # This brand mention is in prose, not a listing — keep
                return False
            start = idx + len(term_lower)

    return found_any_brand  # True only if all brand mentions were listing entries


def is_relevant(mention: Mention, brand_terms: list[str]) -> bool:
    haystack = (mention.text + " " + (mention.title or "")).lower()
    return any(term.lower() in haystack for term in brand_terms)


def dedup_exact(mentions: Iterable[Mention]) -> Iterator[Mention]:
    seen: set[tuple[str, str]] = set()
    for m in mentions:
        key = (m.source.value, m.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        yield m


def filter_pipeline_with_terms(mentions: Iterable[Mention], brand_terms: list[str]) -> list[Mention]:
    relevant = (m for m in mentions
                if is_relevant(m, brand_terms)
                and not is_error_page(m)
                and not is_sidebar_noise(m, brand_terms))
    return list(dedup_exact(relevant))


# Back-compat alias for the old smoke test
def filter_pipeline(mentions: Iterable[Mention]) -> list[Mention]:
    """Legacy entry point — assumes NFCU terms. Prefer filter_pipeline_with_terms."""
    return filter_pipeline_with_terms(mentions, [
        "navy federal", "navyfederal", "nfcu", "navy fcu", "navy fed"
    ])
