"""
Full-text enrichment stage. Runs after filter, before classify.

Why: Serper returns 150-300 char snippets. For Habermasian classification
the validity-claim distinction often turns on context the snippet doesn't
include — a snippet might show 'NFCU charged me' (TRUTH-leaning) without
the surrounding 'and they should reverse this' (RIGHTNESS) that changes
the dominant claim.

Strategy: prioritize fetches by source-likelihood-of-yielding-signal and
mention-likelihood-of-being-substantive. Fetch the top N per run.

Bounded by:
- Per-run cap (default 50 fetches) to keep runs fast and polite
- Per-domain rate limiting (default 1 req/sec per domain)
- Skip patterns for domains that block automated fetches or yield no useful body
  (Twitter walls auth, LinkedIn shows snippets only without login, etc.)

Returns mentions with .text replaced by full content where successful,
.full_text_fetched=True, and original snippet preserved in .snippet.
"""

from __future__ import annotations
import time
from collections import defaultdict
from typing import Iterable
from urllib.parse import urlparse

import requests

from .schema import Mention, Source


# Sources where snippet is usually sufficient or fetching is blocked
_SKIP_FETCH_SOURCES: set[Source] = {
    Source.TWITTER,    # auth wall
    Source.LINKEDIN,   # auth wall (mostly)
    Source.YOUTUBE,    # video page, not text
}

# Per-source priority for fetching (higher = fetch first)
_SOURCE_PRIORITY: dict[Source, int] = {
    Source.REDDIT: 10,         # long-form discussion, big classification benefit
    Source.NEWS: 7,
    Source.INDUSTRY_PRESS: 7,
    Source.TRUSTPILOT: 5,      # snippet often captures the review
    Source.BBB: 5,
    Source.GENERAL_WEB: 4,
    Source.LINKEDIN: 0,
    Source.TWITTER: 0,
    Source.YOUTUBE: 0,
}


class WebFetchEnricher:
    """Fetches full text for selected mentions."""

    def __init__(
        self,
        max_fetches_per_run: int = 50,
        per_domain_min_interval: float = 1.0,
        timeout: int = 15,
        user_agent: str = "Mozilla/5.0 (compatible; VoC-Research/0.1)",
    ):
        self._max_fetches = max_fetches_per_run
        self._per_domain_min_interval = per_domain_min_interval
        self._timeout = timeout
        self._user_agent = user_agent
        self._last_fetch_per_domain: dict[str, float] = defaultdict(float)

    def enrich(self, mentions: list[Mention]) -> list[Mention]:
        """Return a new list with full text fetched for prioritized mentions."""
        # Sort by priority: source priority desc, then snippet length asc
        # (shorter snippets benefit most from full fetch)
        prioritized = sorted(
            mentions,
            key=lambda m: (
                -_SOURCE_PRIORITY.get(m.source, 0),
                len(m.snippet or m.text),
            ),
        )

        enriched: dict[str, Mention] = {m.id: m for m in mentions}
        fetches_done = 0

        for m in prioritized:
            if fetches_done >= self._max_fetches:
                break
            if m.source in _SKIP_FETCH_SOURCES:
                continue
            if m.full_text_fetched:
                continue
            full_text = self._fetch_one(m.url)
            if full_text:
                enriched[m.id] = m.model_copy(update={
                    "text": full_text,
                    "full_text_fetched": True,
                })
                fetches_done += 1

        return list(enriched.values())

    def _fetch_one(self, url: str) -> str | None:
        domain = urlparse(url).netloc
        # Per-domain rate limit
        elapsed = time.time() - self._last_fetch_per_domain[domain]
        if elapsed < self._per_domain_min_interval:
            time.sleep(self._per_domain_min_interval - elapsed)

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
            self._last_fetch_per_domain[domain] = time.time()
            if resp.status_code != 200:
                return None
            return self._extract_text(resp.text, url)
        except Exception as e:
            print(f"[fetch] {url}: {e}")
            return None

    def _extract_text(self, html: str, url: str) -> str:
        """Cheap text extraction. For v1 we use a minimal heuristic;
        v1.1 swaps in trafilatura or readability-lxml for better quality."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: very rough strip
            import re
            text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            return " ".join(text.split())[:8000]

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style/nav/footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Cap to avoid bloating LLM calls
        return text[:8000]
