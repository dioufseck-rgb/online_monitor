"""
Serper-based multi-source collector.

One adapter, many sources. We use Google search (via Serper) as a unified
discovery layer across the open web, then route results to source-specific
parsers based on the URL pattern.

Why this design:
- Per-platform APIs have inconsistent auth, terms, rate limits, and approval gates
- Google has already indexed all of it
- Serper is a thin wrapper: 1 credit per query, snippets returned in 1-2s
- Adding a source = adding a query template + URL pattern, not a new SDK

Source routing happens by URL pattern at parse-time. A single web search for
'"navy federal" complaint' might surface Reddit, Trustpilot, and forum results
in one response — we tag each by inspecting the URL.

Time filtering: Serper supports `tbs=qdr:w` (past week), `qdr:d` (past day), etc.
We default to weekly cadence with `qdr:w`. For backfill or shorter windows, the
caller can override.

Endpoints used:
- /search   (web results)
- /news     (Google News results)

Cost accounting: each call is 1 credit normally, 2 credits for >10 results.
We default to num=10 to stay at 1 credit/call.
"""

from __future__ import annotations
import hashlib
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional
from urllib.parse import urlparse

import requests

from ..schema import Mention, Source
from .base import Collector


SERPER_BASE = "https://google.serper.dev"


# URL pattern -> Source mapping. Order matters; first match wins.
_URL_PATTERNS: list[tuple[str, Source]] = [
    ("reddit.com", Source.REDDIT),
    ("twitter.com", Source.TWITTER),
    ("x.com", Source.TWITTER),
    ("linkedin.com", Source.LINKEDIN),
    ("youtube.com", Source.YOUTUBE),
    ("youtu.be", Source.YOUTUBE),
    ("trustpilot.com", Source.TRUSTPILOT),
    ("bbb.org", Source.BBB),
    ("americanbanker.com", Source.INDUSTRY_PRESS),
    ("cutoday.info", Source.INDUSTRY_PRESS),
    ("cutimes.com", Source.INDUSTRY_PRESS),
    ("bankingdive.com", Source.INDUSTRY_PRESS),
    ("yelp.com", Source.YELP),
    ("instagram.com", Source.INSTAGRAM),
    ("tiktok.com", Source.TIKTOK),
    ("threads.net", Source.THREADS),
    ("threads.com", Source.THREADS),
    ("facebook.com", Source.FACEBOOK),
    ("fb.com", Source.FACEBOOK),
]


def classify_url(url: str) -> Source:
    """Map a result URL to a Source enum. Defaults to GENERAL_WEB."""
    domain = urlparse(url).netloc.lower()
    for pattern, source in _URL_PATTERNS:
        if pattern in domain:
            return source
    return Source.GENERAL_WEB


class SerperCollector(Collector):
    """Multi-source collector via Google search.

    v1.6 query strategy:
      - Brand terms are consolidated into one OR-clause per template (saves
        ~80% of credits vs running each brand_term as a separate query).
      - For templates with `intent_eligible: True`, the collector ALSO runs
        intent-keyword queries from the domain registry (complaint, fee,
        outage, etc.). Each intent surfaces customer-voice content the
        brand-only query misses.
      - For templates with `dayrange_eligible: True`, the collector runs
        each query against each of the past N days (via Google's `cdr:1`
        parameter) instead of a single weekly window. Probes show this
        breaks the per-query result cap and adds 5-7× more URLs.
      - Mentions surfaced via intent queries get tagged with `mention.intent`
        so the topic classifier can be cross-validated against the intent.

    Backwards-compatible: with `intent_queries=False, dayrange=False`, the
    collector runs the same way as v1.5 (one query per template per brand_term).
    """

    def __init__(
        self,
        api_key: str,
        brand: str,
        brand_terms: list[str],
        domain: str = "financial_services",
        query_templates: Optional[list[dict]] = None,
        num_results: int = 10,
        time_window: str = "qdr:w",  # past week (used when dayrange=False)
        rate_limit_sleep: float = 0.1,
        intent_queries: bool = False,
        dayrange: bool = False,
        dayrange_days: int = 7,
    ):
        """
        Args:
            api_key: Serper API key
            brand: Canonical brand name for tagging mentions ('nfcu', 'nike', etc.)
            brand_terms: List of search-term variants to consolidate into one
                         OR-clause (e.g., ['navy federal', 'navyfederal', 'nfcu']
                         becomes `("navy federal" OR navyfederal OR nfcu)`)
            domain: Domain registry key — drives intent_queries from domain config
            query_templates: List of template dicts. If None, uses _DEFAULT_TEMPLATES.
            num_results: Results per query (Google caps at ~10 currently regardless,
                         per probe finding May 2026; left configurable for future)
            time_window: Default tbs param when dayrange is False
            rate_limit_sleep: Seconds between calls
            intent_queries: If True, also run intent-keyword queries for templates
                            marked intent_eligible. Intent keywords come from the
                            domain registry.
            dayrange: If True, fan out queries across past N days instead of one
                      weekly window. Applied only to templates marked
                      dayrange_eligible.
            dayrange_days: Number of past days to query when dayrange=True
        """
        self._api_key = api_key
        self._brand = brand
        self._brand_terms = brand_terms
        self._domain = domain
        self._num_results = num_results
        self._time_window = time_window
        self._rate_limit_sleep = rate_limit_sleep
        self._templates = query_templates or _DEFAULT_TEMPLATES
        self._intent_queries_enabled = intent_queries
        self._dayrange_enabled = dayrange
        self._dayrange_days = dayrange_days

        # Resolve intent queries from the domain registry once
        self._intent_specs: list[tuple[str, str]] = []
        if intent_queries:
            try:
                from ..domains import get_domain
                self._intent_specs = list(get_domain(domain).intent_queries)
            except (ValueError, ImportError):
                # Unknown domain or no intent_queries on it — skip intent queries
                self._intent_specs = []

    @property
    def source_name(self) -> str:
        return f"serper({self._brand})"

    def collect(self, since: datetime, until: datetime) -> Iterator[Mention]:
        """Run the configured query strategy and yield deduped Mentions.

        Strategy: for each template, build the appropriate query plan
        (brand-only ± intents) × (single weekly window ± per-day windows),
        run all queries, dedup by URL, yield mentions that fall in the
        [since, until) window.

        Mentions are tagged with `intent` (the keyword that surfaced them)
        on first discovery — so URL dedup keeps the first intent.
        """
        seen_urls: set[str] = set()
        seen_intent: dict[str, Optional[str]] = {}  # url -> intent

        # Build the brand OR-clause once
        brand_or = self._brand_or_clause()

        # Build the time-window plan (one entry if not dayrange, else N entries)
        windows = self._build_window_plan()

        # Total query budget — useful for sanity in logs
        total_queries = self._estimate_query_count(windows)
        print(f"[serper:{self._brand}] Query plan: ~{total_queries} queries "
              f"(intent={self._intent_queries_enabled}, dayrange={self._dayrange_enabled})")

        for tmpl in self._templates:
            tmpl_name = tmpl.get("name", "?")
            endpoint = tmpl.get("endpoint", "search")
            suffix = tmpl.get("q_suffix", "").strip()

            # 1) Brand-only query (always runs, against the default time window —
            # day-range fan-out doesn't apply to brand-only since it returns
            # mostly brand-owned content regardless of date).
            brand_query = self._compose_query(brand_or, suffix=suffix, intent_clause=None)
            yield from self._run_one(
                endpoint, brand_query, tbs=self._time_window,
                seen_urls=seen_urls, seen_intent=seen_intent,
                since=since, until=until, intent=None, tmpl_name=tmpl_name,
            )

            # 2) Intent queries (only on intent_eligible templates)
            if self._intent_queries_enabled and tmpl.get("intent_eligible") and self._intent_specs:
                # If dayrange ALSO enabled and template is dayrange-eligible,
                # fan out across days; otherwise single weekly window.
                template_windows = (
                    windows if (self._dayrange_enabled and tmpl.get("dayrange_eligible"))
                    else [self._time_window]
                )
                for intent_label, intent_clause in self._intent_specs:
                    intent_query = self._compose_query(brand_or, suffix=suffix,
                                                       intent_clause=intent_clause)
                    for tbs in template_windows:
                        yield from self._run_one(
                            endpoint, intent_query, tbs=tbs,
                            seen_urls=seen_urls, seen_intent=seen_intent,
                            since=since, until=until,
                            intent=intent_label, tmpl_name=tmpl_name,
                        )

            # 3) Brand-only dayrange — applies if dayrange enabled, template
            #    is dayrange_eligible, AND intent_queries is OFF (otherwise the
            #    intent loop above already does dayrange. Only matters for
            #    dayrange-without-intent runs).
            elif self._dayrange_enabled and tmpl.get("dayrange_eligible"):
                for tbs in windows:
                    if tbs == self._time_window:
                        continue  # already ran brand-only at this window
                    yield from self._run_one(
                        endpoint, brand_query, tbs=tbs,
                        seen_urls=seen_urls, seen_intent=seen_intent,
                        since=since, until=until, intent=None, tmpl_name=tmpl_name,
                    )

    # ----- internals -----

    def _brand_or_clause(self) -> str:
        """Consolidate brand_terms into a single OR-clause.

        ['navy federal', 'navyfederal', 'nfcu'] →
        '("navy federal" OR navyfederal OR nfcu)'
        """
        parts = []
        for term in self._brand_terms:
            if " " in term:
                parts.append(f'"{term}"')
            else:
                parts.append(term)
        return "(" + " OR ".join(parts) + ")"

    def _build_window_plan(self) -> list[str]:
        """Return list of `tbs` values to query against. Single value when
        dayrange is off; N day-specific values when dayrange is on."""
        if not self._dayrange_enabled:
            return [self._time_window]

        from datetime import date as _date, timedelta as _timedelta
        today = _date.today()
        plan = []
        for offset in range(1, self._dayrange_days + 1):
            d = today - _timedelta(days=offset)
            # Google expects M/D/YYYY (no leading zeros)
            plan.append(f"cdr:1,cd_min:{d.month}/{d.day}/{d.year},"
                        f"cd_max:{d.month}/{d.day}/{d.year}")
        return plan

    def _compose_query(self, brand_or: str, suffix: str = "",
                       intent_clause: Optional[str] = None) -> str:
        """Assemble a single query string from brand_or + optional intent + suffix."""
        parts = [brand_or]
        if intent_clause:
            parts.append(intent_clause)
        if suffix:
            parts.append(suffix)
        return " ".join(parts)

    def _estimate_query_count(self, windows: list[str]) -> int:
        """Rough credit-cost estimate for logging."""
        n = 0
        for tmpl in self._templates:
            # 1 brand-only query at the default window
            n += 1
            if self._intent_queries_enabled and tmpl.get("intent_eligible"):
                template_windows = (
                    len(windows) if (self._dayrange_enabled and tmpl.get("dayrange_eligible"))
                    else 1
                )
                n += len(self._intent_specs) * template_windows
            elif self._dayrange_enabled and tmpl.get("dayrange_eligible"):
                # day-range without intent: extra (windows - 1) brand-only queries
                n += max(0, len(windows) - 1)
        return n

    def _run_one(self, endpoint: str, query: str, tbs: str,
                 seen_urls: set, seen_intent: dict, since: datetime, until: datetime,
                 intent: Optional[str], tmpl_name: str) -> Iterator[Mention]:
        """Execute one Serper query, yield window-filtered + dedup'd Mentions."""
        try:
            results = self._call_serper(endpoint, query, tbs=tbs)
        except Exception as e:
            print(f"[serper:{self._brand}] {tmpl_name}/{intent or 'brand'} query failed: {e}")
            return

        for raw in results:
            url = raw.get("link") or raw.get("url")
            if not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            seen_intent[url] = intent

            mention = self._raw_to_mention(raw, endpoint)
            if mention is None:
                continue

            # Stamp the intent that first surfaced this URL
            mention.intent = intent

            # Window filter — Serper time-bounds are approximate
            if mention.timestamp < since or mention.timestamp >= until:
                continue

            yield mention

        time.sleep(self._rate_limit_sleep)

    def _call_serper(self, endpoint: str, query: str, tbs: Optional[str] = None) -> list[dict]:
        url = f"{SERPER_BASE}/{endpoint}"
        payload = {
            "q": query,
            "num": self._num_results,
            "tbs": tbs if tbs is not None else self._time_window,
        }
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Serper returns different keys per endpoint
        if endpoint == "news":
            return data.get("news", [])
        return data.get("organic", [])

    def _raw_to_mention(self, raw: dict, endpoint: str) -> Optional[Mention]:
        url = raw.get("link") or raw.get("url")
        if not url:
            return None

        title = raw.get("title", "")
        snippet = raw.get("snippet", "")
        if not (title or snippet):
            return None

        text = f"{title}\n\n{snippet}".strip()

        ts = self._parse_timestamp(raw)

        source = classify_url(url)

        # Stable id: hash of url so re-runs dedup
        mention_id = f"{source.value}:{hashlib.sha1(url.encode()).hexdigest()[:16]}"

        author = raw.get("source") or raw.get("author") or urlparse(url).netloc

        return Mention(
            id=mention_id,
            source=source,
            brand=self._brand,
            domain=self._domain,
            author_handle=author,
            timestamp=ts,
            title=title,
            text=text,
            snippet=snippet,
            url=url,
            engagement={},  # Serper doesn't provide engagement; per-source enrichment can add later
            raw_payload={"serper_endpoint": endpoint, "position": raw.get("position")},
        )

    def _parse_timestamp(self, raw: dict) -> datetime:
        """Serper returns timestamps in several formats:
        - Absolute: '2026-05-06', 'May 6, 2026', '2026-05-06T12:34:56Z'
        - Relative: '3 days ago', '5 hours ago', '2 weeks ago', 'yesterday'
        - Sometimes missing entirely

        We parse what we can. Unparseable -> fallback to a safe in-window
        sentinel so the result isn't dropped by the window filter on the
        caller's side.
        """
        date_str = raw.get("date") or raw.get("publishedDate") or ""
        date_str = date_str.strip().lower()

        if date_str:
            # Try absolute formats first
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
                try:
                    return datetime.strptime(raw.get("date") or raw.get("publishedDate"), fmt).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

            # Relative formats: "N units ago", "yesterday", "today"
            now = datetime.now(tz=timezone.utc)
            if date_str in ("just now", "today"):
                return now
            if date_str == "yesterday":
                return now - timedelta(days=1)

            # "3 days ago", "5 hours ago", "2 weeks ago", "1 month ago", "1 year ago"
            import re
            m = re.match(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", date_str)
            if m:
                n = int(m.group(1))
                unit = m.group(2)
                deltas = {
                    "second": timedelta(seconds=n),
                    "minute": timedelta(minutes=n),
                    "hour": timedelta(hours=n),
                    "day": timedelta(days=n),
                    "week": timedelta(weeks=n),
                    "month": timedelta(days=30 * n),  # approximate
                    "year": timedelta(days=365 * n),  # approximate
                }
                return now - deltas[unit]

        # Unparseable: return a safe fallback. We use "now minus 1 second"
        # so the timestamp is unambiguously in the past and inside any
        # caller-specified window that includes the present.
        return datetime.now(tz=timezone.utc) - timedelta(seconds=1)


# -----------------------------------------------------------------------------
# Default query templates — multi-source coverage
# -----------------------------------------------------------------------------

_DEFAULT_TEMPLATES = [
    # `intent_eligible: True` means intent-keyword queries are run against this
    # template (it tends to surface customer-voice content). False means the
    # template stays brand-only — useful for sources dominated by brand-owned
    # content where intent queries would mostly return more brand marketing.
    # `dayrange_eligible: True` means day-by-day queries are run for this template.

    # Reddit — discussion forums; high customer-voice density. Note that
    # intent queries HURT Reddit yield: probe v1.6 found Reddit content uses
    # vernacular ("frustrated", "rant", "pissed") rather than formal terms
    # ("complaint", "problem"), so intent-keyword filtering shrinks rather
    # than expands Reddit results. Day-range also doesn't help Reddit —
    # empirically (v1.6.1 audit) Google returns the same Reddit URLs across
    # weekly and day-specific queries (dedup collapses to no lift).
    {"name": "reddit", "q_suffix": "site:reddit.com", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # News — Google News endpoint, no site filter
    {"name": "news", "q_suffix": "", "endpoint": "news",
     "intent_eligible": True, "dayrange_eligible": True},

    # LinkedIn — public posts; mostly brand-owned + employee_personal
    {"name": "linkedin", "q_suffix": "site:linkedin.com/posts", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},
    {"name": "linkedin_pulse", "q_suffix": "site:linkedin.com/pulse", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # Trustpilot — formal reviews. v1.6.1 audit: brand-only weekly returns
    # 1 result; intent queries return 0. Intent narrowing is too aggressive
    # for Trustpilot's weekly index density.
    {"name": "trustpilot", "q_suffix": "site:trustpilot.com", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # BBB — formal complaints. v1.6.1 audit: brand-only weekly returns 1 result;
    # intent queries return 0. BBB's index is too sparse for intent multiplication.
    {"name": "bbb", "q_suffix": "site:bbb.org", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # YouTube — mostly brand-owned + creator content
    {"name": "youtube", "q_suffix": "site:youtube.com", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # Industry press — small high-quality pool, brand-only is enough
    {"name": "industry_press", "q_suffix":
        "(site:americanbanker.com OR site:cutoday.info OR site:cutimes.com OR site:bankingdive.com)",
     "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},

    # General web — catch-all; most of the value comes from intent×day here
    {"name": "general", "q_suffix":
        "-site:reddit.com -site:linkedin.com -site:trustpilot.com -site:bbb.org -site:youtube.com",
     "endpoint": "search",
     "intent_eligible": True, "dayrange_eligible": True},

    # Yelp — branch-level customer reviews; new in v1.6
    {"name": "yelp", "q_suffix": "site:yelp.com", "endpoint": "search",
     "intent_eligible": False, "dayrange_eligible": False},
]