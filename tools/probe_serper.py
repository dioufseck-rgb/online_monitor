"""
Serper coverage probe — measure what each proposed query change actually returns
before committing to code changes in the ingest pipeline.

Tests these hypotheses against a single brand (default: nfcu):
  A. num_results bump — does Google's index actually have more for these brands?
  B. OR-clause brand-term consolidation — does it return the union of separate
     queries, or does Serper internally cap it the same way?
  C. Reddit-comments-explicit query — does forcing inurl:comment surface comment
     URLs vs post landing pages?
  D. Time-window expansion — does qdr:m yield more BBB / Trustpilot than qdr:w?
  E. Yelp source — does site:yelp.com produce useful customer-voice mentions?

Outputs a markdown report with:
  - Raw count per variant
  - Credit cost per variant
  - URL overlap with baseline (existing v1.5 query set)
  - URL overlap between variants (so we can see incremental value)
  - Sample of NEW urls per variant (qualitative inspection)

Usage:
    python -m tools.probe_serper --brand nfcu
    python -m tools.probe_serper --brand gmu --domain higher_education
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests


def _load_dotenv() -> None:
    """Load .env from repo root if present. Existing env vars take precedence."""
    # tools/probe_serper.py -> repo root is two parents up
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()  # run at import time so argparse sees SERPER_API_KEY


SERPER_BASE = "https://google.serper.dev"


# Brand → (search terms, suggested subreddits) for probing
BRAND_PROFILES = {
    "nfcu": {
        "terms": ["navy federal", "navyfederal", "nfcu", "navy fcu", "navy fed"],
        "subreddits": ["NavyFederal"],
        "domain": "financial_services",
    },
    "bofa": {
        "terms": ["bank of america", "bofa", "bankofamerica"],
        "subreddits": ["BankOfAmerica"],
        "domain": "financial_services",
    },
    "chase": {
        "terms": ["chase bank", "jpmorgan chase", "chase"],
        "subreddits": ["Chase", "ChaseBank"],
        "domain": "financial_services",
    },
    "gmu": {
        "terms": ["george mason university", "gmu", "george mason"],
        "subreddits": ["gmu", "georgemason"],
        "domain": "higher_education",
    },
    "uva": {
        "terms": ["university of virginia", "uva"],
        "subreddits": ["UVA"],
        "domain": "higher_education",
    },
}


@dataclass
class ProbeResult:
    name: str
    description: str
    query: str
    endpoint: str
    num_results_requested: int
    time_window: str
    credit_cost: int                     # 1 per 10 results, rounded up
    raw_count: int = 0
    urls_returned: set[str] = field(default_factory=set)
    new_urls_vs_baseline: set[str] = field(default_factory=set)
    sample_titles: list[str] = field(default_factory=list)
    error: Optional[str] = None


def call_serper(api_key: str, endpoint: str, query: str,
                num: int = 10, tbs: str = "qdr:w",
                page: int = 1) -> dict:
    """Make one Serper call; return the raw response."""
    url = f"{SERPER_BASE}/{endpoint}"
    payload = {"q": query, "num": num, "tbs": tbs}
    if page > 1:
        payload["page"] = page
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_urls(data: dict, endpoint: str) -> list[dict]:
    """Pull the result list out of Serper's response."""
    if endpoint == "news":
        return data.get("news", [])
    return data.get("organic", [])


def credit_cost(num_results: int) -> int:
    """Serper bills 1 credit per 10 results, rounded up."""
    return max(1, (num_results + 9) // 10)


def run_probe(api_key: str, name: str, description: str,
              endpoint: str, query: str,
              num_results: int = 10, tbs: str = "qdr:w",
              page: int = 1) -> ProbeResult:
    """Run one probe and report what came back."""
    p = ProbeResult(
        name=name,
        description=description,
        query=query,
        endpoint=endpoint,
        num_results_requested=num_results,
        time_window=tbs,
        credit_cost=credit_cost(num_results),
    )

    try:
        data = call_serper(api_key, endpoint, query, num=num_results, tbs=tbs, page=page)
    except Exception as e:
        p.error = f"{type(e).__name__}: {e}"
        return p

    results = extract_urls(data, endpoint)
    p.raw_count = len(results)

    for r in results:
        url = r.get("link") or r.get("url")
        if url:
            p.urls_returned.add(url)
            if len(p.sample_titles) < 5:
                title = r.get("title", "<no title>")
                p.sample_titles.append(f"  {title[:80]}\n    -> {url[:120]}")

    return p


def build_baseline_probes(terms: list[str]) -> list[dict]:
    """Reproduce the v1.5 default query set: 9 templates × N brand terms.
    Returns a list of (name, query, endpoint, num_results, tbs) dicts."""
    templates = [
        ("reddit", "site:reddit.com", "search"),
        ("news", "", "news"),
        ("linkedin", "site:linkedin.com/posts", "search"),
        ("linkedin_pulse", "site:linkedin.com/pulse", "search"),
        ("trustpilot", "site:trustpilot.com", "search"),
        ("bbb", "site:bbb.org", "search"),
        ("youtube", "site:youtube.com", "search"),
        ("industry", "(site:americanbanker.com OR site:cutoday.info OR "
                     "site:cutimes.com OR site:bankingdive.com)", "search"),
        ("general", "-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                    "-site:bbb.org -site:youtube.com", "search"),
    ]
    probes = []
    for term in terms:
        quoted = f'"{term}"' if " " in term else term
        for tname, suffix, endpoint in templates:
            q = f"{quoted} {suffix}".strip() if suffix else quoted
            probes.append({
                "name": f"baseline.{tname}.{term.replace(' ', '_')}",
                "query": q,
                "endpoint": endpoint,
                "num_results": 10,
                "tbs": "qdr:w",
            })
    return probes


def build_monthly_probes(terms: list[str]) -> list[dict]:
    """Compare each template at qdr:w vs qdr:m to measure the volume bump
    from going monthly. The same template runs twice — once at each window —
    so we can directly compare yields.

    Note: bumping num_results to 30 here. The earlier probe established Google
    caps at ~10 for weekly NFCU queries; the open question is whether monthly
    queries actually return more (and whether num=30 finally pays off when
    Google has a deeper index for the longer window).
    """
    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    templates = [
        ("reddit", "site:reddit.com", "search"),
        ("news", "", "news"),
        ("linkedin", "site:linkedin.com/posts", "search"),
        ("trustpilot", "site:trustpilot.com", "search"),
        ("bbb", "site:bbb.org", "search"),
        ("youtube", "site:youtube.com", "search"),
        ("general", "-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                    "-site:bbb.org -site:youtube.com", "search"),
        ("yelp", "site:yelp.com", "search"),  # added per probe round 1 finding
    ]
    probes = []
    for name, suffix, endpoint in templates:
        q = f"{or_terms} {suffix}".strip() if suffix else or_terms
        # Weekly version (baseline-equivalent, num=30 to test ceiling)
        probes.append({
            "name": f"weekly.{name}",
            "query": q,
            "endpoint": endpoint,
            "num_results": 30,
            "tbs": "qdr:w",
            "description": f"{name} template, past week, num=30",
        })
        # Monthly version
        probes.append({
            "name": f"monthly.{name}",
            "query": q,
            "endpoint": endpoint,
            "num_results": 30,
            "tbs": "qdr:m",
            "description": f"{name} template, past month, num=30",
        })
    return probes


def build_dayrange_probes(terms: list[str], days_back: int = 7) -> list[dict]:
    """Test day-by-day querying via Google's `cdr:1,cd_min,cd_max` parameter.

    Runs the same query against each of the past N days as separate single-day
    windows, then compares aggregate unique URLs to a single weekly query.

    The hypothesis: each day-specific query returns Google's top-10 results
    PUBLISHED on that exact date, which differs day-to-day. Aggregating 7
    days' results dedups to 30-50 unique URLs vs 10 from one weekly query.

    Tests 1 high-yield template (general) × 7 days + 1 weekly baseline.
    ~8 credits per probe brand.
    """
    from datetime import date, timedelta

    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    template_suffix = ("-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                       "-site:bbb.org -site:youtube.com")
    q = f"{or_terms} {template_suffix}"

    probes = []

    # Weekly baseline (the comparator)
    probes.append({
        "name": "dayrange.weekly_baseline",
        "query": q,
        "endpoint": "search",
        "num_results": 10,
        "tbs": "qdr:w",
        "description": "Weekly window (qdr:w), 1 query — the comparator",
    })

    # 7 day-specific queries
    today = date.today()
    for offset in range(1, days_back + 1):
        d = today - timedelta(days=offset)
        # Google expects M/D/YYYY (no leading zeros)
        date_str = f"{d.month}/{d.day}/{d.year}"
        tbs = f"cdr:1,cd_min:{date_str},cd_max:{date_str}"
        probes.append({
            "name": f"dayrange.{d.isoformat()}",
            "query": q,
            "endpoint": "search",
            "num_results": 10,
            "tbs": tbs,
            "description": f"Day-specific: {d.isoformat()} (cdr:1)",
        })

    return probes


def build_intent_probes(terms: list[str]) -> list[dict]:
    """Test whether adding intent keywords (complaint, fee, outage, etc.) to
    the brand query reorders Google's ranking enough to surface a different
    URL set than the brand-only query.

    Tests the hypothesis: intent queries return customer-voice content at
    higher density than brand-only queries, and different intents return
    different URL sets (so they multiply, not duplicate).

    8 intent queries + 1 brand-only baseline = 9 queries, ~9 credits.
    """
    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    template_suffix = ("-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                       "-site:bbb.org -site:youtube.com")

    # Intent suffixes — each maps to a category of customer voice
    intents = [
        ("baseline", ""),
        ("complaint", "(complaint OR complaints OR problem)"),
        ("review", "(review OR reviews OR experience)"),
        ("fee", "(fee OR fees OR charged OR \"hidden fee\")"),
        ("scam", "(scam OR fraud OR \"unauthorized charge\")"),
        ("outage", "(outage OR down OR \"not working\" OR error)"),
        ("denied", "(denied OR declined OR rejected)"),
        ("vs", "(vs OR \"better than\" OR \"switching from\")"),
        ("praise", "(love OR \"highly recommend\" OR \"best bank\")"),
    ]

    probes = []
    for name, intent_suffix in intents:
        if intent_suffix:
            q = f"{or_terms} {intent_suffix} {template_suffix}"
        else:
            q = f"{or_terms} {template_suffix}"
        probes.append({
            "name": f"intent.{name}",
            "query": q,
            "endpoint": "search",
            "num_results": 10,
            "tbs": "qdr:w",
            "description": f"Intent: {name} (general-web template)",
        })

    return probes


def build_intent_x_day_probes(terms: list[str]) -> list[dict]:
    """Test whether intent × day stacks multiplicatively.

    Picks one source (general web), 4 high-yield intents, 4 days =
    16 queries. Asks: does the intent×day combination return URL sets
    that are pairwise different, or do they collapse to overlapping sets?

    Mostly a feasibility check — if the matrix multiplies, we know the
    150+ query production design is worth building. If it collapses,
    we choose either intent OR day-range, not both.

    16 queries, ~16 credits.
    """
    from datetime import date, timedelta

    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    template_suffix = ("-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                       "-site:bbb.org -site:youtube.com")

    # 4 intents — pick the highest-yield based on intent probe expectations
    intents = [
        ("complaint", "(complaint OR complaints OR problem)"),
        ("fee", "(fee OR fees OR charged)"),
        ("review", "(review OR reviews)"),
        ("outage", "(outage OR down OR \"not working\")"),
    ]

    # 4 days — recent enough that Google's index is dense
    today = date.today()
    days = [today - timedelta(days=offset) for offset in range(2, 6)]

    probes = []
    for intent_name, intent_suffix in intents:
        for d in days:
            date_str = f"{d.month}/{d.day}/{d.year}"
            tbs = f"cdr:1,cd_min:{date_str},cd_max:{date_str}"
            q = f"{or_terms} {intent_suffix} {template_suffix}"
            probes.append({
                "name": f"ixd.{intent_name}.{d.isoformat()}",
                "query": q,
                "endpoint": "search",
                "num_results": 10,
                "tbs": tbs,
                "description": f"Intent={intent_name} × Day={d.isoformat()}",
            })

    return probes


def build_pagination_probes(terms: list[str]) -> list[dict]:
    """Test whether Serper's `page` parameter actually returns a deeper index,
    and whether subsequent pages contain genuinely NEW URLs vs repeating page 1.

    Tests 3 high-yield templates × 3 pages each = 9 calls, ~9 credits.

    Each call costs 1 credit (page+num=10 stays at 1 credit). The question we
    want answered: when we ask Google for page 2 of "(navy federal OR ...)",
    do we get URLs that didn't appear on page 1, or does Google return the
    same top-10 again?
    """
    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    templates = [
        ("general", "-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                    "-site:bbb.org -site:youtube.com", "search"),
        ("news", "", "news"),
        ("reddit", "site:reddit.com", "search"),
    ]
    probes = []
    for tmpl_name, suffix, endpoint in templates:
        q = f"{or_terms} {suffix}".strip() if suffix else or_terms
        for page in (1, 2, 3):
            probes.append({
                "name": f"page{page}.{tmpl_name}",
                "query": q,
                "endpoint": endpoint,
                "num_results": 10,
                "tbs": "qdr:w",
                "page": page,
                "description": f"{tmpl_name} template, page {page}",
            })
    return probes


def build_daily_probes(terms: list[str]) -> list[dict]:
    """Test whether 7 daily fetches (qdr:d, run on different days but simulated
    here by querying right now) yield more unique URLs than one weekly fetch.

    NOTE: this probe is approximate. We can't actually go back in time and
    fetch 'yesterday's' results — Google won't return time-shifted snapshots.
    What we CAN test is whether qdr:d at this moment returns a different set
    than qdr:w, and how much overlap there is. If qdr:d returns mostly the
    same top URLs as qdr:w, daily aggregation won't help. If qdr:d returns a
    different ~10 (presumably the most recent), then accumulating 7 such
    days' worth would yield ~30-50 unique URLs across the week.

    Tests 3 high-yield templates × 2 windows = 6 calls, ~6 credits.
    """
    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"

    templates = [
        ("general", "-site:reddit.com -site:linkedin.com -site:trustpilot.com "
                    "-site:bbb.org -site:youtube.com", "search"),
        ("news", "", "news"),
        ("reddit", "site:reddit.com", "search"),
    ]
    probes = []
    for tmpl_name, suffix, endpoint in templates:
        q = f"{or_terms} {suffix}".strip() if suffix else or_terms
        for tbs_label, tbs in (("daily", "qdr:d"), ("weekly", "qdr:w")):
            probes.append({
                "name": f"{tbs_label}.{tmpl_name}",
                "query": q,
                "endpoint": endpoint,
                "num_results": 10,
                "tbs": tbs,
                "description": f"{tmpl_name} template, {tbs_label} window",
            })
    return probes


def build_variant_probes(terms: list[str], subreddits: list[str]) -> list[dict]:
    """The proposals we want to evaluate."""
    or_terms = "(" + " OR ".join(
        f'"{t}"' if " " in t else t for t in terms
    ) + ")"
    probes = []

    # ====== A. num_results bump on high-yield templates ======
    probes.append({
        "name": "A.general_n30",
        "query": f"{or_terms} -site:reddit.com -site:linkedin.com "
                 f"-site:trustpilot.com -site:bbb.org -site:youtube.com",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:w",
        "description": "General-web template at num=30 (3 credits)",
    })
    probes.append({
        "name": "A.news_n30",
        "query": or_terms,
        "endpoint": "news", "num_results": 30, "tbs": "qdr:w",
        "description": "News template at num=30 (3 credits)",
    })
    probes.append({
        "name": "A.reddit_n30",
        "query": f"{or_terms} site:reddit.com",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:w",
        "description": "Reddit template at num=30 (3 credits)",
    })

    # ====== B. OR-clause consolidation vs separate brand-term queries ======
    probes.append({
        "name": "B.general_or_clause",
        "query": f"{or_terms} -site:reddit.com -site:linkedin.com "
                 f"-site:trustpilot.com -site:bbb.org -site:youtube.com",
        "endpoint": "search", "num_results": 10, "tbs": "qdr:w",
        "description": f"OR-clause for all {len(terms)} brand terms in one query",
    })

    # ====== C. Reddit comments-explicit ======
    probes.append({
        "name": "C.reddit_comments",
        "query": f"{or_terms} site:reddit.com inurl:comment",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:w",
        "description": "Force Reddit comment URLs (inurl:comment)",
    })
    for sub in subreddits[:2]:
        probes.append({
            "name": f"C.subreddit_{sub}",
            "query": f"{or_terms} site:reddit.com/r/{sub}",
            "endpoint": "search", "num_results": 30, "tbs": "qdr:w",
            "description": f"Specifically target r/{sub} (where customers are)",
        })

    # ====== D. Time-window expansion for low-velocity sources ======
    probes.append({
        "name": "D.bbb_qdr_m",
        "query": f"{or_terms} site:bbb.org",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:m",
        "description": "BBB at qdr:m instead of qdr:w (filter post-hoc)",
    })
    probes.append({
        "name": "D.trustpilot_qdr_m",
        "query": f"{or_terms} site:trustpilot.com",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:m",
        "description": "Trustpilot at qdr:m instead of qdr:w",
    })

    # ====== E. New source: Yelp ======
    probes.append({
        "name": "E.yelp",
        "query": f"{or_terms} site:yelp.com",
        "endpoint": "search", "num_results": 30, "tbs": "qdr:w",
        "description": "Yelp customer reviews — currently not in pipeline",
    })

    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default="nfcu",
                        choices=list(BRAND_PROFILES.keys()),
                        help="Brand to probe against")
    parser.add_argument("--out", default="reports/serper_probe.md",
                        help="Output markdown path")
    parser.add_argument("--api-key", default=os.environ.get("SERPER_API_KEY"),
                        help="Serper API key (default: $SERPER_API_KEY)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run the v1.5 baseline set (uses lots of credits)")
    parser.add_argument("--monthly", action="store_true",
                        help="Run weekly-vs-monthly comparison probe instead of variants. "
                             "Pairs each template at qdr:w and qdr:m so you can directly "
                             "compare yields. ~48 credits for 8 templates.")
    parser.add_argument("--paginate", action="store_true",
                        help="Run pagination probe — page 1, 2, 3 across high-yield templates. "
                             "Tests whether Serper's `page` parameter actually returns deeper "
                             "results. ~9 credits.")
    parser.add_argument("--daily", action="store_true",
                        help="Run daily-vs-weekly comparison probe — qdr:d vs qdr:w on high-yield "
                             "templates. Tests whether daily fetches return different URL sets "
                             "than weekly fetches (which would justify daily aggregation). "
                             "~6 credits.")
    parser.add_argument("--dayrange", action="store_true",
                        help="Run day-specific probe — query each of the past 7 days separately "
                             "via Google's cdr:1 parameter, compare aggregate to one weekly "
                             "query. Tests whether day-by-day querying breaks the per-query cap. "
                             "~8 credits.")
    parser.add_argument("--intent", action="store_true",
                        help="Run intent-keyword probe — test whether adding 'complaint', 'fee', "
                             "'outage' etc. to the brand query surfaces different URLs than the "
                             "brand-only query. ~9 credits.")
    parser.add_argument("--intent-x-day", action="store_true",
                        help="Run intent × day stacking probe — 4 intents × 4 days = 16 queries, "
                             "tests whether intent and day-range stack multiplicatively. "
                             "~16 credits.")
    parser.add_argument("--rate-sleep", type=float, default=0.3,
                        help="Sleep between calls (seconds)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: SERPER_API_KEY not set in env, and --api-key not provided.",
              file=sys.stderr)
        sys.exit(1)

    profile = BRAND_PROFILES[args.brand]
    print(f"Probe brand: {args.brand}")
    print(f"  Terms: {profile['terms']}")
    print(f"  Subreddits: {profile['subreddits']}")
    print(f"  Domain: {profile['domain']}")
    print()

    baseline_probes = build_baseline_probes(profile["terms"])
    if args.monthly:
        variant_probes = build_monthly_probes(profile["terms"])
    elif args.paginate:
        variant_probes = build_pagination_probes(profile["terms"])
    elif args.daily:
        variant_probes = build_daily_probes(profile["terms"])
    elif args.dayrange:
        variant_probes = build_dayrange_probes(profile["terms"])
    elif args.intent:
        variant_probes = build_intent_probes(profile["terms"])
    elif getattr(args, "intent_x_day", False):
        variant_probes = build_intent_x_day_probes(profile["terms"])
    else:
        variant_probes = build_variant_probes(profile["terms"], profile["subreddits"])

    if args.baseline:
        print(f"Baseline set: {len(baseline_probes)} queries (~{len(baseline_probes)} credits)")
    else:
        print(f"Skipping baseline (use --baseline to include; would cost ~{len(baseline_probes)} credits)")
    variant_credits = sum(credit_cost(p["num_results"]) for p in variant_probes)
    print(f"Variant set: {len(variant_probes)} queries (~{variant_credits} credits)")
    print()

    total_credits_estimated = (
        len(baseline_probes) if args.baseline else 0
    ) + variant_credits

    confirm = input(f"This will use ~{total_credits_estimated} Serper credits. "
                    f"Continue? [y/N] ")
    if confirm.lower() not in ("y", "yes"):
        print("Cancelled.")
        sys.exit(0)

    print()

    # Run baseline first — used as the reference set for "new URLs"
    baseline_urls: set[str] = set()
    baseline_results: list[ProbeResult] = []

    if args.baseline:
        print(f"=== Running baseline ({len(baseline_probes)} queries) ===")
        for i, p in enumerate(baseline_probes, 1):
            print(f"  [{i}/{len(baseline_probes)}] {p['name']}")
            r = run_probe(args.api_key,
                          name=p["name"],
                          description=p.get("description", "v1.5 baseline"),
                          endpoint=p["endpoint"],
                          query=p["query"],
                          num_results=p["num_results"],
                          tbs=p["tbs"])
            baseline_results.append(r)
            baseline_urls.update(r.urls_returned)
            if r.error:
                print(f"      ERROR: {r.error}")
            else:
                print(f"      {r.raw_count} results")
            time.sleep(args.rate_sleep)

    print()
    print(f"=== Running variants ({len(variant_probes)} queries) ===")
    variant_results: list[ProbeResult] = []
    for i, p in enumerate(variant_probes, 1):
        print(f"  [{i}/{len(variant_probes)}] {p['name']}")
        r = run_probe(args.api_key,
                      name=p["name"],
                      description=p["description"],
                      endpoint=p["endpoint"],
                      query=p["query"],
                      num_results=p["num_results"],
                      tbs=p["tbs"],
                      page=p.get("page", 1))
        if baseline_urls:
            r.new_urls_vs_baseline = r.urls_returned - baseline_urls
        variant_results.append(r)
        if r.error:
            print(f"      ERROR: {r.error}")
        else:
            new_count = len(r.new_urls_vs_baseline) if baseline_urls else r.raw_count
            print(f"      {r.raw_count} results, {new_count} not in baseline")
        time.sleep(args.rate_sleep)

    # ====== Write the report ======
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Serper Coverage Probe — {args.brand.upper()}")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_")
    lines.append(f"_Brand: {args.brand} ({profile['domain']})_  ")
    lines.append(f"_Terms tested: {profile['terms']}_  ")
    if profile["subreddits"]:
        lines.append(f"_Subreddits tested: {profile['subreddits']}_")
    lines.append("")

    if args.baseline:
        baseline_credits = sum(r.credit_cost for r in baseline_results)
        lines.append("## Baseline (current v1.5 query set)")
        lines.append("")
        lines.append(f"- Queries run: {len(baseline_results)}")
        lines.append(f"- Credits used: {baseline_credits}")
        lines.append(f"- Total raw results: {sum(r.raw_count for r in baseline_results)}")
        lines.append(f"- Unique URLs after dedup: {len(baseline_urls)}")
        lines.append(f"- Dedup ratio: "
                     f"{sum(r.raw_count for r in baseline_results) - len(baseline_urls)} "
                     f"redundant results across queries")
        lines.append("")

        lines.append("### Baseline per-query yield")
        lines.append("")
        lines.append("| Query | Raw | Cumulative unique |")
        lines.append("|---|---:|---:|")
        cum_urls: set[str] = set()
        for r in baseline_results:
            cum_urls.update(r.urls_returned)
            lines.append(f"| {r.name} | {r.raw_count} | {len(cum_urls)} |")
        lines.append("")

    # ====== Intent probe section (only when --intent) ======
    if args.intent:
        lines.append("## Intent Keywords — Does Adding Topic Words Reorder Google's Top-10?")
        lines.append("")
        lines.append("Tests whether adding intent keywords (complaint, fee, outage, etc.) to "
                     "the brand query returns a different URL set than the brand-only query, "
                     "and whether different intents return different URL sets from each other "
                     "(i.e. they multiply rather than duplicate).")
        lines.append("")

        baseline = next((r for r in variant_results if r.name == "intent.baseline"), None)
        intent_results = [r for r in variant_results
                          if r.name.startswith("intent.") and r.name != "intent.baseline"]

        all_intent_urls: set[str] = set()
        cumulative: list[tuple[str, int, int, int]] = []
        for r in intent_results:
            new_vs_prior = r.urls_returned - all_intent_urls
            new_vs_baseline = r.urls_returned - (baseline.urls_returned if baseline else set())
            all_intent_urls.update(r.urls_returned)
            label = r.name.replace("intent.", "")
            cumulative.append((label, r.raw_count, len(new_vs_baseline), len(new_vs_prior),
                               len(all_intent_urls)))

        lines.append("| Intent | Raw | New vs baseline (brand-only) | New vs prior intents | Cumulative |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, raw, new_v_base, new_v_prior, cum in cumulative:
            lines.append(f"| {label} | {raw} | {new_v_base} | {new_v_prior} | {cum} |")
        lines.append("")

        if baseline:
            lines.append(f"**Brand-only baseline:** {baseline.raw_count} URLs (1 credit)")
            lines.append(f"**8 intents aggregate:** {len(all_intent_urls)} URLs (8 credits)")
            shared_with_baseline = len(all_intent_urls & baseline.urls_returned)
            lines.append(f"**Lift:** {len(all_intent_urls) / max(1, baseline.raw_count):.1f}× "
                         f"more URLs from intent queries")
            lines.append(f"**Overlap with baseline:** {shared_with_baseline} URLs are in both")
            lines.append("")

        lines.append("**Reading this:** If 'New vs baseline' is high (most of an intent query's "
                     "results don't appear in the brand-only top-10), intent queries reorder "
                     "Google's ranking effectively — they ARE asking different questions. If "
                     "'New vs prior intents' stays high across the table, intents return distinct "
                     "URL sets from each other (they multiply). If it drops to near-zero quickly, "
                     "intents collapse to a few overlapping result sets and the 'lift' is "
                     "smaller than the cumulative count suggests.")
        lines.append("")

        lines.append("**Qualitative check:** Skim the per-variant sample titles below. If intent "
                     "queries surface customer-voice content (Reddit posts, complaint forums, "
                     "Trustpilot/BBB pages, blog rants), intent works as a customer-voice filter. "
                     "If they surface SEO comparison sites and listicles, the 'customer-voice' "
                     "claim is overstated.")
        lines.append("")

    # ====== Intent × Day probe section (only when --intent-x-day) ======
    if getattr(args, "intent_x_day", False):
        lines.append("## Intent × Day Stacking — Does the Matrix Multiply?")
        lines.append("")
        lines.append("Tests whether intent and day-range stack multiplicatively. 4 intents × 4 "
                     "days = 16 queries. Each cell shows raw result count; the row/column "
                     "totals show the unique URLs each intent and each day contribute.")
        lines.append("")

        # Parse names: ixd.{intent}.{date}
        cells: dict[tuple[str, str], ProbeResult] = {}
        intents_seen: list[str] = []
        days_seen: list[str] = []
        for r in variant_results:
            if not r.name.startswith("ixd."):
                continue
            parts = r.name.split(".", 2)  # ixd, intent, date
            intent = parts[1]
            day = parts[2]
            cells[(intent, day)] = r
            if intent not in intents_seen:
                intents_seen.append(intent)
            if day not in days_seen:
                days_seen.append(day)

        # Raw count grid
        lines.append("### Raw count grid")
        lines.append("")
        header = "| Intent | " + " | ".join(days_seen) + " | Row total (unique URLs) |"
        sep = "|---|" + "---:|" * (len(days_seen) + 1)
        lines.append(header)
        lines.append(sep)
        for intent in intents_seen:
            row_urls: set[str] = set()
            row_cells = []
            for day in days_seen:
                cell = cells.get((intent, day))
                if cell:
                    row_urls.update(cell.urls_returned)
                    row_cells.append(str(cell.raw_count))
                else:
                    row_cells.append("-")
            lines.append(f"| {intent} | " + " | ".join(row_cells) + f" | {len(row_urls)} |")

        # Column totals
        col_totals = []
        for day in days_seen:
            day_urls: set[str] = set()
            for intent in intents_seen:
                cell = cells.get((intent, day))
                if cell:
                    day_urls.update(cell.urls_returned)
            col_totals.append(str(len(day_urls)))
        lines.append("| **Col total (unique URLs)** | " + " | ".join(col_totals) + " | |")
        lines.append("")

        # Overall stats
        all_cells_urls = set().union(*(c.urls_returned for c in cells.values())) if cells else set()
        total_raw = sum(c.raw_count for c in cells.values())

        lines.append(f"**Total raw results:** {total_raw} (across {len(cells)} cells)")
        lines.append(f"**Total unique URLs:** {len(all_cells_urls)}")
        if total_raw > 0:
            lines.append(f"**Dedup ratio:** "
                         f"{(1 - len(all_cells_urls) / total_raw) * 100:.0f}% of raw results "
                         f"are duplicates across cells")
        lines.append("")

        lines.append("**Reading this:** The matrix multiplies if Total Unique ≈ Total Raw "
                     "(every cell returns mostly distinct URLs). It collapses if Total Unique "
                     "is much smaller than Total Raw (cells overlap heavily). Look at row "
                     "totals: if 'complaint' across 4 days has ≥30 unique URLs, day-range "
                     "works for that intent. Look at column totals: if a single day across 4 "
                     "intents has ≥30 unique URLs, intent×day works for that day.")
        lines.append("")
        lines.append("If the matrix multiplies cleanly, the production design is intent × day "
                     "across customer-voice templates. If it collapses, we pick whichever "
                     "single dimension gives the bigger lift on its own.")
        lines.append("")

    # ====== Day-range probe section (only when --dayrange) ======
    if args.dayrange:
        lines.append("## Day-Range Querying — Aggregating Day-Specific Results")
        lines.append("")
        lines.append("Tests whether running the same query against 7 individual past days "
                     "(via Google's `cdr:1,cd_min,cd_max` parameter) returns more unique URLs "
                     "than one weekly query.")
        lines.append("")

        weekly_baseline = next((r for r in variant_results
                                if r.name == "dayrange.weekly_baseline"), None)
        day_results = [r for r in variant_results
                       if r.name.startswith("dayrange.") and r.name != "dayrange.weekly_baseline"]

        all_day_urls: set[str] = set()
        cumulative: list[tuple[str, int, int, int]] = []  # (date, raw, new, cum)
        for r in day_results:
            new = r.urls_returned - all_day_urls
            all_day_urls.update(r.urls_returned)
            date_label = r.name.replace("dayrange.", "")
            cumulative.append((date_label, r.raw_count, len(new), len(all_day_urls)))

        lines.append("| Date | Raw | New (vs prior days) | Cumulative unique |")
        lines.append("|---|---:|---:|---:|")
        for date_label, raw, new, cum in cumulative:
            lines.append(f"| {date_label} | {raw} | {new} | {cum} |")
        lines.append("")

        if weekly_baseline:
            weekly_count = len(weekly_baseline.urls_returned)
            day_count = len(all_day_urls)
            shared = len(weekly_baseline.urls_returned & all_day_urls)
            unique_to_weekly = len(weekly_baseline.urls_returned - all_day_urls)
            unique_to_dayrange = len(all_day_urls - weekly_baseline.urls_returned)

            lines.append(f"**Weekly baseline:** {weekly_count} URLs (1 credit)")
            lines.append(f"**Day-range aggregate:** {day_count} URLs ({len(day_results)} credits)")
            lines.append(f"**Lift:** {day_count / max(1, weekly_count):.1f}× more URLs from "
                         f"day-by-day querying")
            lines.append("")
            lines.append(f"**Overlap analysis:**")
            lines.append(f"- URLs in both: {shared}")
            lines.append(f"- URLs only in weekly: {unique_to_weekly}")
            lines.append(f"- URLs only in day-range: {unique_to_dayrange}")
            lines.append("")
            lines.append("**Reading this:** If day-range aggregate is significantly larger than "
                         "weekly, day-by-day querying genuinely surfaces URLs Google ranked too "
                         "low to appear in the weekly top-10. Cost is ~7× higher per template, "
                         "but corpus depth is the bottleneck this whole exercise is trying to "
                         "address.")
            lines.append("")
            if unique_to_weekly > 0:
                lines.append(f"**Note:** {unique_to_weekly} URLs appeared in the weekly query but "
                             f"NOT in any day-specific query. These are typically pages with "
                             f"ambiguous publication dates (rolling content, undated posts) "
                             f"that Google's weekly window catches but cdr:1 doesn't. Worth "
                             f"keeping the weekly query alongside day-range to cover both.")
                lines.append("")

    # ====== Pagination probe section (only when --paginate) ======
    if args.paginate:
        lines.append("## Pagination — Does page 2/3 surface new URLs?")
        lines.append("")
        lines.append("Tests whether Google has results past page 1 for these queries, "
                     "and whether subsequent pages contain genuinely new URLs.")
        lines.append("")
        lines.append("| Template | Page 1 raw | Page 2 raw | Page 3 raw | P2 NEW vs P1 | P3 NEW vs P1+P2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        # Group by template
        by_template: dict[str, dict[int, ProbeResult]] = {}
        for r in variant_results:
            # name is like "page1.general"
            parts = r.name.split(".")
            page_num = int(parts[0].replace("page", ""))
            tmpl = parts[1]
            by_template.setdefault(tmpl, {})[page_num] = r

        for tmpl, pages in by_template.items():
            p1 = pages.get(1)
            p2 = pages.get(2)
            p3 = pages.get(3)
            if not (p1 and p2 and p3):
                continue
            p2_new = len(p2.urls_returned - p1.urls_returned)
            p3_new = len(p3.urls_returned - p1.urls_returned - p2.urls_returned)
            lines.append(f"| {tmpl} | {p1.raw_count} | {p2.raw_count} | {p3.raw_count} | "
                         f"{p2_new} | {p3_new} |")
        lines.append("")
        lines.append("**Reading this:** If P2 NEW = 10 (or close), pagination works — "
                     "Google has a deeper index than Serper's per-page cap. If P2 NEW ≈ 0, "
                     "Google is returning the same top-10 regardless of `page`. P3 NEW ≈ 0 "
                     "would mean Google's index doesn't go that deep for these queries.")
        lines.append("")

    # ====== Daily-vs-weekly probe section (only when --daily) ======
    if args.daily:
        lines.append("## Daily vs Weekly — Time-Window Overlap")
        lines.append("")
        lines.append("Tests whether `qdr:d` (past 24 hours) returns a different URL set than "
                     "`qdr:w` (past 7 days). If daily and weekly return mostly the same URLs, "
                     "running daily across the week buys nothing. If they're substantially "
                     "different, daily aggregation would compound across 7 runs.")
        lines.append("")
        lines.append("| Template | Daily raw | Weekly raw | URLs in BOTH | URLs only in daily | URLs only in weekly |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        by_template: dict[str, dict[str, ProbeResult]] = {}
        for r in variant_results:
            parts = r.name.split(".")
            window = parts[0]  # "daily" or "weekly"
            tmpl = parts[1]
            by_template.setdefault(tmpl, {})[window] = r

        for tmpl, windows in by_template.items():
            d = windows.get("daily")
            w = windows.get("weekly")
            if not (d and w):
                continue
            both = d.urls_returned & w.urls_returned
            only_d = d.urls_returned - w.urls_returned
            only_w = w.urls_returned - d.urls_returned
            lines.append(f"| {tmpl} | {d.raw_count} | {w.raw_count} | "
                         f"{len(both)} | {len(only_d)} | {len(only_w)} |")
        lines.append("")
        lines.append("**Reading this:** Large 'only in daily' values are good — they mean "
                     "running daily would compound new URLs each day. The expected pattern: "
                     "daily returns the most recent ~10 URLs, weekly returns the most "
                     "relevant-by-Google's-ranking ~10 URLs (which are usually older but "
                     "more substantive). If 'URLs in BOTH' dominates, daily and weekly are "
                     "asking effectively the same question — daily aggregation won't help.")
        lines.append("")

    # ====== Weekly-vs-monthly pair comparison (only when --monthly) ======
    if args.monthly:
        lines.append("## Weekly vs Monthly — Volume Lift")
        lines.append("")
        lines.append("Same template, two time windows (`qdr:w` vs `qdr:m`). "
                     "The monthly column reports raw count and the count of NEW URLs not "
                     "already in the weekly result for that template. The lift factor "
                     "tells you how much going monthly multiplies your effective corpus.")
        lines.append("")
        lines.append("| Template | Weekly raw | Monthly raw | Monthly NEW | Lift (raw) | Lift (new) |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        # Pair them by template name
        weekly_by_template = {
            r.name.replace("weekly.", ""): r
            for r in variant_results if r.name.startswith("weekly.")
        }
        monthly_by_template = {
            r.name.replace("monthly.", ""): r
            for r in variant_results if r.name.startswith("monthly.")
        }

        total_weekly_raw = 0
        total_monthly_raw = 0
        total_monthly_new = 0
        all_weekly_urls: set[str] = set()
        all_monthly_urls: set[str] = set()

        for tmpl in weekly_by_template:
            w_r = weekly_by_template.get(tmpl)
            m_r = monthly_by_template.get(tmpl)
            if not w_r or not m_r:
                continue
            new_urls = m_r.urls_returned - w_r.urls_returned
            lift_raw = m_r.raw_count / max(1, w_r.raw_count)
            lift_new = (m_r.raw_count + len(new_urls)) / max(1, w_r.raw_count)  # rough
            actual_lift_new_urls = len(new_urls) / max(1, w_r.raw_count)
            lines.append(f"| {tmpl} | {w_r.raw_count} | {m_r.raw_count} | "
                         f"{len(new_urls)} | "
                         f"{lift_raw:.1f}× | {actual_lift_new_urls:.1f}× |")
            total_weekly_raw += w_r.raw_count
            total_monthly_raw += m_r.raw_count
            total_monthly_new += len(new_urls)
            all_weekly_urls.update(w_r.urls_returned)
            all_monthly_urls.update(m_r.urls_returned)

        lines.append(f"| **TOTAL** | **{total_weekly_raw}** | **{total_monthly_raw}** | "
                     f"**{total_monthly_new}** | "
                     f"**{total_monthly_raw/max(1,total_weekly_raw):.1f}×** | "
                     f"**{total_monthly_new/max(1,total_weekly_raw):.1f}×** |")
        lines.append("")
        lines.append(f"**Unique URLs across all templates:** weekly {len(all_weekly_urls)}, "
                     f"monthly {len(all_monthly_urls)}, "
                     f"net new URLs from going monthly: {len(all_monthly_urls - all_weekly_urls)}.")
        lines.append("")
        lines.append("**Reading this:** Lift of 1.0× means monthly returns the same count "
                     "as weekly (Google's `qdr:m` index is no deeper than `qdr:w` for that "
                     "template). Lift > 2.0× on raw means monthly genuinely surfaces more. "
                     "Lift on *new URLs* is the more meaningful number — if monthly's 30 "
                     "results are mostly the same URLs weekly already returned, the "
                     "effective lift is much smaller than the raw count suggests.")
        lines.append("")

    # ====== Variant analysis ======
    lines.append("## Variant probes")
    lines.append("")

    variant_credits = sum(r.credit_cost for r in variant_results)
    new_urls_total = set().union(*(r.new_urls_vs_baseline for r in variant_results)) \
        if baseline_urls else set().union(*(r.urls_returned for r in variant_results))

    lines.append(f"- Queries run: {len(variant_results)}")
    lines.append(f"- Credits used: {variant_credits}")
    lines.append(f"- Total raw results: {sum(r.raw_count for r in variant_results)}")
    if baseline_urls:
        lines.append(f"- Unique NEW URLs (not in baseline): {len(new_urls_total)}")
    lines.append("")

    lines.append("### Per-variant results")
    lines.append("")
    lines.append("| Variant | Cost | Raw | New vs baseline | Description |")
    lines.append("|---|---:|---:|---:|---|")
    for r in variant_results:
        if r.error:
            new_count_str = "ERR"
        elif baseline_urls:
            new_count_str = str(len(r.new_urls_vs_baseline))
        else:
            new_count_str = "n/a"
        lines.append(f"| {r.name} | {r.credit_cost} | {r.raw_count} | "
                     f"{new_count_str} | {r.description} |")
    lines.append("")

    # ====== Per-variant detail ======
    lines.append("## Variant detail")
    lines.append("")
    for r in variant_results:
        lines.append(f"### {r.name}")
        lines.append(f"_{r.description}_")
        lines.append(f"")
        lines.append(f"- Query: `{r.query}`")
        lines.append(f"- Endpoint: `{r.endpoint}` &middot; "
                     f"num={r.num_results_requested} &middot; tbs={r.time_window}")
        lines.append(f"- Cost: {r.credit_cost} credit(s)")

        if r.error:
            lines.append(f"- **ERROR**: {r.error}")
            lines.append("")
            continue

        lines.append(f"- Raw results: {r.raw_count}")
        if baseline_urls:
            lines.append(f"- New vs baseline: {len(r.new_urls_vs_baseline)} "
                         f"({100*len(r.new_urls_vs_baseline)/max(1,r.raw_count):.0f}% novel)")
        lines.append("")
        lines.append("Sample titles:")
        lines.append("```")
        for s in r.sample_titles:
            lines.append(s)
        lines.append("```")
        lines.append("")

    # ====== Recommendations stub ======
    lines.append("## Suggested reads")
    lines.append("")
    lines.append("Look at each variant's *new vs baseline* count. A variant that "
                 "returns 25 results with 25 new is high-value. A variant that "
                 "returns 30 results with 3 new is mostly redundant with what "
                 "the existing pipeline already gets.")
    lines.append("")
    lines.append("Then look at the *sample titles* — even if the count is high, "
                 "if every title is junk (sidebar listings, error pages, paid "
                 "promotional content), the marginal value is low.")
    lines.append("")
    lines.append("Cost-per-new-URL is the right metric. Variants with a low "
                 "cost-per-new-URL are worth committing to code.")

    out_path.write_text("\n".join(lines))
    print()
    print(f"Report written to {out_path}")
    print(f"Total Serper credits used: "
          f"{sum(r.credit_cost for r in baseline_results) + variant_credits}")


if __name__ == "__main__":
    main()
