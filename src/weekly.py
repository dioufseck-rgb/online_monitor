"""
Weekly orchestrator. Multi-brand, multi-source.

For each brand in config:
  1. Build collectors (Serper + optional Reddit if approved)
  2. Ingest -> Mention records
  3. Filter for relevance + dedup
  4. Persist to store
  5. Enrich high-priority mentions with full-text fetch
  6. Classify any unclassified mentions
  7. Aggregate -> WeeklyReport (per brand)
  8. Render markdown (per brand)

Usage:
  python -m src.weekly                           # all brands, last 7 days
  python -m src.weekly --brand nfcu              # single brand
  python -m src.weekly --dry-run --brand nfcu    # preview, no LLM calls
  python -m src.weekly --days 2                  # narrower window
"""

from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


def _load_dotenv() -> None:
    """Load .env from repo root if present. Existing env vars take precedence
    (so explicit `export` overrides .env)."""
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


_load_dotenv()  # run at import time so config resolution sees the vars


from .aggregate import build_weekly_report
from .classify import classify_batch
from .enrich import WebFetchEnricher
from .filter import filter_pipeline_with_terms
from .ingest import RedditCollector, SerperCollector
from .llm import build_adapter
from .report import finalize_and_render
from .store import Store


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            "Copy config/config.example.yaml to config/config.yaml and fill it in."
        )
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _resolve_env(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def _build_collectors_for_brand(cfg: dict, brand_cfg: dict) -> list:
    collectors = []
    sources = cfg.get("sources", {})
    brand_name = brand_cfg["name"]
    brand_terms = brand_cfg["terms"]
    brand_domain = brand_cfg.get("domain", "financial_services")

    if sources.get("serper", {}).get("enabled"):
        scfg = sources["serper"]
        api_key = _resolve_env(scfg["api_key"])
        if not api_key:
            print("[orchestrator] Serper enabled but no API key resolved; skipping")
        else:
            collectors.append(SerperCollector(
                api_key=api_key,
                brand=brand_name,
                brand_terms=brand_terms,
                domain=brand_domain,
                query_templates=scfg.get("query_templates"),
                num_results=scfg.get("num_results", 10),
                time_window=scfg.get("time_window", "qdr:w"),
                rate_limit_sleep=scfg.get("rate_limit_sleep", 0.1),
                intent_queries=scfg.get("intent_queries", False),
                dayrange=scfg.get("dayrange", False),
                dayrange_days=scfg.get("dayrange_days", 7),
            ))

    if sources.get("reddit", {}).get("enabled"):
        rcfg = sources["reddit"]
        collectors.append(RedditCollector(
            client_id=_resolve_env(rcfg["client_id"]),
            client_secret=_resolve_env(rcfg["client_secret"]),
            user_agent=rcfg.get("user_agent", "voc-monitor/0.1"),
            brand=brand_name,
            username=_resolve_env(rcfg.get("username")) if rcfg.get("username") else None,
            password=_resolve_env(rcfg.get("password")) if rcfg.get("password") else None,
            max_per_query=rcfg.get("max_per_query", 250),
        ))

    return collectors


def _build_llm_adapter(cfg: dict):
    pcfg = cfg["provider"]
    name = pcfg["name"]
    if name == "gemini":
        return build_adapter("gemini",
                             model=pcfg.get("model", "gemini-2.0-flash"),
                             api_key=_resolve_env(pcfg.get("api_key")) or None)
    if name == "azure_openai":
        return build_adapter("azure_openai",
                             deployment=pcfg["deployment"],
                             endpoint=_resolve_env(pcfg.get("endpoint")) or None,
                             api_key=_resolve_env(pcfg.get("api_key")) or None,
                             api_version=pcfg.get("api_version", "2024-10-21"))
    if name == "anthropic":
        return build_adapter("anthropic",
                             model=pcfg.get("model", "claude-haiku-4-5-20251001"),
                             api_key=_resolve_env(pcfg.get("api_key")) or None)
    raise ValueError(f"Unknown provider: {name}")


def run_for_brand(
    cfg: dict,
    brand_cfg: dict,
    since: datetime,
    until: datetime,
    dry_run: bool = False,
    max_classify: int | None = None,
) -> Path | None:
    brand = brand_cfg["name"]
    brand_terms = brand_cfg["terms"]
    domain_name = brand_cfg.get("domain", "financial_services")

    print(f"\n{'='*70}")
    print(f"[brand={brand}] Window: {since.isoformat()} -> {until.isoformat()}")
    print(f"{'='*70}")

    if dry_run:
        print(f"[brand={brand}] DRY RUN — ingest + filter + preview, skip classification")

    store = Store(cfg.get("store_path", "data/mentions.db"))
    collectors = _build_collectors_for_brand(cfg, brand_cfg)
    if not collectors:
        print(f"[brand={brand}] No collectors enabled. Skipping.")
        store.close()
        return None

    raw_mentions = []
    for c in collectors:
        print(f"[brand={brand}] Collecting from {c.source_name}...")
        before = len(raw_mentions)
        for m in c.collect(since, until):
            raw_mentions.append(m)
        print(f"[brand={brand}]   +{len(raw_mentions) - before} mentions")

    filtered = filter_pipeline_with_terms(raw_mentions, brand_terms)
    print(f"[brand={brand}] {len(filtered)} after filter (from {len(raw_mentions)} raw)")

    inserted = store.upsert_mentions(filtered)
    print(f"[brand={brand}] {inserted} mentions persisted")

    window_mentions = store.get_mentions_in_window(since, until, brand=brand)
    unclassified_ids = store.get_unclassified_mention_ids([m.id for m in window_mentions])
    to_classify = [m for m in window_mentions if m.id in set(unclassified_ids)]

    print(f"\n[brand={brand}] Volume preview:")
    print(f"  Mentions in window: {len(window_mentions)}")
    print(f"  Already classified: {len(window_mentions) - len(to_classify)}")
    print(f"  To classify:        {len(to_classify)}")
    print(f"  LLM calls expected: {len(to_classify) * 3} + 1 exec summary")

    by_source = {}
    for m in window_mentions:
        by_source[m.source.value] = by_source.get(m.source.value, 0) + 1
    print(f"  By source:          {by_source}")

    if dry_run:
        print(f"\n[brand={brand}] Sample mentions (first 5):")
        for m in to_classify[:5]:
            preview = m.text[:140].replace("\n", " ")
            print(f"  - [{m.source.value}] {preview}{'...' if len(m.text) > 140 else ''}")
            print(f"    {m.url}")
        store.close()
        print(f"\n[brand={brand}] Dry run complete.")
        return None

    if max_classify and len(to_classify) > max_classify:
        print(f"[brand={brand}] Capping at {max_classify} mentions (--max-classify)")
        to_classify = to_classify[:max_classify]

    enrich_cfg = cfg.get("enrich", {})
    if enrich_cfg.get("enabled", True) and to_classify:
        enricher = WebFetchEnricher(
            max_fetches_per_run=enrich_cfg.get("max_fetches_per_run", 50),
            per_domain_min_interval=enrich_cfg.get("per_domain_min_interval", 1.0),
            timeout=enrich_cfg.get("timeout", 15),
        )
        print(f"[brand={brand}] Enriching with full-text fetches...")
        enriched = enricher.enrich(to_classify)
        store.upsert_mentions(enriched)
        enriched_by_id = {m.id: m for m in enriched}
        to_classify = [enriched_by_id.get(m.id, m) for m in to_classify]
        fetch_count = sum(1 for m in to_classify if m.full_text_fetched)
        print(f"[brand={brand}]   {fetch_count} mentions enriched with full text")

        # Post-enrichment filter: web_fetch may have replaced the original
        # snippet with an error/maintenance page, or with content that no
        # longer mentions the brand. Re-run the filter on enriched text so
        # we don't waste classification calls on pages that became noise.
        from .filter import is_relevant, is_error_page, is_sidebar_noise
        before = len(to_classify)
        to_classify = [m for m in to_classify
                       if is_relevant(m, brand_terms)
                       and not is_error_page(m)
                       and not is_sidebar_noise(m, brand_terms)]
        dropped = before - len(to_classify)
        if dropped:
            print(f"[brand={brand}]   {dropped} mentions dropped post-enrichment "
                  f"(error pages, sidebar noise, or lost relevance)")

    llm = _build_llm_adapter(cfg)
    print(f"[brand={brand}] Classifying {len(to_classify)} mentions...")
    new_classifications = classify_batch(
        llm, to_classify,
        max_workers=cfg.get("classify_workers", 4),
    )
    store.upsert_classifications(new_classifications)
    print(f"[brand={brand}] {len(new_classifications)} classifications persisted")

    all_classifications = store.get_classifications([m.id for m in window_mentions])
    prior_report = store.get_prior_report(since, brand=brand)
    report = build_weekly_report(
        mentions=window_mentions,
        classifications=all_classifications,
        week_start=since,
        week_end=until,
        prior_report=prior_report,
        domain_name=domain_name,
    )
    print(f"[brand={brand}] Report: {report.total_mentions} mentions, "
          f"{len(report.clusters)} clusters, {len(report.risk_signals)} risk signals")

    report, md = finalize_and_render(llm, report, brand=brand, domain_name=domain_name)
    store.save_report(report, brand=brand)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{brand}_{since.date().isoformat()}.md"
    out_path.write_text(md)
    print(f"[brand={brand}] Report written to {out_path}")

    store.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", help="ISO date for window start")
    parser.add_argument("--until", help="ISO date for window end")
    parser.add_argument("--days", type=int, default=7, help="Window length")
    parser.add_argument("--brand", help="Run for a single brand (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest + filter + preview, skip classification.")
    parser.add_argument("--max-classify", type=int, default=None,
                        help="Cap mentions to classify per brand.")
    args = parser.parse_args()

    if args.until:
        until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
    else:
        until = datetime.now(tz=timezone.utc)

    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    else:
        since = until - timedelta(days=args.days)

    cfg = _load_config()
    brands = cfg.get("brands", [])
    if args.brand:
        brands = [b for b in brands if b["name"] == args.brand]
        if not brands:
            print(f"Brand '{args.brand}' not found in config.")
            return

    for brand_cfg in brands:
        run_for_brand(cfg, brand_cfg, since, until,
                      dry_run=args.dry_run, max_classify=args.max_classify)


if __name__ == "__main__":
    main()
