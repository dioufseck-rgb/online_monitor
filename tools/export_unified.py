"""
Export VoC monitoring data for the unified weekly briefing HTML report.

Reads the most recent WeeklyReport per brand from the SQLite store, computes
perception dimensions, and emits a single JSON file. Includes full mention
text for every mention referenced by an actionable cluster, so the HTML
report can show expand-to-full-text behavior.

Usage:
    python -m tools.export_unified --target nfcu \\
        --brands nfcu bofa chase usaa penfed capitalone \\
        --output-path unified_export.json

Outputs unified_export.json in the current directory by default.
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as either `python tools/export_unified.py` or `python -m tools.export_unified`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compare import compute_perception_dimensions, cohort_median_dimensions
from src.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target brand for the report")
    parser.add_argument("--brands", nargs="*", required=True,
                        help="Cohort brands (target must be in this list)")
    parser.add_argument("--week", help="ISO date for week start (default: most recent)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--output-path", default="unified_export.json")
    args = parser.parse_args()

    if args.target not in args.brands:
        print(f"[export] --target '{args.target}' must be in --brands list; "
              f"got {args.brands}")
        sys.exit(1)

    store = Store(args.store_path)

    if args.week:
        cutoff = datetime.fromisoformat(args.week).replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        cutoff = datetime.now(tz=timezone.utc) + timedelta(days=1)

    reports = {}
    mentions_by_id = {}

    for brand in args.brands:
        r = store.get_prior_report(cutoff, brand=brand)
        if r is None:
            print(f"[export] No report for brand={brand}; skipping.")
            continue
        reports[brand] = r

        # Collect the set of mention IDs referenced by any cluster in this report
        cluster_mention_ids = set()
        for c in r.clusters:
            cluster_mention_ids.update(c.mention_ids)

        if not cluster_mention_ids:
            continue

        # Pull the actual mention rows from the store
        mentions = store.get_mentions_in_window(
            r.week_start, r.week_end, brand=brand,
        )
        for m in mentions:
            if m.id in cluster_mention_ids:
                # Defensive: source may be enum or string depending on how loaded
                src = m.source.value if hasattr(m.source, "value") else str(m.source)
                mentions_by_id[m.id] = {
                    "id": m.id,
                    "source": src,
                    "title": m.title or "",
                    "text": m.text,
                    "snippet": m.snippet or "",
                    "url": m.url,
                    "author": m.author_handle,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "engagement": m.engagement,
                }

    if not reports:
        print("[export] No reports found. Run `python -m src.weekly` first.")
        sys.exit(1)

    if args.target not in reports:
        print(f"[export] No report for target '{args.target}'. "
              f"Loaded: {list(reports.keys())}")
        sys.exit(1)

    if len(reports) < 2:
        print(f"[export] Only one brand loaded ({list(reports.keys())}); "
              f"cohort comparison requires at least 2.")
        sys.exit(1)

    perceptions = compute_perception_dimensions(reports)
    medians = cohort_median_dimensions(perceptions)

    export = {
        "target": args.target,
        "cohort": list(reports.keys()),
        "week_start": next(iter(reports.values())).week_start.isoformat(),
        "week_end": next(iter(reports.values())).week_end.isoformat(),
        "reports": {b: r.model_dump(mode="json") for b, r in reports.items()},
        "perceptions": {b: [asdict(d) for d in perceptions[b]] for b in perceptions},
        "medians": medians,
        "mentions_by_id": mentions_by_id,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export, indent=2, default=str))

    size_kb = output_path.stat().st_size / 1024
    print(f"[export] Written: {output_path} ({size_kb:.1f} KB)")
    print(f"[export] Cohort: {list(reports.keys())}")
    print(f"[export] Target: {args.target}")
    print(f"[export] Mentions with full text: {len(mentions_by_id)}")

    store.close()


if __name__ == "__main__":
    main()