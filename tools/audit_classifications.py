"""
Audit script: pull mentions matching given topic filters along with their
classification rationales, so you can read what the classifier did and decide
whether it's right.

Usage:
    python -m tools.audit_classifications --brand nfcu --topics fraud_claim product_complaint
    python -m tools.audit_classifications --brand nfcu --topics outage_or_service_issue --since 2026-05-02
    python -m tools.audit_classifications --brand nfcu --validity comprehensibility  # debug the COMPREHENSIBILITY leak

Outputs to stdout. Pipe to a file with > audit.txt if you want to share.
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Path hack so this works from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", required=True, help="Brand name (e.g. 'nfcu')")
    parser.add_argument("--topics", nargs="*", default=[],
                        help="Topic labels to filter (e.g. fraud_claim product_complaint)")
    parser.add_argument("--validity", nargs="*", default=[],
                        help="Validity-claim labels to filter (e.g. comprehensibility)")
    parser.add_argument("--sentiment", nargs="*", default=[],
                        help="Sentiment labels to filter")
    parser.add_argument("--origin", nargs="*", default=[],
                        help="Origin labels to filter (e.g. customer brand_owned employee_personal)")
    parser.add_argument("--since", help="ISO date for window start (default: 14 days ago)")
    parser.add_argument("--until", help="ISO date for window end (default: now)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc) if args.until \
        else datetime.now(tz=timezone.utc)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since \
        else until - timedelta(days=14)

    store = Store(args.store_path)
    mentions = store.get_mentions_in_window(since, until, brand=args.brand)
    classifications = store.get_classifications([m.id for m in mentions])
    cls_by_id = {c.mention_id: c for c in classifications}

    # Filter
    matches = []
    for m in mentions:
        c = cls_by_id.get(m.id)
        if c is None:
            continue
        if args.topics and c.topic.label not in args.topics:
            continue
        if args.validity and c.validity_claim.label.value not in args.validity:
            continue
        if args.sentiment and c.sentiment.label.value not in args.sentiment:
            continue
        if args.origin and c.origin.label.value not in args.origin:
            continue
        matches.append((m, c))

    print(f"Window: {since.date()} -> {until.date()}")
    print(f"Brand: {args.brand}")
    print(f"Filters: topics={args.topics or 'any'} validity={args.validity or 'any'} "
          f"sentiment={args.sentiment or 'any'} origin={args.origin or 'any'}")
    print(f"Matches: {len(matches)} (showing up to {args.limit})")
    print("=" * 80)

    for i, (m, c) in enumerate(matches[:args.limit], 1):
        print(f"\n[{i}] {m.source.value} | {m.timestamp.date()} | {m.url}")
        print(f"    Title: {m.title or '(none)'}")
        text_preview = m.text[:400].replace("\n", " ")
        print(f"    Text:  {text_preview}{'...' if len(m.text) > 400 else ''}")
        print(f"    Full-text fetched: {m.full_text_fetched}")
        print(f"    SENTIMENT: {c.sentiment.label.value} (intensity {c.sentiment.intensity:.2f})")
        print(f"      rationale: {c.sentiment.rationale}")
        print(f"    TOPIC:     {c.topic.label} (confidence {c.topic.confidence:.2f})")
        print(f"      rationale: {c.topic.rationale}")
        print(f"    VALIDITY:  {c.validity_claim.label.value} (confidence {c.validity_claim.confidence:.2f})")
        print(f"      rationale: {c.validity_claim.rationale}")
        print(f"    ORIGIN:    {c.origin.label.value} (confidence {c.origin.confidence:.2f})")
        print(f"      rationale: {c.origin.rationale}")
        print("-" * 80)

    store.close()


if __name__ == "__main__":
    main()
