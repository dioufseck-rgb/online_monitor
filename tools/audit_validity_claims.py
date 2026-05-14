"""
Validity-claim audit: stratified sampling across validity-claim labels per brand.

Use this to verify that the Habermasian classifier is actually distinguishing
fact / norm / experience / meaning claims correctly. The week's per-brand
distribution can look right at the aggregate level even when individual
classifications are wrong — this tool lets you read N mentions per
(brand, label) cell and decide for yourself.

Usage:
    # Default: 5 mentions per validity-claim label per brand
    python -m tools.audit_validity_claims

    # More samples per cell
    python -m tools.audit_validity_claims --n 10

    # Single brand
    python -m tools.audit_validity_claims --brands nfcu

    # Specific labels
    python -m tools.audit_validity_claims --labels truth sincerity --n 8

    # Write to file for review
    python -m tools.audit_validity_claims > audit.txt
"""

from __future__ import annotations
import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import ValidityClaim
from src.store import Store


HABERMAS_PARAPHRASE = {
    "truth": "fact claim",
    "rightness": "norm claim",
    "sincerity": "experience claim",
    "comprehensibility": "meaning claim",
}


def _truncate(text: str, max_len: int = 400) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brands", nargs="*", help="Brands to audit (default: all)")
    parser.add_argument("--labels", nargs="*",
                        default=[v.value for v in ValidityClaim],
                        help="Validity-claim labels to sample from")
    parser.add_argument("--n", type=int, default=5,
                        help="Mentions per (brand, label) cell (default: 5)")
    parser.add_argument("--since", help="ISO date for window start (default: 14 days ago)")
    parser.add_argument("--until", help="ISO date for window end (default: now)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc) if args.until \
        else datetime.now(tz=timezone.utc)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since \
        else until - timedelta(days=14)

    store = Store(args.store_path)

    if args.brands:
        brands = args.brands
    else:
        cur = store._conn.execute("SELECT DISTINCT brand FROM mentions")
        brands = [row[0] for row in cur]

    print(f"Validity-Claim Audit")
    print(f"Window: {since.date()} -> {until.date()}")
    print(f"Brands: {brands}")
    print(f"Labels: {args.labels}")
    print(f"Samples per cell: {args.n}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    for brand in brands:
        mentions = store.get_mentions_in_window(since, until, brand=brand)
        if not mentions:
            print(f"\n[brand={brand}] No mentions in window.\n")
            continue

        classifications = store.get_classifications([m.id for m in mentions])
        cls_by_id = {c.mention_id: c for c in classifications}

        # Bucket by validity-claim label
        buckets: dict[str, list] = {label: [] for label in args.labels}
        for m in mentions:
            c = cls_by_id.get(m.id)
            if c is None:
                continue
            label = c.validity_claim.label.value
            if label in buckets:
                buckets[label].append((m, c))

        print(f"\n{'#' * 80}")
        print(f"# BRAND: {brand}")
        print(f"# Bucket sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
        print(f"{'#' * 80}")

        for label, items in buckets.items():
            paraphrase = HABERMAS_PARAPHRASE.get(label, label)
            print(f"\n----- {label.upper()} ({paraphrase}) — population {len(items)}, "
                  f"sampling {min(args.n, len(items))} -----")
            if not items:
                print("  (no mentions in this bucket)")
                continue

            sample = rng.sample(items, min(args.n, len(items)))
            for i, (m, c) in enumerate(sample, 1):
                print(f"\n  [{i}] {m.source.value} | {m.timestamp.date()} | {m.url}")
                print(f"      Title: {m.title or '(none)'}")
                print(f"      Text:  {_truncate(m.text)}")
                print(f"      VALIDITY:  {c.validity_claim.label.value} "
                      f"(conf {c.validity_claim.confidence:.2f})")
                print(f"        rationale: {c.validity_claim.rationale}")
                print(f"      Topic: {c.topic.label} ({c.topic.confidence:.2f}) | "
                      f"Sentiment: {c.sentiment.label.value} ({c.sentiment.intensity:.2f}) | "
                      f"Origin: {c.origin.label.value}")

    print("\n" + "=" * 80)
    print("Audit complete. Read each entry and decide if the validity-claim label fits.")
    print("If a substantial fraction look wrong, the prompt needs another iteration.")
    store.close()


if __name__ == "__main__":
    main()
