"""
Generate the HTML dashboard report from data already in the store.

Reads the most-recent WeeklyReport per brand, plus the underlying mentions
and classifications, and emits a single self-contained HTML file with tabs,
search, and progressive disclosure.

Usage:
    python -m src.html_dashboard
    python -m src.html_dashboard --brands nfcu usaa
    python -m src.html_dashboard --week 2026-05-02
    python -m src.html_dashboard --target nfcu        # explicit target framing
"""

from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .html_report import render_html
from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brands", nargs="*", help="Brands to include (default: all)")
    parser.add_argument("--week", help="ISO date for the week start (default: most recent)")
    parser.add_argument("--target", help="Target brand for the comparison panel "
                        "(default: from config target_brand)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--config-path", default="config/config.yaml")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    # Resolve target: --target > config target_brand > None
    resolved_target = args.target
    if not resolved_target:
        config_path = Path(args.config_path)
        if config_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_path.read_text())
                resolved_target = cfg.get("target_brand")
            except Exception as e:
                print(f"[html_dashboard] Could not load config ({e}); proceeding without target")

    store = Store(args.store_path)

    if args.week:
        cutoff = datetime.fromisoformat(args.week).replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        cutoff = datetime.now(tz=timezone.utc) + timedelta(days=1)

    if args.brands:
        brands_to_load = args.brands
    else:
        cur = store._conn.execute("SELECT DISTINCT brand FROM reports")
        brands_to_load = [row[0] for row in cur]

    brand_data = {}
    for brand in brands_to_load:
        report = store.get_prior_report(cutoff, brand=brand)
        if report is None:
            print(f"[html_dashboard] No report for brand={brand}; skipping.")
            continue

        # Pull mentions in the report's window for this brand
        mentions = store.get_mentions_in_window(
            report.week_start, report.week_end, brand=brand
        )
        classifications = store.get_classifications([m.id for m in mentions])

        # Resolve the domain — read from the first mention if available
        brand_domain = mentions[0].domain if mentions else "financial_services"

        brand_data[brand] = {
            "report": report,
            "mentions": mentions,
            "classifications": classifications,
            "domain": brand_domain,
        }

    if not brand_data:
        print("[html_dashboard] No data found. Run `python -m src.weekly` first.")
        return

    if resolved_target and resolved_target not in brand_data:
        print(f"[html_dashboard] Target brand '{resolved_target}' not in available reports; "
              f"comparison panel will be descriptive-only.")
        resolved_target = None

    # Convert WeeklyReport / Mention / Classification objects to JSON-friendly form
    payload = {}
    for brand, data in brand_data.items():
        cls_by_id = {c.mention_id: c for c in data["classifications"]}
        mentions_serialized = []
        for m in data["mentions"]:
            c = cls_by_id.get(m.id)
            if c is None:
                continue
            mentions_serialized.append({
                "id": m.id,
                "source": m.source.value,
                "title": m.title or "",
                "text": m.text,
                "snippet": m.snippet or "",
                "url": m.url,
                "timestamp": m.timestamp.isoformat(),
                "author": m.author_handle,
                "full_text_fetched": m.full_text_fetched,
                "engagement": m.engagement,
                "sentiment": {
                    "label": c.sentiment.label.value,
                    "intensity": c.sentiment.intensity,
                    "rationale": c.sentiment.rationale,
                },
                "topic": {
                    "label": c.topic.label,
                    "confidence": c.topic.confidence,
                    "rationale": c.topic.rationale,
                },
                "validity": {
                    "label": c.validity_claim.label.value,
                    "confidence": c.validity_claim.confidence,
                    "rationale": c.validity_claim.rationale,
                },
                "origin": {
                    "label": c.origin.label.value,
                    "confidence": c.origin.confidence,
                    "rationale": c.origin.rationale,
                },
            })
        payload[brand] = {
            "report": data["report"].model_dump(mode="json"),
            "mentions": mentions_serialized,
            "domain": data["domain"],
        }

    sample = next(iter(payload.values()))
    title = f"Week of {sample['report']['week_start'][:10]}"
    if resolved_target:
        title = f"{resolved_target.upper()} — {title}"

    html_content = render_html(payload, title=title, target=resolved_target)

    week_label = sample['report']['week_start'][:10]
    if not args.output_path and resolved_target:
        # Prefix filename with target brand for clarity
        output_path = Path(__file__).parent.parent / "reports" / \
            f"{resolved_target}_dashboard_{week_label}.html"
    else:
        output_path = Path(args.output_path) if args.output_path else \
            Path(__file__).parent.parent / "reports" / f"dashboard_{week_label}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    file_size_kb = len(html_content) / 1024
    total_mentions = sum(len(d["mentions"]) for d in payload.values())
    print(f"[html_dashboard] Dashboard written to {output_path}")
    print(f"[html_dashboard] {len(payload)} brands, {total_mentions} mentions, "
          f"{file_size_kb:.0f}KB")
    if resolved_target:
        print(f"[html_dashboard] Target: {resolved_target}; "
              f"peers: {[b for b in payload if b != resolved_target]}")
    print(f"[html_dashboard] Open in browser: file://{output_path.resolve()}")

    store.close()


if __name__ == "__main__":
    main()
