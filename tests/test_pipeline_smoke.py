"""
Smoke test: end-to-end pipeline using synthetic mentions and a mock LLM.

Run from the repo root:
    python -m tests.test_pipeline_smoke

This validates the full data flow without hitting any real API.
"""

from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Type, TypeVar

# Path hack so this runs from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from src.aggregate import build_weekly_report
from src.classify import classify_batch
from src.filter import filter_pipeline
from src.llm import LLMAdapter
from src.report import finalize_and_render
from src.schema import (
    Mention,
    Sentiment,
    Source,
    Topic,
    ValidityClaim,
)


T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMAdapter):
    """Deterministic LLM stub. Returns canned but plausible classifications
    based on keywords in the input text."""

    def _classify_structured_once(self, system, user, response_model, temperature=0.0):
        from src.schema import (
            SentimentResult, TopicResult, ValidityClaimResult, OriginResult, MentionOrigin,
        )
        from src.report import ExecutiveSummary as _ExecSummary

        text = user.lower()

        if response_model is SentimentResult:
            if "fraud" in text or "scam" in text or "stolen" in text:
                return SentimentResult(label=Sentiment.VERY_NEGATIVE, intensity=0.9,
                                       rationale="fraud reference")
            if "love" in text or "great" in text or "amazing" in text:
                return SentimentResult(label=Sentiment.POSITIVE, intensity=0.7,
                                       rationale="positive expression")
            if "hate" in text or "frustrated" in text or "angry" in text:
                return SentimentResult(label=Sentiment.NEGATIVE, intensity=0.7,
                                       rationale="negative emotion")
            return SentimentResult(label=Sentiment.NEUTRAL, intensity=0.3,
                                   rationale="default neutral")

        if response_model is TopicResult:
            if "fraud" in text or "scam" in text or "stolen" in text:
                return TopicResult(label=Topic.FRAUD_CLAIM, confidence=0.9,
                                   rationale="fraud language")
            if "app" in text and ("down" in text or "broken" in text or "can't" in text):
                return TopicResult(label=Topic.OUTAGE_OR_SERVICE_ISSUE, confidence=0.85,
                                   rationale="app issue")
            if "fee" in text or "overdraft" in text or "$35" in text:
                return TopicResult(label=Topic.FEE_DISPUTE, confidence=0.85,
                                   rationale="fee mention")
            if "love" in text or "great" in text:
                return TopicResult(label=Topic.PRAISE, confidence=0.8,
                                   rationale="praise language")
            if "usaa" in text or "penfed" in text:
                return TopicResult(label=Topic.COMPETITOR_COMPARISON, confidence=0.85,
                                   rationale="competitor named")
            return TopicResult(label=Topic.GENERIC_DISCUSSION, confidence=0.5,
                               rationale="default")

        if response_model is ValidityClaimResult:
            if "$" in user or "charged" in text or "down at" in text:
                return ValidityClaimResult(label=ValidityClaim.TRUTH, confidence=0.8,
                                           rationale="factual claim")
            if "should" in text or "shouldn't" in text or "ought" in text:
                return ValidityClaimResult(label=ValidityClaim.RIGHTNESS, confidence=0.8,
                                           rationale="normative claim")
            if "love" in text or "feel" in text or "frustrated" in text:
                return ValidityClaimResult(label=ValidityClaim.SINCERITY, confidence=0.8,
                                           rationale="personal expression")
            return ValidityClaimResult(label=ValidityClaim.SINCERITY, confidence=0.5,
                                       rationale="default")

        if response_model is _ExecSummary:
            return _ExecSummary(bullets=[
                "Synthetic test run — exec summary stub.",
                "Volume and sentiment numbers match structured input.",
                "No real risk routing applies in this test.",
            ])

        if response_model is OriginResult:
            # Default to CUSTOMER for the synthetic test corpus
            return OriginResult(
                label=MentionOrigin.CUSTOMER,
                confidence=0.7,
                rationale="Synthetic test mention; default CUSTOMER classification.",
            )

        raise NotImplementedError(f"MockLLM doesn't handle {response_model}")


def _synthetic_mentions() -> list[Mention]:
    base = datetime.now(tz=timezone.utc) - timedelta(days=3)
    samples = [
        ("NFCU charged me a $35 overdraft fee even though I had a pending deposit. Frustrated.",
         "user_a", 12, 4),
        ("Anyone else having issues with the Navy Federal app today? Can't log in.",
         "user_b", 25, 18),
        ("Navy Federal app is down again. Third time this month.",
         "user_c", 40, 22),
        ("Just had a great call with my Navy Federal rep — she fixed everything.",
         "user_d", 8, 1),
        ("I love NFCU honestly. Best decision I made.", "user_e", 15, 2),
        ("My account got compromised and Navy Federal took 5 days to refund. Scam-level service.",
         "user_f", 55, 31),
        ("NFCU shouldn't charge fees on accounts that are clearly fraud victims.",
         "user_g", 33, 12),
        ("USAA vs Navy Federal — which is better for VA loans?",
         "user_h", 18, 27),
        ("My nfcu credit card got skimmed. They handled it but the fraud team was slow.",
         "user_i", 22, 7),
        ("Navy Federal app crashed during my transfer. Money disappeared for 2 hours.",
         "user_j", 30, 14),
        ("nfcu fraud department is a nightmare to deal with",
         "user_k", 19, 9),
        ("Random military discussion not really about anything specific",  # should be filtered
         "user_l", 5, 1),
    ]
    out = []
    for i, (text, author, score, comments) in enumerate(samples):
        out.append(Mention(
            id=f"reddit:test_{i}",
            source=Source.REDDIT,
            brand="nfcu",
            author_handle=author,
            timestamp=base + timedelta(hours=i * 4),
            text=text,
            url=f"https://reddit.com/test/{i}",
            engagement={"score": score, "num_comments": comments},
            raw_payload={"subreddit": "personalfinance"},
        ))
    return out


def main():
    print("=" * 60)
    print("NFCU Monitor — end-to-end smoke test")
    print("=" * 60)

    mentions = _synthetic_mentions()
    print(f"\n[1] Generated {len(mentions)} synthetic mentions")

    filtered = filter_pipeline(mentions)
    print(f"[2] After filter: {len(filtered)} (dropped {len(mentions) - len(filtered)})")

    llm = MockLLM()
    classifications = classify_batch(llm, filtered, max_workers=2)
    print(f"[3] Classified {len(classifications)} mentions")

    until = datetime.now(tz=timezone.utc)
    since = until - timedelta(days=7)

    report = build_weekly_report(
        mentions=filtered,
        classifications=classifications,
        week_start=since,
        week_end=until,
        prior_report=None,
    )
    print(f"[4] Aggregated report:")
    print(f"    total_mentions={report.total_mentions}")
    print(f"    clusters={len(report.clusters)}")
    print(f"    risk_signals={len(report.risk_signals)}")
    for sig in report.risk_signals:
        print(f"      - {sig.kind}: {sig.recommended_owner}")

    report, md = finalize_and_render(llm, report)
    out_path = Path(__file__).resolve().parent.parent / "reports" / "smoke_test.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\n[5] Wrote report to {out_path}")

    print("\n--- First 1500 chars of generated report ---\n")
    print(md[:1500])

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
