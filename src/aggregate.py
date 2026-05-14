"""
Aggregation stage. Reads (mention, classification) pairs and produces the
WeeklyReport's analytical content — distributions, deltas, clusters, risk
signals.

Domain-aware: topic weights, actionable-topic sets, and risk-signal rules
are read from the domain registry based on the mention's `domain` field.
This lets one pipeline serve financial_services, higher_education, etc.,
without hardcoding topic enums.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from .domains import DomainConfig, all_topics_for_domain, get_domain
from .schema import (
    Classification,
    Cluster,
    Mention,
    MentionOrigin,
    RiskSignal,
    Sentiment,
    Topic,
    ValidityClaim,
    WeeklyReport,
)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def _sentiment_dist(classifications: list[Classification]) -> dict[str, int]:
    c = Counter(cl.sentiment.label.value for cl in classifications)
    return {label.value: c.get(label.value, 0) for label in Sentiment}


def _topic_dist(classifications: list[Classification],
                domain_name: str | None = None) -> dict[str, int]:
    """Topic distribution. If domain provided, ensures all domain topics appear
    in the output (with zero counts where absent) for stable rendering."""
    c = Counter(cl.topic.label for cl in classifications)
    if domain_name:
        all_topics = all_topics_for_domain(domain_name)
        return {label: c.get(label, 0) for label in all_topics}
    return dict(c)


def _validity_dist(classifications: list[Classification]) -> dict[str, int]:
    c = Counter(cl.validity_claim.label.value for cl in classifications)
    return {label.value: c.get(label.value, 0) for label in ValidityClaim}


def _origin_dist(classifications: list[Classification]) -> dict[str, int]:
    c = Counter(cl.origin.label.value for cl in classifications)
    return {label.value: c.get(label.value, 0) for label in MentionOrigin}


def _negative_share(dist: dict[str, int]) -> float:
    total = sum(dist.values()) or 1
    neg = dist.get(Sentiment.NEGATIVE.value, 0) + dist.get(Sentiment.VERY_NEGATIVE.value, 0)
    return neg / total


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _polarity(label: Sentiment) -> str:
    if label in (Sentiment.NEGATIVE, Sentiment.VERY_NEGATIVE):
        return "neg"
    if label in (Sentiment.POSITIVE, Sentiment.VERY_POSITIVE):
        return "pos"
    return "neu"


def _reach_of(m: Mention) -> int:
    eng = m.engagement or {}
    score = eng.get("score") or 0
    comments = eng.get("num_comments") or 0
    return max(0, int(score)) + 2 * max(0, int(comments))


def _build_clusters(
    mentions: dict[str, Mention],
    classifications: list[Classification],
    domain: DomainConfig,
) -> list[Cluster]:
    """Group by (topic, polarity). Skip 'unrelated'. Emit a Cluster per group
    with >= 2 mentions, scored using the domain's topic weights.

    The string topic label is stashed on cluster._topic_label so reports can
    render the right name even when the label isn't in the legacy Topic enum.
    """
    buckets: dict[tuple[str, str], list[Classification]] = defaultdict(list)
    for cl in classifications:
        if cl.topic.label == "unrelated":
            continue
        key = (cl.topic.label, _polarity(cl.sentiment.label))
        buckets[key].append(cl)

    clusters: list[Cluster] = []
    for (topic_val, polarity), members in buckets.items():
        if len(members) < 2:
            continue
        member_mentions = [mentions[cl.mention_id] for cl in members
                           if cl.mention_id in mentions]
        if not member_mentions:
            continue
        reach = sum(_reach_of(m) for m in member_mentions)

        topic_weight = domain.topic_weights.get(topic_val, 0.5)
        polarity_weight = {"neg": 1.0, "neu": 0.5, "pos": 0.4}[polarity]
        volume_factor = math.log1p(len(members)) / math.log(20)
        severity = min(1.0, topic_weight * polarity_weight * volume_factor)

        examples = [
            (m.text[:300] + "…") if len(m.text) > 300 else m.text
            for m in member_mentions[:3]
        ]

        # Cluster.topic is typed Topic (legacy FS enum). For domain topics
        # that aren't in that enum, use GENERIC_DISCUSSION as a placeholder
        # and stash the real label as ._topic_label.
        try:
            topic_enum = Topic(topic_val)
        except ValueError:
            topic_enum = Topic.GENERIC_DISCUSSION

        cluster = Cluster(
            cluster_id=f"{topic_val}:{polarity}",
            topic=topic_enum,
            topic_label=topic_val,
            theme=f"{topic_val.replace('_', ' ')} ({polarity})",
            mention_ids=[cl.mention_id for cl in members],
            severity=round(severity, 3),
            reach_estimate=reach,
            representative_examples=examples,
        )
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.severity, reverse=True)
    return clusters


# Back-compat: legacy callers that imported this directly
ACTIONABLE_TOPICS: set[Topic] = {
    Topic.FRAUD_CLAIM,
    Topic.OUTAGE_OR_SERVICE_ISSUE,
    Topic.PRODUCT_COMPLAINT,
    Topic.FEE_DISPUTE,
    Topic.COMPETITOR_COMPARISON,
}


# ---------------------------------------------------------------------------
# Risk signals — domain-specific rule sets
# ---------------------------------------------------------------------------

RISK_SIGNAL_RULES: dict[str, list[dict]] = {
    "financial_services": [
        {
            "topic": "fraud_claim", "threshold": 5, "kind": "fraud_spike",
            "owner": "fraud_ops",
            "action_template": (
                "Investigate {n} fraud-related mentions this week. Cross-check against "
                "internal fraud-ops case volume. If external complaints exceed internal "
                "baseline, escalate to fraud leadership."
            ),
        },
        {
            "topic": "outage_or_service_issue", "threshold": 4, "kind": "outage_cluster",
            "owner": "service_desk",
            "action_template": (
                "{n} service-issue mentions this week. Reconcile with platform monitoring "
                "and incident logs to confirm whether external chatter matches internal incidents."
            ),
        },
    ],
    "higher_education": [
        {
            "topic": "safety_or_incident", "threshold": 3, "kind": "safety_cluster",
            "owner": "comms_or_dean",
            "action_template": (
                "{n} safety/incident mentions this week. Verify with campus security and "
                "comms; align any public statement with active investigations."
            ),
        },
        {
            "topic": "system_or_service_issue", "threshold": 4, "kind": "system_cluster",
            "owner": "it_or_registrar",
            "action_template": (
                "{n} system/service mentions this week (registration, financial-aid portal, "
                "IT). Confirm with IT operations whether internal incidents correspond."
            ),
        },
        {
            "topic": "academic_experience", "threshold": 6, "kind": "academic_dissatisfaction",
            "owner": "provost_or_dean",
            "action_template": (
                "{n} academic-experience mentions this week. If trending negative, surface "
                "to academic affairs for course/faculty review."
            ),
        },
    ],
}

SENTIMENT_DRIFT_THRESHOLD = 0.10


def detect_risk_signals(
    clusters: list[Cluster],
    sentiment_dist: dict[str, int],
    prior_sentiment_dist: Optional[dict[str, int]],
    domain_name: str = "financial_services",
) -> list[RiskSignal]:
    signals: list[RiskSignal] = []
    rules = RISK_SIGNAL_RULES.get(domain_name, [])

    # Build topic -> clusters lookup using topic_label (preferred) or topic.value
    by_topic: dict[str, list[Cluster]] = {}
    for cl in clusters:
        topic_label = cl.topic_label or cl.topic.value
        by_topic.setdefault(topic_label, []).append(cl)

    # Topic-driven signals
    for rule in rules:
        cls_for_topic = by_topic.get(rule["topic"], [])
        negative_clusters = [c for c in cls_for_topic if "neg" in c.cluster_id]
        if not negative_clusters:
            continue
        cluster = max(negative_clusters, key=lambda c: len(c.mention_ids))
        if len(cluster.mention_ids) >= rule["threshold"]:
            n = len(cluster.mention_ids)
            signals.append(RiskSignal(
                signal_id=f"{rule['kind']}:{cluster.cluster_id}",
                kind=rule["kind"],
                cluster_id=cluster.cluster_id,
                severity=cluster.severity,
                recommended_owner=rule["owner"],
                recommended_action=rule["action_template"].format(n=n),
                confidence=min(1.0, n / (rule["threshold"] * 2)),
            ))

    # Sentiment drift (universal)
    if prior_sentiment_dist is not None:
        delta = _negative_share(sentiment_dist) - _negative_share(prior_sentiment_dist)
        if delta >= SENTIMENT_DRIFT_THRESHOLD:
            signals.append(RiskSignal(
                signal_id="sentiment_drift",
                kind="sentiment_drift",
                cluster_id="(global)",
                severity=min(1.0, delta * 5),
                recommended_owner="comms",
                recommended_action=(
                    f"Negative-share rose {delta*100:.1f} pts WoW "
                    f"({_negative_share(prior_sentiment_dist)*100:.1f}% → "
                    f"{_negative_share(sentiment_dist)*100:.1f}%). Review top "
                    f"clusters for the dominant driver before responding."
                ),
                confidence=0.8,
            ))

    return signals


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

def build_weekly_report(
    mentions: list[Mention],
    classifications: list[Classification],
    week_start: datetime,
    week_end: datetime,
    prior_report: Optional[WeeklyReport] = None,
    domain_name: str | None = None,
) -> WeeklyReport:
    mentions_by_id = {m.id: m for m in mentions}
    by_source = Counter(m.source.value for m in mentions)

    # Resolve domain — explicit arg > first mention's domain > default
    if domain_name is None:
        domain_name = mentions[0].domain if mentions else "financial_services"
    domain = get_domain(domain_name)

    customer_classifications = [c for c in classifications
                                 if c.origin.label == MentionOrigin.CUSTOMER]

    # Customer-voice distributions — these drive the report
    sentiment_dist_customer = _sentiment_dist(customer_classifications)
    topic_dist_customer = _topic_dist(customer_classifications, domain_name)
    validity_dist_customer = _validity_dist(customer_classifications)

    # Overall — corpus shape, kept for transparency
    sentiment_dist_overall = _sentiment_dist(classifications)
    topic_dist_overall = _topic_dist(classifications, domain_name)
    validity_dist_overall = _validity_dist(classifications)

    origin_dist = _origin_dist(classifications)

    clusters = _build_clusters(mentions_by_id, customer_classifications, domain)

    prior_customer_sentiment = (prior_report.sentiment_distribution_customer
                                 if prior_report else None)
    risk_signals = detect_risk_signals(
        clusters, sentiment_dist_customer, prior_customer_sentiment, domain_name
    )

    wow_volume_delta = None
    wow_sentiment_delta = None
    if prior_report is not None:
        prior_total = prior_report.total_mentions or 1
        wow_volume_delta = (len(mentions) - prior_report.total_mentions) / prior_total
        wow_sentiment_delta = (_negative_share(sentiment_dist_customer) -
                               _negative_share(prior_report.sentiment_distribution_customer))

    return WeeklyReport(
        week_start=week_start,
        week_end=week_end,
        total_mentions=len(mentions),
        total_customer_mentions=len(customer_classifications),
        mentions_by_source=dict(by_source),
        sentiment_distribution=sentiment_dist_customer,
        topic_distribution=topic_dist_customer,
        validity_claim_distribution=validity_dist_customer,
        sentiment_distribution_customer=sentiment_dist_customer,
        topic_distribution_customer=topic_dist_customer,
        validity_claim_distribution_customer=validity_dist_customer,
        sentiment_distribution_overall=sentiment_dist_overall,
        topic_distribution_overall=topic_dist_overall,
        validity_claim_distribution_overall=validity_dist_overall,
        origin_distribution=origin_dist,
        wow_volume_delta=wow_volume_delta,
        wow_sentiment_delta=wow_sentiment_delta,
        clusters=clusters,
        risk_signals=risk_signals,
        executive_summary=[],
    )
