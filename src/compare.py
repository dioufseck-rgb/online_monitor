"""
Cross-brand comparison report — written from the target brand's perspective.

The target brand is the protagonist of the report. Every section frames the
comparison around how the target stacks up: tables put the target in the first
row, the executive read summarizes the target's standing, and computed
percentile ranks ground the narrative claims.

How the insight layer works:

1. For each comparable metric (customer-voice share, negative share, sincerity
   share, brand-owned posting share, etc.), compute the target's percentile
   rank within the cohort and a categorical position (`top`, `upper`, `mid`,
   `lower`, `bottom`).
2. Surface those positions as ground-truth facts that downstream prose can use.
3. The executive read narrates the target's standing using those facts.
   Optional LLM enrichment (--llm) writes a 4-6 sentence prose summary; without
   --llm, a deterministic template-based read is used.

Sample-size humility:
- Per-cell `(%/n)` annotations.
- ⚠ flag when customer-voice n < 20.
- Comparative claims gated when n < 30 in the relevant slice OR the spread
  between brands is too small to be meaningful.

CLI examples:
    python -m src.compare --target nfcu                # explicit target
    python -m src.compare                              # uses config target_brand
    python -m src.compare --target gmu --brands gmu uva vt
    python -m src.compare --target nfcu --weeks 4       # multi-week trend
    python -m src.compare --target nfcu --llm           # LLM-narrated exec read
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

from .llm import build_adapter
from .schema import MentionOrigin, Sentiment, Topic, WeeklyReport
from .store import Store


def _resolve_env(value):
    """Resolve ${VAR} references in config values."""
    import os
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value


def _build_llm_adapter(cfg: dict):
    """Build an LLMAdapter from the provider section of config — matches
    weekly.py so behavior is consistent across the two entry points."""
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


_RISK_TOPICS = [
    Topic.FRAUD_CLAIM,
    Topic.OUTAGE_OR_SERVICE_ISSUE,
    Topic.PRODUCT_COMPLAINT,
    Topic.FEE_DISPUTE,
]

LOW_CONFIDENCE_N = 20
COMPARATIVE_CLAIM_MIN_N = 30


HABERMAS_INTRO = """_The Habermasian framework distinguishes four kinds of claims people make.
**Fact claims** assert something objectively (rates, fees, what the company did).
**Norm claims** assert what should be (policies, values, fairness).
**Experience claims** assert personal feelings (gratitude, frustration, anxiety).
**Meaning claims** ask about understanding (what something means, why something is).
Healthy customer-voice corpora blend all four; corpora that skew heavily toward
one type tell us something about the discourse genre — fact-heavy means
newsworthy/informational, experience-heavy means relational/emotional._
"""


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _negative_share(sentiment_dist: dict[str, int]) -> float:
    total = sum(sentiment_dist.values()) or 1
    return (sentiment_dist.get(Sentiment.NEGATIVE.value, 0) +
            sentiment_dist.get(Sentiment.VERY_NEGATIVE.value, 0)) / total


def _customer_share(origin_dist: dict[str, int]) -> float:
    total = sum(origin_dist.values()) or 1
    return origin_dist.get(MentionOrigin.CUSTOMER.value, 0) / total


def _brand_owned_share(origin_dist: dict[str, int]) -> float:
    total = sum(origin_dist.values()) or 1
    return origin_dist.get(MentionOrigin.BRAND_OWNED.value, 0) / total


def _journalism_share(origin_dist: dict[str, int]) -> float:
    total = sum(origin_dist.values()) or 1
    return origin_dist.get(MentionOrigin.JOURNALISM.value, 0) / total


def _sincerity_share(validity_dist: dict[str, int]) -> float:
    total = sum(validity_dist.values()) or 1
    return validity_dist.get("sincerity", 0) / total


def _rightness_share(validity_dist: dict[str, int]) -> float:
    total = sum(validity_dist.values()) or 1
    return validity_dist.get("rightness", 0) / total


def _risk_topic_share(topic_dist: dict[str, int]) -> float:
    total = sum(topic_dist.values()) or 1
    risk_total = sum(topic_dist.get(t.value, 0) for t in _RISK_TOPICS)
    return risk_total / total


def _pct_with_n(count: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{count / total * 100:.1f}% ({count}/{total})"


def _confidence_marker(n: int) -> str:
    return " ⚠" if n < LOW_CONFIDENCE_N else ""


# ---------------------------------------------------------------------------
# Insight engine — percentile ranks and categorical positions
# ---------------------------------------------------------------------------

@dataclass
class MetricStanding:
    """Where the target brand stands on one metric within the cohort."""
    metric: str            # human-readable name
    target_value: float    # the target's value
    rank: int              # 1-indexed rank (1 = highest)
    cohort_size: int
    median: float          # cohort median
    spread: float          # range between min and max
    position: str          # 'top' | 'upper' | 'mid' | 'lower' | 'bottom'
    direction: str         # 'higher_is_better' | 'lower_is_better' | 'neutral'
    sample_n: int          # the target's sample size for this metric

    @property
    def confident(self) -> bool:
        return self.sample_n >= COMPARATIVE_CLAIM_MIN_N

    @property
    def meaningful_spread(self) -> bool:
        # If max - min is tiny across the cohort, the rank ordering is noise
        return self.spread >= 0.05

    @property
    def differentiated_from_median(self) -> bool:
        """True only if the target's value differs from the cohort median by
        enough to support a comparative claim. Tied or near-tied values
        produce nonsense prose like 'runs more negative than peers (0% vs 0%)'.
        """
        return abs(self.target_value - self.median) >= 0.02  # 2 percentage points


def _categorical_position(rank: int, cohort_size: int) -> str:
    if cohort_size == 1:
        return "alone"
    pct = (cohort_size - rank) / (cohort_size - 1)  # 1.0 = top, 0.0 = bottom
    if pct >= 0.85:
        return "top"
    elif pct >= 0.6:
        return "upper"
    elif pct >= 0.4:
        return "mid"
    elif pct >= 0.15:
        return "lower"
    else:
        return "bottom"


def compute_standings(
    target: str,
    reports: dict[str, WeeklyReport],
) -> list[MetricStanding]:
    """Public entry point; alias for _compute_standings.
    Used by html_report.py to share the target-framed analysis layer."""
    return _compute_standings(target, reports)


def deterministic_exec_read(target: str, standings: list[MetricStanding]) -> list[str]:
    """Public entry point; alias for _deterministic_exec_read."""
    return _deterministic_exec_read(target, standings)


def interpret(s: MetricStanding) -> str:
    """Public entry point; alias for _interpret."""
    return _interpret(s)


def _compute_standings(
    target: str,
    reports: dict[str, WeeklyReport],
) -> list[MetricStanding]:
    """For each comparable metric, compute the target's standing in the cohort.

    Position semantics: `top` = most of this metric, `bottom` = least.
    The prose layer interprets these against `direction` (higher_is_better,
    lower_is_better, neutral) when forming sentences.
    """
    if target not in reports:
        return []

    cohort = list(reports.keys())
    target_r = reports[target]
    target_cust_n = target_r.total_customer_mentions
    target_total_n = target_r.total_mentions

    metrics: list[tuple[str, callable, str, str, int]] = [
        ("customer-voice share",
         lambda r: _customer_share(r.origin_distribution),
         "higher_is_better", "total_mentions", target_total_n),
        ("negative share (customer voice)",
         lambda r: _negative_share(r.sentiment_distribution),
         "lower_is_better", "total_customer_mentions", target_cust_n),
        ("risk-topic share (customer voice)",
         lambda r: _risk_topic_share(r.topic_distribution),
         "lower_is_better", "total_customer_mentions", target_cust_n),
        ("experience-claim share (customer voice)",
         lambda r: _sincerity_share(r.validity_claim_distribution),
         "neutral", "total_customer_mentions", target_cust_n),
        ("norm-claim share (customer voice)",
         lambda r: _rightness_share(r.validity_claim_distribution),
         "neutral", "total_customer_mentions", target_cust_n),
        ("brand-owned posting share",
         lambda r: _brand_owned_share(r.origin_distribution),
         "lower_is_better", "total_mentions", target_total_n),
        ("journalism share",
         lambda r: _journalism_share(r.origin_distribution),
         "neutral", "total_mentions", target_total_n),
    ]

    standings: list[MetricStanding] = []
    for name, value_fn, direction, _, sample_n in metrics:
        values = [(b, value_fn(r)) for b, r in reports.items()]
        # ALWAYS sort highest first — rank 1 = most of this metric.
        # Interpretation against direction happens in the prose layer.
        sorted_pairs = sorted(values, key=lambda kv: kv[1], reverse=True)
        ranks = {b: i + 1 for i, (b, _) in enumerate(sorted_pairs)}

        target_value = value_fn(target_r)
        nums = [v for _, v in values]
        median = sorted(nums)[len(nums) // 2]
        spread = max(nums) - min(nums)

        standings.append(MetricStanding(
            metric=name,
            target_value=target_value,
            rank=ranks[target],
            cohort_size=len(cohort),
            median=median,
            spread=spread,
            position=_categorical_position(ranks[target], len(cohort)),
            direction=direction,
            sample_n=sample_n,
        ))

    return standings


# ---------------------------------------------------------------------------
# Perception dimensions — per-axis cohort positioning for the radar view
# ---------------------------------------------------------------------------

@dataclass
class PerceptionDimension:
    """One axis of perception standing for a single brand.

    Each dimension is computed from cluster data (which preserves polarity)
    plus claim-type distribution, then normalized cohort-relative so the
    radar's max-radius corresponds to the cohort's best score on that axis,
    not an absolute reference.

    `raw_value` is the underlying measurement (e.g. severity of fee_dispute:neg
    cluster). `score` is the normalized 0-1 value where 1 = best in cohort,
    0 = worst in cohort. The radar plots `score`.

    `direction` documents which way "raw" maps to "score":
      - lower_is_better: smaller raw_value -> higher score (e.g. fraud severity)
      - higher_is_better: larger raw_value -> higher score (e.g. praise concentration)
    """
    brand: str
    dimension: str           # human-readable axis name
    raw_value: float         # underlying measurement, 0-1
    score: float             # normalized cohort-relative score, 0-1
    direction: str           # 'lower_is_better' | 'higher_is_better'
    sample_n: int            # customer-voice n that backs this dimension
    rationale: str           # one-line explanation of the raw_value


def _cluster_severity(report: WeeklyReport, cluster_id: str) -> float:
    """Return the severity of a specific cluster, or 0.0 if absent."""
    for c in report.clusters:
        if c.cluster_id == cluster_id:
            return c.severity
    return 0.0


def _cluster_mention_count(report: WeeklyReport, cluster_id: str) -> int:
    """Return the size of a specific cluster, or 0 if absent."""
    for c in report.clusters:
        if c.cluster_id == cluster_id:
            return len(c.mention_ids)
    return 0


# Dimension definitions: (axis_label, computation_fn, direction, rationale_fn)
# Each computation_fn takes a WeeklyReport and returns a raw value in [0, 1].
# Each rationale_fn takes a WeeklyReport and returns a one-line explanation.
_PERCEPTION_DIMENSIONS = [
    (
        "Fraud handling",
        lambda r: _cluster_severity(r, "fraud_claim:neg"),
        "lower_is_better",
        lambda r: (
            f"{_cluster_mention_count(r, 'fraud_claim:neg')} negative fraud_claim "
            f"mentions, severity {_cluster_severity(r, 'fraud_claim:neg'):.2f}"
        ),
    ),
    (
        "Fee fairness",
        lambda r: _cluster_severity(r, "fee_dispute:neg"),
        "lower_is_better",
        lambda r: (
            f"{_cluster_mention_count(r, 'fee_dispute:neg')} negative fee_dispute "
            f"mentions, severity {_cluster_severity(r, 'fee_dispute:neg'):.2f}"
        ),
    ),
    (
        "Service reliability",
        lambda r: _cluster_severity(r, "outage_or_service_issue:neg"),
        "lower_is_better",
        lambda r: (
            f"{_cluster_mention_count(r, 'outage_or_service_issue:neg')} negative "
            f"outage mentions, severity "
            f"{_cluster_severity(r, 'outage_or_service_issue:neg'):.2f}"
        ),
    ),
    (
        "Product",
        lambda r: _cluster_severity(r, "product_complaint:neg"),
        "lower_is_better",
        lambda r: (
            f"{_cluster_mention_count(r, 'product_complaint:neg')} negative "
            f"product_complaint mentions, severity "
            f"{_cluster_severity(r, 'product_complaint:neg'):.2f}"
        ),
    ),
    (
        "Fairness discourse",
        lambda r: _rightness_share(r.validity_claim_distribution),
        "lower_is_better",
        lambda r: (
            f"norm-claim share {_rightness_share(r.validity_claim_distribution)*100:.1f}% "
            f"of customer voice "
            f"(n={sum(r.validity_claim_distribution.values())})"
        ),
    ),
    (
        "Customer service",
        lambda r: _cluster_severity(r, "praise:pos"),
        "higher_is_better",
        lambda r: (
            f"{_cluster_mention_count(r, 'praise:pos')} positive praise mentions, "
            f"severity {_cluster_severity(r, 'praise:pos'):.2f}"
        ),
    ),
]


def compute_perception_dimensions(
    reports: dict[str, WeeklyReport],
) -> dict[str, list[PerceptionDimension]]:
    """Compute per-brand-per-dimension scores, normalized cohort-relative.

    Returns a dict mapping brand -> list of PerceptionDimension (one per axis).
    Each brand's list contains the same axes in the same order.

    Normalization strategy: cohort-relative min-max.
      - For lower_is_better dims: score = 1 - (raw - min) / (max - min)
      - For higher_is_better dims: score = (raw - min) / (max - min)
      - When the cohort has identical values on a dimension (max == min),
        all brands get score 0.5 (uninformative, but well-defined).

    The output is suitable for direct rendering as radar polygon vertices.
    """
    if not reports:
        return {}

    # Compute raw values for every (brand, dimension) pair
    raw_by_brand: dict[str, list[tuple[str, float, str, int, str]]] = {}
    # Track the min/max of each dimension across the cohort
    dim_min: dict[str, float] = {}
    dim_max: dict[str, float] = {}

    for brand, report in reports.items():
        dims_for_brand: list[tuple[str, float, str, int, str]] = []
        for axis, value_fn, direction, rationale_fn in _PERCEPTION_DIMENSIONS:
            raw = value_fn(report)
            rationale = rationale_fn(report)
            dims_for_brand.append((axis, raw, direction, report.total_customer_mentions, rationale))

            if axis not in dim_min or raw < dim_min[axis]:
                dim_min[axis] = raw
            if axis not in dim_max or raw > dim_max[axis]:
                dim_max[axis] = raw
        raw_by_brand[brand] = dims_for_brand

    # Normalize each (brand, dimension) to 0-1 score
    out: dict[str, list[PerceptionDimension]] = {}
    for brand, dims_for_brand in raw_by_brand.items():
        result: list[PerceptionDimension] = []
        for axis, raw, direction, sample_n, rationale in dims_for_brand:
            lo, hi = dim_min[axis], dim_max[axis]
            if hi - lo < 1e-9:
                # Cohort identical on this axis — uninformative, use 0.5
                score = 0.5
            else:
                normalized = (raw - lo) / (hi - lo)  # 0 = cohort-low, 1 = cohort-high
                if direction == "lower_is_better":
                    score = 1.0 - normalized
                else:
                    score = normalized
            result.append(PerceptionDimension(
                brand=brand,
                dimension=axis,
                raw_value=raw,
                score=score,
                direction=direction,
                sample_n=sample_n,
                rationale=rationale,
            ))
        out[brand] = result
    return out


def cohort_median_dimensions(
    perception_by_brand: dict[str, list[PerceptionDimension]],
) -> list[tuple[str, float]]:
    """Compute the cohort median score per dimension.

    Returns a list of (dimension_name, median_score) tuples in the same
    order as PERCEPTION_DIMENSIONS, ready to be plotted as the reference
    polygon on the radar.
    """
    if not perception_by_brand:
        return []
    # Use any brand to get the axis order
    axes = [d.dimension for d in next(iter(perception_by_brand.values()))]
    medians: list[tuple[str, float]] = []
    for axis_idx, axis_name in enumerate(axes):
        scores = [perception_by_brand[b][axis_idx].score
                  for b in perception_by_brand]
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 1:
            median = sorted_scores[n // 2]
        else:
            median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        medians.append((axis_name, median))
    return medians


# ---------------------------------------------------------------------------
# Executive read — deterministic template + optional LLM enrichment
# ---------------------------------------------------------------------------

def _interpret(s: MetricStanding) -> str:
    """Map (position, direction) to a plain-language judgment."""
    if not s.confident:
        return f"low-confidence (position {s.position}, n={s.sample_n})"
    if not s.meaningful_spread:
        return f"in line with peers (cohort spread only {s.spread*100:.1f}pts)"
    if not s.differentiated_from_median:
        return f"near the cohort median ({s.target_value*100:.1f}% vs {s.median*100:.1f}%)"

    # position is always "amount of this metric" with `top` = most
    pos = s.position
    direction = s.direction

    if direction == "higher_is_better":
        return {
            "top": "leads the cohort", "upper": "above the cohort median",
            "mid": "in the middle of the cohort", "lower": "below the cohort median",
            "bottom": "trails the cohort",
        }.get(pos, pos)
    elif direction == "lower_is_better":
        return {
            "top": "trails the cohort (highest)", "upper": "above the cohort median (worse)",
            "mid": "in the middle of the cohort", "lower": "below the cohort median (better)",
            "bottom": "leads the cohort (lowest)",
        }.get(pos, pos)
    else:  # neutral
        return {
            "top": "highest in the cohort", "upper": "above the cohort median",
            "mid": "in the middle of the cohort", "lower": "below the cohort median",
            "bottom": "lowest in the cohort",
        }.get(pos, pos)


def _deterministic_exec_read(target: str, standings: list[MetricStanding]) -> list[str]:
    """Generate insight bullets from computed standings without an LLM."""
    bullets: list[str] = []
    by_metric = {s.metric: s for s in standings}

    customer = by_metric.get("customer-voice share")
    negative = by_metric.get("negative share (customer voice)")
    risk = by_metric.get("risk-topic share (customer voice)")
    sincerity = by_metric.get("experience-claim share (customer voice)")
    brand_owned = by_metric.get("brand-owned posting share")

    # Customer voice (higher is better)
    if customer and customer.confident and customer.meaningful_spread:
        if customer.position in ("top", "upper"):
            bullets.append(
                f"**Organic conversation**: {target.upper()} {_interpret(customer)} on customer-voice share "
                f"({customer.target_value*100:.1f}% vs cohort median {customer.median*100:.1f}%) — "
                f"a larger share of discoverable mentions are actual customers rather than corporate posts "
                f"or journalism. Typically a sign of an engaged community."
            )
        elif customer.position in ("lower", "bottom"):
            bullets.append(
                f"**Organic conversation**: {target.upper()} {_interpret(customer)} on customer-voice share "
                f"({customer.target_value*100:.1f}% vs cohort median {customer.median*100:.1f}%) — "
                f"the bulk of {target.upper()} mentions come from corporate posting, journalism, or "
                f"partners rather than customers themselves."
            )
        else:
            bullets.append(
                f"**Organic conversation**: {target.upper()} is {_interpret(customer)} on customer-voice share "
                f"({customer.target_value*100:.1f}%, cohort median {customer.median*100:.1f}%)."
            )

    # Negative sentiment (lower is better)
    if negative and negative.confident and negative.meaningful_spread:
        if negative.position in ("top", "upper"):
            bullets.append(
                f"**Negative sentiment**: {target.upper()} customer voice runs more negative than peers "
                f"({negative.target_value*100:.1f}% vs cohort median {negative.median*100:.1f}%). "
                f"Worth investigating whether this is driven by a specific topic cluster or a broader shift."
            )
        elif negative.position in ("lower", "bottom"):
            bullets.append(
                f"**Negative sentiment**: {target.upper()} customer voice runs less negative than peers "
                f"({negative.target_value*100:.1f}% vs cohort median {negative.median*100:.1f}%)."
            )
        else:
            bullets.append(
                f"**Negative sentiment**: {target.upper()} is {_interpret(negative)} on negative share "
                f"({negative.target_value*100:.1f}%, cohort median {negative.median*100:.1f}%)."
            )

    # Risk topics (lower is better)
    if risk and risk.confident and risk.meaningful_spread:
        if risk.position in ("top", "upper"):
            bullets.append(
                f"**Risk topics**: {target.upper()} customer voice has a higher concentration of risk-relevant "
                f"discussion (fraud / outage / complaint / fee disputes) at {risk.target_value*100:.1f}% "
                f"vs cohort median {risk.median*100:.1f}%. Surface clusters in those topics for review."
            )
        elif risk.position in ("lower", "bottom"):
            bullets.append(
                f"**Risk topics**: {target.upper()} customer voice has comparatively little risk-topic "
                f"chatter ({risk.target_value*100:.1f}% vs cohort median {risk.median*100:.1f}%)."
            )

    # Discourse genre — sincerity (neutral)
    if sincerity and sincerity.confident and sincerity.meaningful_spread:
        if sincerity.position in ("top", "upper"):
            bullets.append(
                f"**Discourse genre**: {target.upper()} skews relational/experiential — "
                f"{sincerity.target_value*100:.1f}% of customer-voice mentions are experience claims "
                f"vs cohort median {sincerity.median*100:.1f}%. Customers are sharing personal stories about "
                f"the brand, not just stating facts. This is often a strength for community-driven brands "
                f"and a marker of brand affinity."
            )
        elif sincerity.position in ("lower", "bottom"):
            bullets.append(
                f"**Discourse genre**: {target.upper()} skews informational/factual — "
                f"{sincerity.target_value*100:.1f}% experience claims vs cohort median "
                f"{sincerity.median*100:.1f}%. Customers are stating facts about the brand more than expressing "
                f"personal experience. Common for brands discussed primarily through news and product "
                f"comparisons rather than user testimony."
            )

    # Corporate posting (lower is better)
    if brand_owned and brand_owned.confident and brand_owned.meaningful_spread:
        if brand_owned.position in ("top", "upper"):
            bullets.append(
                f"**Corporate posting**: {target.upper()} carries a high brand-owned share "
                f"({brand_owned.target_value*100:.1f}% vs cohort median {brand_owned.median*100:.1f}%) — "
                f"the brand's own communications make up a large fraction of discoverable mentions. "
                f"Customer voice is correspondingly suppressed in the raw counts."
            )

    return bullets


_LLM_EXEC_SYSTEM = """You write a brief executive read (4-6 sentences) for a Voice-of-Customer comparison report.

You will receive structured FACTS describing where the target brand stands relative to its cohort on each metric. The facts include rank, percentile position, target value, cohort median, and sample size.

Your job is to write a flowing narrative paragraph from the target brand's perspective that:
- Leads with the target's position on the most differentiated metric (where it's most clearly above or below peers).
- Uses ONLY the facts provided — never invent rankings or numbers.
- Names specific percentages from the facts when describing positions.
- Acknowledges low-sample-size or low-spread caveats when the facts flag them.
- Stays direct and analytical; avoid marketing language ("excellent", "market leader") and consultant-speak ("leverages", "drives synergies").
- Never says "data shows" or "analysis reveals" — just state what's true.

Output format: a single paragraph of 4-6 sentences, no bullets, no headers. Plain prose.
Place the entire paragraph in the `prose` field of the response."""


# Pydantic model for prose response (kept inline to avoid schema.py churn)
def _prose_model():
    from pydantic import BaseModel, Field
    class ProseResponse(BaseModel):
        prose: str = Field(..., max_length=2000)
    return ProseResponse


def _llm_exec_read(llm, target: str, standings: list[MetricStanding]) -> str:
    """Generate a flowing prose executive read from computed standings."""
    facts_lines = []
    for s in standings:
        confident = "confident" if s.confident else f"low-confidence (n={s.sample_n})"
        spread = "meaningful spread" if s.meaningful_spread else f"low spread ({s.spread*100:.1f}pts)"
        facts_lines.append(
            f"- {s.metric}: target={s.target_value*100:.1f}%, "
            f"cohort_median={s.median*100:.1f}%, "
            f"rank={s.rank}/{s.cohort_size}, "
            f"position={s.position}, "
            f"direction={s.direction}, "
            f"{confident}, {spread}"
        )
    facts = "\n".join(facts_lines)

    user = (
        f"Target brand: {target.upper()}\n"
        f"Cohort size: {standings[0].cohort_size if standings else 0}\n\n"
        f"Facts:\n{facts}\n\n"
        f"Write the executive read."
    )

    try:
        ProseResponse = _prose_model()
        response = llm.classify_structured(_LLM_EXEC_SYSTEM, user, ProseResponse)
        return response.prose.strip()
    except Exception as e:
        return f"(LLM exec read failed: {e})"


# ---------------------------------------------------------------------------
# Single-week comparison rendering
# ---------------------------------------------------------------------------

def render_comparison_markdown(
    reports: dict[str, WeeklyReport],
    target: str | None = None,
    llm=None,
) -> str:
    """Render a target-centered cross-brand comparison.

    If `target` is provided and present in reports, the target leads every
    table, the executive read frames the comparison from its perspective, and
    insight bullets describe its standing.
    """
    out = StringIO()
    w = out.write

    # Reorder: target first, then peers in alphabetical order
    if target and target in reports:
        ordered = {target: reports[target]}
        for b, r in sorted(reports.items()):
            if b != target:
                ordered[b] = r
        reports = ordered

    sample = next(iter(reports.values()))

    # Header with target prominent
    if target and target in reports:
        w(f"# {target.upper()} — Voice-of-Customer Comparison Report\n\n")
        w(f"_Week of {sample.week_start.date()} &middot; "
          f"window {sample.week_start.date()} → {sample.week_end.date()}_  \n")
        peers = [b for b in reports if b != target]
        w(f"_Compared against: {', '.join(peers)}_\n\n")
    else:
        w(f"# Multi-Brand Comparison — Week of {sample.week_start.date()}\n\n")
        w(f"_Window: {sample.week_start.date()} → {sample.week_end.date()}_  \n")
        w(f"_Brands: {', '.join(reports.keys())}_\n\n")

    # ---- Executive read ----
    if target and target in reports:
        standings = _compute_standings(target, reports)
        w("## Executive Read\n\n")
        if llm is not None:
            prose = _llm_exec_read(llm, target, standings)
            w(prose + "\n\n")
        else:
            bullets = _deterministic_exec_read(target, standings)
            if bullets:
                for b in bullets:
                    w(f"- {b}\n")
                w("\n")
            else:
                w("_Insufficient sample size or spread for confident comparative claims this week._\n\n")

        # ---- Standings table — explicit ranks for transparency ----
        w("## Where " + target.upper() + " stands\n\n")
        w("_Per-metric rank within the cohort (rank 1 = highest value), with the target's "
          "value, the cohort median, and a direction-aware judgment._\n\n")
        w("| Metric | Direction | Target | Cohort median | Rank | Judgment | n |\n")
        w("|---|---|---:|---:|---:|---|---:|\n")
        for s in standings:
            confident_marker = "" if s.confident else " ⚠"
            judgment = _interpret(s)
            direction_short = {
                "higher_is_better": "↑ better",
                "lower_is_better": "↓ better",
                "neutral": "—",
            }.get(s.direction, s.direction)
            w(f"| {s.metric}{confident_marker} | {direction_short} | "
              f"{s.target_value*100:.1f}% | "
              f"{s.median*100:.1f}% | "
              f"{s.rank}/{s.cohort_size} | "
              f"{judgment} | "
              f"{s.sample_n} |\n")
        w("\n")
        w("_Rank 1 means highest value of the metric. The **Direction** column says whether higher "
          "or lower is better; the **Judgment** column reads position-against-direction in plain "
          "language (\"leads the cohort,\" \"trails the cohort,\" etc.). "
          f"⚠ marks metrics where the target's sample size is below {COMPARATIVE_CLAIM_MIN_N}; "
          "rankings are unreliable._\n\n")

    w("> ⚠ marks brands with fewer than 20 mentions in the relevant slice. "
      "Numbers are reported with sample sizes (`%/n`).\n\n")

    # ---- Headline ----
    w("## Headline Comparison\n\n")
    w("| Brand | Total | Customer voice (n) | Negative share | Risk-topic share | Risk signals |\n")
    w("|---|---:|---:|---:|---:|---:|\n")
    for brand, r in reports.items():
        n = r.total_mentions
        cust_n = r.total_customer_mentions or r.origin_distribution.get(MentionOrigin.CUSTOMER.value, 0)
        marker = _confidence_marker(cust_n)
        target_marker = " **(target)**" if brand == target else ""
        neg_n = (r.sentiment_distribution.get(Sentiment.NEGATIVE.value, 0) +
                 r.sentiment_distribution.get(Sentiment.VERY_NEGATIVE.value, 0))
        risk_n = sum(r.topic_distribution.get(t.value, 0) for t in _RISK_TOPICS)
        w(f"| {brand}{marker}{target_marker} | {n} | {cust_n} | "
          f"{_pct_with_n(neg_n, cust_n)} | {_pct_with_n(risk_n, cust_n)} | "
          f"{len(r.risk_signals)} |\n")
    w("\n")
    w("_**Negative share** and **risk-topic share** are computed against "
      "customer-voice mentions only — they describe how customers talk about each "
      "brand, not the full corpus._\n\n")

    # ---- Sentiment ----
    w("## Sentiment Side-by-Side (customer voice only)\n\n")
    w("| Brand | very_neg | neg | neutral | pos | very_pos |\n")
    w("|---|---:|---:|---:|---:|---:|\n")
    for brand, r in reports.items():
        sd = r.sentiment_distribution
        n = sum(sd.values())
        marker = _confidence_marker(n)
        target_marker = " **(target)**" if brand == target else ""
        def cell(label, _sd=sd, _n=n):
            return _pct_with_n(_sd.get(label, 0), _n)
        w(f"| {brand}{marker}{target_marker} | {cell('very_negative')} | {cell('negative')} | "
          f"{cell('neutral')} | {cell('positive')} | {cell('very_positive')} |\n")
    w("\n")

    # ---- Topic ----
    w("## Topic Mix Side-by-Side (customer voice only)\n\n")
    topic_labels = [t.value for t in Topic if t != Topic.UNRELATED]
    w("| Brand | " + " | ".join(topic_labels) + " |\n")
    w("|---|" + "|".join(["---:"] * len(topic_labels)) + "|\n")
    for brand, r in reports.items():
        td = r.topic_distribution
        n = sum(td.values())
        marker = _confidence_marker(n)
        target_marker = " **(target)**" if brand == target else ""
        cells = [_pct_with_n(td.get(t, 0), n) for t in topic_labels]
        w(f"| {brand}{marker}{target_marker} | " + " | ".join(cells) + " |\n")
    w("\n")

    # ---- Validity-claim ----
    w("## How customers talk: claim types (Habermasian, customer voice only)\n\n")
    w(HABERMAS_INTRO)
    w("\n")
    w("| Brand | fact claims | norm claims | experience claims | meaning claims |\n")
    w("|---|---:|---:|---:|---:|\n")
    for brand, r in reports.items():
        vd = r.validity_claim_distribution
        n = sum(vd.values())
        marker = _confidence_marker(n)
        target_marker = " **(target)**" if brand == target else ""
        def cell(label, _vd=vd, _n=n):
            return _pct_with_n(_vd.get(label, 0), _n)
        w(f"| {brand}{marker}{target_marker} | {cell('truth')} | {cell('rightness')} | "
          f"{cell('sincerity')} | {cell('comprehensibility')} |\n")
    w("\n")
    w("_**Reading this table:** a brand high on **fact claims** is being discussed informationally. "
      "A brand high on **experience claims** is being discussed relationally. "
      "**Norm claims** indicate active customer pushback on policy or fairness. "
      "**Meaning claims** should be rare; if they're not, the topic-classification confidence "
      "should be inspected._\n\n")

    # ---- Origin ----
    w("## Origin Mix: Who is doing the talking\n\n")
    w("_How much of each brand's discoverable conversation is the brand itself vs actual customer voice. "
      "High brand-owned share usually means heavy corporate posting; high customer share means more "
      "organic conversation; high journalism share means the brand is currently newsworthy._\n\n")
    w("| Brand | customer | brand_owned | employee_personal | journalism | partner |\n")
    w("|---|---:|---:|---:|---:|---:|\n")
    for brand, r in reports.items():
        od = r.origin_distribution
        n = sum(od.values())
        marker = _confidence_marker(n)
        target_marker = " **(target)**" if brand == target else ""
        def cell(label, _od=od, _n=n):
            return _pct_with_n(_od.get(label, 0), _n)
        w(f"| {brand}{marker}{target_marker} | {cell('customer')} | {cell('brand_owned')} | "
          f"{cell('employee_personal')} | {cell('journalism')} | {cell('partner')} |\n")
    w("\n")

    # ---- Risk signals ----
    w("## Risk Signals This Week\n\n")
    any_signals = False
    for brand, r in reports.items():
        if r.risk_signals:
            any_signals = True
            target_marker = " (target)" if brand == target else ""
            w(f"### {brand}{target_marker}\n")
            for sig in r.risk_signals:
                w(f"- **{sig.kind}** (severity {sig.severity:.2f}, owner `{sig.recommended_owner}`): "
                  f"{sig.recommended_action}\n")
            w("\n")
    if not any_signals:
        w("_No brand triggered risk signals this week._\n\n")

    # ---- Methodology ----
    w("## Methodology Notes\n\n")
    w("- All brands run through the same pipeline (Serper discovery, full-text enrichment, "
      "four-classifier set: sentiment, topic, validity-claim, origin).\n")
    w("- Customer-voice filter applies before cluster construction — actionable clusters per brand "
      "exclude brand-owned, journalism, employee, and partner content.\n")
    w("- Negative share = (very_negative + negative) / customer-voice mentions.\n")
    w("- Risk-topic share = sum of mentions in fraud / outage / product_complaint / fee_dispute, "
      "divided by customer-voice mentions.\n")
    w(f"- Confidence threshold: brands with n < {LOW_CONFIDENCE_N} are flagged ⚠. "
      f"Comparative claims require n ≥ {COMPARATIVE_CLAIM_MIN_N} in the relevant slice.\n")
    w("- Position categories: `top` (top 15%), `upper` (15-40%), `mid` (40-60%), "
      "`lower` (60-85%), `bottom` (bottom 15%) of cohort rank.\n")
    w("- Comparisons are within-week unless `--weeks N` is used; trends require multi-week history.\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Multi-week trend rendering
# ---------------------------------------------------------------------------

def render_trend_markdown(
    history: dict[str, list[WeeklyReport]],
    target: str | None = None,
) -> str:
    out = StringIO()
    w = out.write

    if not history:
        return "_No history available._\n"

    # Reorder: target first
    if target and target in history:
        ordered = {target: history[target]}
        for b in sorted(history.keys()):
            if b != target:
                ordered[b] = history[b]
        history = ordered

    weeks_count = max(len(v) for v in history.values()) if history else 0
    sample_brand = next(iter(history.keys()))
    sample_reports = history[sample_brand]
    if not sample_reports:
        return "_No history available._\n"

    if target and target in history:
        w(f"# {target.upper()} — Multi-Week Trend Report\n\n")
        w(f"_{weeks_count} weeks ending {sample_reports[-1].week_start.date()}_  \n")
        peers = [b for b in history if b != target]
        w(f"_Compared against: {', '.join(peers)}_\n\n")
    else:
        w(f"# Multi-Brand Trend Report — {weeks_count} weeks ending "
          f"{sample_reports[-1].week_start.date()}\n\n")
        w(f"_Brands compared: {', '.join(history.keys())}_\n\n")

    week_labels = sorted({r.week_start.date().isoformat()
                          for reps in history.values() for r in reps})

    def _row_marker(brand: str) -> str:
        return " **(target)**" if brand == target else ""

    # Volume
    w("## Total Mention Volume by Week\n\n")
    w("_Both bars: total mentions (all origins) and customer-voice subset._\n\n")
    w("| Brand | " + " | ".join(week_labels) + " |\n")
    w("|---|" + "|".join(["---:"] * len(week_labels)) + "|\n")
    for brand, reps in history.items():
        by_week = {r.week_start.date().isoformat():
                   f"{r.total_mentions} ({r.total_customer_mentions} cust)"
                   for r in reps}
        cells = [str(by_week.get(wk, "—")) for wk in week_labels]
        w(f"| {brand}{_row_marker(brand)} | " + " | ".join(cells) + " |\n")
    w("\n")

    # Customer-voice share trend
    w("## Customer-Voice Share by Week\n\n")
    w("| Brand | " + " | ".join(week_labels) + " |\n")
    w("|---|" + "|".join(["---:"] * len(week_labels)) + "|\n")
    for brand, reps in history.items():
        by_week = {r.week_start.date().isoformat():
                   _customer_share(r.origin_distribution) for r in reps}
        cells = [f"{by_week[wk]*100:.1f}%" if wk in by_week else "—" for wk in week_labels]
        w(f"| {brand}{_row_marker(brand)} | " + " | ".join(cells) + " |\n")
    w("\n")

    # Negative share trend
    w("## Negative Share by Week (customer voice only)\n\n")
    w("| Brand | " + " | ".join(week_labels) + " |\n")
    w("|---|" + "|".join(["---:"] * len(week_labels)) + "|\n")
    for brand, reps in history.items():
        by_week = {r.week_start.date().isoformat():
                   _negative_share(r.sentiment_distribution) for r in reps}
        cells = [f"{by_week[wk]*100:.1f}%" if wk in by_week else "—" for wk in week_labels]
        w(f"| {brand}{_row_marker(brand)} | " + " | ".join(cells) + " |\n")
    w("\n")

    # Risk-topic share trend
    w("## Risk-Topic Share by Week (customer voice only)\n\n")
    w("| Brand | " + " | ".join(week_labels) + " |\n")
    w("|---|" + "|".join(["---:"] * len(week_labels)) + "|\n")
    for brand, reps in history.items():
        by_week = {r.week_start.date().isoformat():
                   _risk_topic_share(r.topic_distribution) for r in reps}
        cells = [f"{by_week[wk]*100:.1f}%" if wk in by_week else "—" for wk in week_labels]
        w(f"| {brand}{_row_marker(brand)} | " + " | ".join(cells) + " |\n")
    w("\n")

    # Reading the trends
    w("## Reading the Trends\n\n")
    w("- **Customer-voice share**: a sustained drop usually means corporate posting picked up "
      "relative to organic discussion. A sustained rise may mean a controversy or product launch "
      "is generating customer chatter.\n")
    w("- **Negative-share trend**: weekly noise is high; multi-week trends need ≥3 consecutive "
      "weeks of movement in the same direction to be worth investigating.\n")
    w("- **Risk-topic share**: sustained rises in fraud + outage + complaint share are the "
      "strongest pre-incident signal this pipeline produces. Watch for two consecutive rises.\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_target(args, cfg: dict | None) -> str | None:
    if args.target:
        return args.target
    if cfg and cfg.get("target_brand"):
        return cfg["target_brand"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Target brand to frame the report around")
    parser.add_argument("--brands", nargs="*", help="Brands to include (default: all in store)")
    parser.add_argument("--week", help="ISO date for the week start (default: most recent)")
    parser.add_argument("--weeks", type=int, default=1,
                        help="Number of weeks for trend view (default: 1, snapshot only)")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM to write a narrative executive read (otherwise deterministic)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--config-path", default="config/config.yaml")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    # Load config (for target_brand default)
    cfg = None
    config_path = Path(args.config_path)
    if config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text())
        except Exception as e:
            print(f"[compare] Could not load config ({e}); proceeding without")

    target = _resolve_target(args, cfg)

    store = Store(args.store_path)

    if args.week:
        target_date = datetime.fromisoformat(args.week).replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        target_date = datetime.now(tz=timezone.utc) + timedelta(days=1)

    if args.brands:
        brands_to_load = args.brands
    else:
        cur = store._conn.execute("SELECT DISTINCT brand FROM reports")
        brands_to_load = [row[0] for row in cur]

    # Build LLM if requested
    llm = None
    if args.llm:
        if cfg is None:
            print("[compare] --llm requested but no config loaded; cannot build LLM. "
                  "Falling back to deterministic exec read.")
        else:
            try:
                llm = _build_llm_adapter(cfg)
                print(f"[compare] LLM-narrated exec read enabled.")
            except Exception as e:
                print(f"[compare] Could not build LLM ({e}); falling back to deterministic.")

    if args.weeks == 1:
        # Snapshot
        reports: dict[str, WeeklyReport] = {}
        for brand in brands_to_load:
            r = store.get_prior_report(target_date, brand=brand)
            if r is None:
                print(f"[compare] No report for brand={brand}; skipping.")
                continue
            reports[brand] = r

        if not reports:
            print("[compare] No reports found. Run `python -m src.weekly` first.")
            return

        if target and target not in reports:
            print(f"[compare] Target brand '{target}' not in available reports; "
                  f"falling back to alphabetical first.")
            target = None

        md = render_comparison_markdown(reports, target=target, llm=llm)
        week_label = next(iter(reports.values())).week_start.date().isoformat()
        prefix = f"{target}_comparison" if target else "comparison"
        output_path = Path(args.output_path) if args.output_path else \
            Path(__file__).parent.parent / "reports" / f"{prefix}_{week_label}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md)
        print(f"[compare] Comparison report written to {output_path}")
        if target:
            print(f"[compare] Target: {target}; peers: {[b for b in reports if b != target]}")
        else:
            print(f"[compare] Brands compared: {list(reports.keys())}")
        store.close()
        return

    # Multi-week trend
    history: dict[str, list[WeeklyReport]] = {}
    for brand in brands_to_load:
        reps = _load_history(store, brand, target_date, args.weeks)
        if reps:
            history[brand] = reps

    if not history:
        print("[compare] No history found.")
        return

    if target and target not in history:
        print(f"[compare] Target brand '{target}' not in history; ignoring target framing.")
        target = None

    md = render_trend_markdown(history, target=target)
    sample_reps = next(iter(history.values()))
    week_label = sample_reps[-1].week_start.date().isoformat()
    prefix = f"{target}_trend" if target else "trend"
    output_path = Path(args.output_path) if args.output_path else \
        Path(__file__).parent.parent / "reports" / f"{prefix}_{args.weeks}w_{week_label}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"[compare] Trend report written to {output_path}")
    if target:
        print(f"[compare] Target: {target}")
    print(f"[compare] Week counts: " +
          ", ".join(f"{b}={len(reps)}" for b, reps in history.items()))
    store.close()


def _load_history(store: Store, brand: str, target: datetime, n_weeks: int) -> list[WeeklyReport]:
    cur = store._conn.execute(
        """SELECT report_json FROM reports
           WHERE brand = ? AND week_start < ?
           ORDER BY week_start DESC LIMIT ?""",
        (brand, target.isoformat(), n_weeks),
    )
    out = []
    for row in cur:
        out.append(WeeklyReport.model_validate_json(row[0]))
    out.reverse()
    return out


if __name__ == "__main__":
    main()