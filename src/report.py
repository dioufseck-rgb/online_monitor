"""
Report rendering. Pure transformation: WeeklyReport -> markdown string.

The executive_summary field is the only piece that needs an LLM. We pass
the structured report data (no raw mentions) to keep the prompt small and
deterministic. The LLM never gets to fabricate numbers — it only paraphrases
the structured signals.

Severity-driven ordering: risk signals first, then top clusters, then
distributions, then competitor panel (placeholder for v1.2), then appendix.
"""

from __future__ import annotations
from datetime import datetime
from io import StringIO

from pydantic import BaseModel, Field

from .llm import LLMAdapter
from .schema import WeeklyReport


# -----------------------------------------------------------------------------
# Executive summary (LLM-generated)
# -----------------------------------------------------------------------------

class ExecutiveSummary(BaseModel):
    """Forced structure for the LLM-generated exec summary."""
    bullets: list[str] = Field(..., min_length=2, max_length=5)


EXEC_SYSTEM = """You write the executive summary of a weekly Voice-of-Customer mentions monitoring report. Your reader is a comms / risk / customer-ops leader who has 30 seconds to read this before deciding what to act on.

You will receive structured analytics. Do NOT invent numbers — only restate or paraphrase what's in the structured data.

Output an ExecutiveSummary with 3-5 bullets. Strict rules:

CONTENT RULES:
- Lead with what is actionable or has changed week-over-week. Risk signals first, then negative-cluster shifts, then notable WoW deltas.
- Each bullet must imply an action, a question to investigate, or a decision — not just describe what is largest.
- If ANY risk signal is present in the input, the first bullet MUST address it specifically (kind, severity, owner) and you MUST NOT use the phrase "Quiet week" anywhere in the summary.
- "Generic discussion is the largest cluster" is FORBIDDEN. That's volume noise. The reader doesn't act on it.
- "Recruitment dominates the LinkedIn signal" is FORBIDDEN unless it's a meaningful WoW shift.
- Quote at most one specific number per bullet, and only if the number drives the action.
- If a fraud or outage or safety cluster appears in actionable clusters, name it explicitly with the recommended owner from the risk-signal data.
- If WoW negative-share rose meaningfully (>5pts), call it out and point at the top driving cluster.
- ONLY if (a) zero risk signals, (b) no notable WoW deltas, AND (c) no negative actionable clusters reach concern threshold — say directly: "Quiet week — no action items, monitoring continues." Otherwise this phrase is FORBIDDEN.
- One bullet maximum may describe the dominant non-actionable signal as context.
- Bullets should be specific to what's IN the actionable clusters. If a product_complaint cluster has a representative example mentioning "mortgage underwriting," reference that specific topic in the bullet, not the generic category name.

STYLE RULES:
- Briefing-grade, dry, direct. No marketing language.
- One sentence per bullet, declarative.
- No hedging ("might be worth considering"), no "we should" — say what to do or watch.
- Address the reader implicitly; no "your team" or "you".
- Use the brand name from the input, not "the brand" or "the institution".

EXAMPLES OF GOOD BULLETS:
- "Fraud_spike triggered: 7 BBB Scam Tracker reports of impersonation scams targeting BofA accounts — fraud_ops should cross-check against internal case volume."
- "Three Reddit complaints about credit card application denials in r/NavyFederal — review underwriting changes if pattern persists next week."
- "Negative share rose 4pp WoW driven by an outage cluster around the recent app update (broken biometric login, forced 2FA) — escalate to mobile-ops."
- "Two fraud_claim mentions surfaced — below threshold but worth cross-checking against fraud_ops case volume."

EXAMPLES OF FORBIDDEN BULLETS:
- "Generic discussion remains the largest topic cluster with 52 mentions." (no action)
- "Sentiment was mostly neutral." (no action, no number worth knowing)
- "Total mentions increased 1.7% WoW to 121." (insignificant delta, no action)
- "Quiet week — no action items, monitoring continues." (when a risk signal fired — direct contradiction)"""


def _generate_executive_summary(
    llm: LLMAdapter,
    report: WeeklyReport,
    brand: str = "the brand",
) -> list[str]:
    payload_lines = [
        f"Brand: {brand.upper()}",
        f"Window: {report.week_start.date()} to {report.week_end.date()}",
        f"Total mentions: {report.total_mentions} "
        f"({report.total_customer_mentions} customer voice)",
        f"By source: {report.mentions_by_source}",
        f"Sentiment distribution (customer voice): {report.sentiment_distribution}",
        f"Topic distribution (customer voice): {report.topic_distribution}",
        f"Validity-claim distribution (customer voice): {report.validity_claim_distribution}",
    ]
    if report.wow_volume_delta is not None:
        payload_lines.append(f"WoW volume delta: {report.wow_volume_delta:+.1%}")
    if report.wow_sentiment_delta is not None:
        payload_lines.append(f"WoW negative-share delta: {report.wow_sentiment_delta:+.3f}")

    payload_lines.append(f"\nRisk signals ({len(report.risk_signals)}):")
    if not report.risk_signals:
        payload_lines.append("  (none triggered this week)")
    for sig in report.risk_signals:
        payload_lines.append(
            f"  - {sig.kind} (severity {sig.severity:.2f}, owner {sig.recommended_owner}): "
            f"{sig.recommended_action}"
        )

    payload_lines.append(f"\nTop clusters by severity (with sample examples for grounding):")
    for cl in report.clusters[:5]:
        topic_label = getattr(cl, "topic_label", None) or cl.topic.value
        payload_lines.append(
            f"  - {cl.theme} (topic: {topic_label}): {len(cl.mention_ids)} mentions, "
            f"severity {cl.severity:.2f}"
        )
        for ex in cl.representative_examples[:2]:
            ex_short = ex[:200].replace("\n", " ")
            payload_lines.append(f"      example: \"{ex_short}\"")

    user = "\n".join(payload_lines)
    result = llm.classify_structured(EXEC_SYSTEM, user, ExecutiveSummary)
    return result.bullets


# -----------------------------------------------------------------------------
# Markdown rendering (pure)
# -----------------------------------------------------------------------------

def _fmt_pct(x: float | None) -> str:
    return f"{x:+.1%}" if x is not None else "n/a (no prior baseline)"


def _render_distribution(dist: dict[str, int], total: int) -> str:
    if total == 0:
        return "_no data_"
    rows = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    lines = [f"| {k} | {v} | {v/total:.1%} |" for k, v in rows if v > 0]
    if not lines:
        return "_no data_"
    return "| Category | Count | Share |\n|---|---|---|\n" + "\n".join(lines)


def render_markdown(
    report: WeeklyReport,
    brand: str = "nfcu",
    domain_name: str = "financial_services",
) -> str:
    out = StringIO()
    w = out.write

    w(f"# {brand.upper()} Mentions Report — Week of {report.week_start.date()}\n\n")
    w(f"_Window: {report.week_start.date()} → {report.week_end.date()}_\n\n")

    # Executive summary
    w("## Executive Summary\n\n")
    if report.executive_summary:
        for b in report.executive_summary:
            w(f"- {b}\n")
    else:
        w("_(executive summary not generated)_\n")
    w("\n")

    # Headline numbers
    w("## Headline Numbers\n\n")
    w(f"- **Total mentions:** {report.total_mentions} "
      f"({report.total_customer_mentions} customer voice, "
      f"{report.total_mentions - report.total_customer_mentions} brand-owned/journalism/employee/partner)\n")
    w(f"- **WoW volume change:** {_fmt_pct(report.wow_volume_delta)}\n")
    w(f"- **WoW negative-share change (customer voice):** "
      f"{_fmt_pct(report.wow_sentiment_delta) if report.wow_sentiment_delta is not None else 'n/a'}\n")
    w(f"- **By source:** {report.mentions_by_source}\n\n")

    # Risk signals — most important section
    w("## Risk Signals\n\n")
    if not report.risk_signals:
        w("_No risk signals triggered this week._\n\n")
    else:
        for sig in report.risk_signals:
            w(f"### {sig.kind} (severity {sig.severity:.2f})\n")
            w(f"- **Owner:** `{sig.recommended_owner}`\n")
            w(f"- **Confidence:** {sig.confidence:.0%}\n")
            w(f"- **Action:** {sig.recommended_action}\n")
            w(f"- **Cluster:** `{sig.cluster_id}`\n\n")

    # Top clusters — split into actionable and context using the domain config
    from .domains import get_domain
    try:
        domain = get_domain(domain_name)
    except ValueError:
        domain = get_domain("financial_services")

    def _cluster_topic_label(cl):
        return cl.topic_label or cl.topic.value

    # Soft-actionable topics: only the negative-polarity bucket counts as actionable.
    # Neutral competitor_comparison is conversation, not action; negative is real signal.
    SOFT_ACTIONABLE = {
        "competitor_comparison",
        "rankings_or_reputation",
        "alumni_or_career_outcomes",
    }

    def _is_actionable(cl):
        topic = _cluster_topic_label(cl)
        if topic not in domain.actionable_topics:
            return False
        if topic not in SOFT_ACTIONABLE:
            return True
        # Soft-actionable: only negative polarity counts
        return cl.cluster_id.endswith(":neg")

    actionable = [c for c in report.clusters if _is_actionable(c)]
    context = [c for c in report.clusters if not _is_actionable(c)]

    w("## Actionable Clusters\n\n")
    if not actionable:
        w("_No actionable clusters this week — no actionable topics reached the 2-mention threshold._\n\n")
    else:
        for cl in actionable:
            w(f"### {cl.theme} — {len(cl.mention_ids)} mentions, severity {cl.severity:.2f}\n")
            w(f"- **Topic:** `{_cluster_topic_label(cl)}`\n")
            w("- **Examples:**\n")
            for ex in cl.representative_examples:
                quoted = "\n  ".join(ex.split("\n"))
                w(f"  > {quoted}\n")
            w("\n")

    w("## Context Clusters\n\n")
    w("_Lower-priority clusters: praise, employment, generic discussion. Useful for ambient sentiment, not action items._\n\n")
    if not context:
        w("_No context clusters this week._\n\n")
    else:
        for cl in context[:5]:
            w(f"### {cl.theme} — {len(cl.mention_ids)} mentions, severity {cl.severity:.2f}\n")
            w(f"- **Topic:** `{_cluster_topic_label(cl)}`\n")
            w("- **Examples:**\n")
            for ex in cl.representative_examples:
                quoted = "\n  ".join(ex.split("\n"))
                w(f"  > {quoted}\n")
            w("\n")

    # Distributions
    w("## Sentiment Distribution (customer voice only)\n\n")
    w(f"_Computed from the {report.total_customer_mentions} customer-voice mentions, "
      f"excluding brand-owned, journalism, employee-personal, and partner content._\n\n")
    w(_render_distribution(report.sentiment_distribution, report.total_customer_mentions))
    w("\n\n")

    w("## Topic Distribution (customer voice only)\n\n")
    w(_render_distribution(report.topic_distribution, report.total_customer_mentions))
    w("\n\n")

    w("## Validity-Claim Distribution (Habermasian, customer voice only)\n\n")
    w("_**Fact claims** (truth) assert facts; **norm claims** (rightness) assert what should be; "
      "**experience claims** (sincerity) express personal feelings; **meaning claims** "
      "(comprehensibility) ask about understanding. Healthy customer-voice corpora blend all four._\n\n")
    paraphrase = {
        "truth": "fact claims (truth)",
        "rightness": "norm claims (rightness)",
        "sincerity": "experience claims (sincerity)",
        "comprehensibility": "meaning claims (comprehensibility)",
    }
    relabeled = {paraphrase.get(k, k): v for k, v in report.validity_claim_distribution.items()}
    w(_render_distribution(relabeled, report.total_customer_mentions))
    w("\n\n")

    # Origin distribution — separates customer voice from brand/journalism/partner
    w("## Mention Origin Distribution\n\n")
    w("_Customer voice is the actionable signal. Other origins are tracked for context._\n\n")
    w(_render_distribution(report.origin_distribution, report.total_mentions))
    w("\n\n")

    # Competitor panel placeholder
    w("## Competitor Comparison\n\n")
    w("_v1.2 — pipeline runs against PenFed, USAA, Service CU. "
      "For this report, see the `competitor_comparison` topic count above as a within-NFCU-conversation proxy._\n\n")

    # Overall (full-corpus) distributions — kept for transparency
    if report.sentiment_distribution_overall:
        w("## Appendix: Full-Corpus Distributions (all mentions)\n\n")
        w("_These include brand-owned, journalism, employee-personal, and partner mentions. "
          "They describe the corpus shape, not customer voice. Use the customer-voice "
          "distributions above for actionable signal._\n\n")
        w("### Sentiment (all mentions)\n\n")
        w(_render_distribution(report.sentiment_distribution_overall, report.total_mentions))
        w("\n\n")
        w("### Topic (all mentions)\n\n")
        w(_render_distribution(report.topic_distribution_overall, report.total_mentions))
        w("\n\n")
        w("### Claim type (all mentions)\n\n")
        relabeled_overall = {paraphrase.get(k, k): v
                             for k, v in report.validity_claim_distribution_overall.items()}
        w(_render_distribution(relabeled_overall, report.total_mentions))
        w("\n\n")

    # Appendix
    w("## Appendix: Methodology Notes\n\n")
    w("- Sources this run: " + ", ".join(report.mentions_by_source.keys()) + "\n")
    w("- Classification: three independent LLM calls per mention "
      "(sentiment, topic, validity-claim) with structured-output enforcement\n")
    w("- Clustering: `(topic, sentiment_polarity)` buckets, min 2 mentions to "
      "form a cluster (v1; v1.1 swaps in embedding-based theming)\n")
    w("- Risk signal thresholds: fraud_spike ≥ 5 mentions; outage_cluster ≥ 4 "
      "mentions; sentiment_drift ≥ +10pts negative share WoW\n")

    return out.getvalue()


# -----------------------------------------------------------------------------
# Top-level entry: enrich + render
# -----------------------------------------------------------------------------

def finalize_and_render(
    llm: LLMAdapter,
    report: WeeklyReport,
    brand: str = "nfcu",
    domain_name: str = "financial_services",
) -> tuple[WeeklyReport, str]:
    """Generate exec summary via LLM, then render markdown. Returns the
    enriched report (now with executive_summary populated) and the markdown."""
    try:
        report.executive_summary = _generate_executive_summary(llm, report, brand=brand)
    except Exception as e:
        print(f"[report] exec summary generation failed: {e}")
        report.executive_summary = [
            f"(exec summary generation failed: {e}; see sections below)"
        ]
    return report, render_markdown(report, brand=brand, domain_name=domain_name)
