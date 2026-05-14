"""
HTML report renderer.

Produces a single self-contained HTML file with:
- Top-level tabs: Comparison, then one tab per brand
- Per-brand sub-tabs: Overview / Clusters / Mentions / Methodology
- Progressive disclosure on clusters (default 3 examples, expand to all,
  expand-individual-mention to see full text + classifier rationales)
- Filter chips and search box per brand
- No external CSS/JS dependencies — opens locally, attachable to email,
  pasteable into Slack as a file.

Design direction: editorial / briefing-grade. Restrained palette, distinctive
typography (Fraunces serif headers + JetBrains Mono for data), generous
whitespace. Matches the seriousness of the content — this is for executives,
researchers, and ops people who need to scan fast and dig deep.

Caller is responsible for assembling the data dict; see render_html() at
the bottom.
"""

from __future__ import annotations
import html
import json
import re
from datetime import datetime
from pathlib import Path

from .schema import (
    Classification,
    Mention,
    MentionOrigin,
    Sentiment,
    Topic,
    WeeklyReport,
)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def assemble_html_data(
    brand_data: dict[str, dict],
    comparison_md: str | None = None,
) -> dict:
    """Build the JSON-serializable data structure the template expects.

    brand_data: {brand_name: {
        'report': WeeklyReport,
        'mentions': list[Mention],
        'classifications': list[Classification],
    }}
    """
    out = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "brands": {},
        "comparison_md": comparison_md or "",
    }

    for brand, payload in brand_data.items():
        report = payload["report"]
        mentions = payload["mentions"]
        classifications = payload["classifications"]

        cls_by_id = {c.mention_id: c for c in classifications}

        # Mentions enriched with their classifications, JSON-serializable
        mentions_data = []
        for m in mentions:
            c = cls_by_id.get(m.id)
            if c is None:
                continue
            mentions_data.append({
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

        out["brands"][brand] = {
            "report": report.model_dump(mode="json"),
            "mentions": mentions_data,
        }

    return out


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice-of-Customer Report — {{TITLE}}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #faf8f3;
  --bg-card: #ffffff;
  --bg-subtle: #f0ece2;
  --ink: #1a1a1a;
  --ink-soft: #4a4a4a;
  --ink-mute: #8a8a82;
  --rule: #d8d2c4;
  --rule-soft: #e8e3d6;
  --accent: #8b2e2e;
  --accent-soft: #c47d7d;
  --pos: #2d5a3d;
  --neg: #8b2e2e;
  --neu: #6a665b;
  --warn: #a85a1c;
  --info: #2d4a6a;
  --shadow: 0 1px 2px rgba(26,26,26,0.04), 0 4px 12px rgba(26,26,26,0.04);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.55;
  font-size: 15px;
  font-feature-settings: 'ss01', 'cv11';
}

h1, h2, h3, h4 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin: 0 0 0.5em;
  color: var(--ink);
}
h1 { font-size: 2.4rem; line-height: 1.1; font-weight: 400; letter-spacing: -0.02em; }
h2 { font-size: 1.5rem; line-height: 1.2; margin-top: 1.6em; }
h3 { font-size: 1.15rem; line-height: 1.3; margin-top: 1.4em; }
h4 { font-size: 1rem; line-height: 1.4; font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: 0; }

a { color: var(--accent); text-decoration: none; border-bottom: 1px solid var(--rule); transition: border-color 120ms; }
a:hover { border-color: var(--accent); }

code, .mono { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.92em; }

/* Layout */
.shell { max-width: 1180px; margin: 0 auto; padding: 0 32px; }
header.top {
  border-bottom: 1px solid var(--rule);
  padding: 28px 0 24px;
  margin-bottom: 24px;
}
header.top .meta {
  display: flex; gap: 24px; align-items: baseline; flex-wrap: wrap;
  font-size: 0.8rem; color: var(--ink-mute); font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
}
header.top .title-row { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
header.top h1 { margin: 0; }
header.top .subtitle { color: var(--ink-soft); font-style: italic; font-family: 'Fraunces', serif; }

/* Brand tabs (primary nav) */
.brand-tabs {
  display: flex; gap: 0; border-bottom: 2px solid var(--rule);
  margin: 0 -32px 32px; padding: 0 32px;
  overflow-x: auto;
}
.brand-tab {
  padding: 14px 22px;
  background: none; border: none;
  cursor: pointer;
  font: inherit; font-size: 0.95rem;
  color: var(--ink-mute);
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  white-space: nowrap;
  transition: color 120ms, border-color 120ms;
  text-transform: uppercase; letter-spacing: 0.06em;
  font-family: 'JetBrains Mono', monospace;
}
.brand-tab:hover { color: var(--ink); }
.brand-tab.active {
  color: var(--ink); border-bottom-color: var(--accent);
  font-weight: 500;
}

/* Section sub-tabs (secondary nav inside a brand) */
.section-tabs {
  display: flex; gap: 4px; margin-bottom: 28px;
  background: var(--bg-subtle); border-radius: 6px; padding: 4px;
  width: fit-content;
}
.section-tab {
  padding: 8px 16px; border: none; background: transparent;
  font: inherit; font-size: 0.85rem;
  color: var(--ink-soft); cursor: pointer;
  border-radius: 4px; transition: all 120ms;
}
.section-tab:hover { color: var(--ink); }
.section-tab.active { background: var(--bg-card); color: var(--ink); box-shadow: var(--shadow); }

/* Cards & layout grids */
.card {
  background: var(--bg-card);
  border: 1px solid var(--rule-soft);
  border-radius: 6px;
  padding: 20px 24px;
  margin-bottom: 16px;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
@media (max-width: 760px) { .grid-2, .grid-4 { grid-template-columns: 1fr; } }

/* Headline stat block */
.stat {
  background: var(--bg-card);
  border: 1px solid var(--rule-soft);
  border-radius: 6px;
  padding: 18px 20px;
}
.stat .label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--ink-mute); font-family: 'JetBrains Mono', monospace;
  margin-bottom: 6px;
}
.stat .value {
  font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 400;
  line-height: 1.1; letter-spacing: -0.01em;
}
.stat .sub {
  font-size: 0.8rem; color: var(--ink-mute); margin-top: 4px;
}

/* Bullets / exec summary */
.exec-summary { font-size: 1.05rem; font-family: 'Fraunces', serif; line-height: 1.65; }
.exec-summary ul { padding-left: 0; list-style: none; }
.exec-summary li {
  position: relative; padding-left: 32px; margin-bottom: 12px;
}
.exec-summary li::before {
  content: ''; position: absolute; left: 0; top: 0.7em;
  width: 16px; height: 1px; background: var(--accent);
}
.exec-summary-list {
  font-family: 'Fraunces', serif; font-size: 1.02rem; line-height: 1.65;
  padding-left: 0; list-style: none; margin: 8px 0;
}
.exec-summary-list li {
  position: relative; padding-left: 32px; margin-bottom: 12px;
}
.exec-summary-list li::before {
  content: ''; position: absolute; left: 0; top: 0.7em;
  width: 16px; height: 1px; background: var(--accent);
}
.exec-summary-list strong { font-family: 'Inter', sans-serif; font-weight: 600; }

/* Tables */
table {
  width: 100%; border-collapse: collapse; font-size: 0.9rem;
  margin: 12px 0;
}
th, td {
  padding: 10px 12px; text-align: left;
  border-bottom: 1px solid var(--rule-soft);
}
th {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-mute); font-weight: 500; font-family: 'JetBrains Mono', monospace;
  background: var(--bg-subtle);
  border-bottom: 2px solid var(--rule);
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: var(--bg-subtle); }

/* Cluster cards */
.cluster {
  border: 1px solid var(--rule-soft); border-radius: 6px;
  margin-bottom: 16px; overflow: hidden;
  background: var(--bg-card);
}
.cluster-header {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 16px 20px; cursor: pointer;
  background: var(--bg-card); transition: background 120ms;
}
.cluster-header:hover { background: var(--bg-subtle); }
.cluster.expanded .cluster-header { border-bottom: 1px solid var(--rule-soft); }
.cluster-title {
  font-family: 'Fraunces', serif; font-size: 1.1rem; font-weight: 500;
}
.cluster-meta {
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
  color: var(--ink-mute);
}
.cluster-body { display: none; padding: 16px 20px; }
.cluster.expanded .cluster-body { display: block; }
.cluster-actionable .cluster-title { color: var(--accent); }

/* Severity bars */
.severity {
  display: inline-block; height: 6px; width: 60px;
  background: var(--bg-subtle); border-radius: 3px; vertical-align: middle;
  margin-left: 8px; position: relative; overflow: hidden;
}
.severity::after {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  background: var(--accent);
}
.severity.s10::after { width: 10%; } .severity.s20::after { width: 20%; }
.severity.s30::after { width: 30%; } .severity.s40::after { width: 40%; }
.severity.s50::after { width: 50%; } .severity.s60::after { width: 60%; }
.severity.s70::after { width: 70%; } .severity.s80::after { width: 80%; }
.severity.s90::after { width: 90%; } .severity.s100::after { width: 100%; }

/* Mention items */
.mention {
  border: 1px solid var(--rule-soft); border-radius: 4px;
  margin-bottom: 8px; background: var(--bg-card);
  font-size: 0.92rem;
}
.mention-summary {
  padding: 10px 14px; cursor: pointer;
  display: flex; align-items: baseline; gap: 12px;
}
.mention-summary:hover { background: var(--bg-subtle); }
.mention.expanded .mention-summary { border-bottom: 1px solid var(--rule-soft); }
.mention-source {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-mute); flex-shrink: 0;
  min-width: 80px;
}
.mention-text {
  flex: 1; color: var(--ink-soft); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.mention.expanded .mention-text { white-space: normal; color: var(--ink); }
.mention-tags { display: flex; gap: 4px; flex-shrink: 0; }
.tag {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  padding: 2px 6px; border-radius: 3px;
  background: var(--bg-subtle); color: var(--ink-mute);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.tag.s-very_negative, .tag.s-negative { background: #f5d8d8; color: var(--neg); }
.tag.s-positive, .tag.s-very_positive { background: #d8e5dc; color: var(--pos); }
.tag.t-fraud_claim, .tag.t-outage_or_service_issue,
.tag.t-product_complaint, .tag.t-fee_dispute { background: #f5e0c8; color: var(--warn); }
.tag.o-customer { background: #d8e5dc; color: var(--pos); }
.tag.o-brand_owned, .tag.o-employee_personal { background: #e0d8e8; color: var(--info); }

.mention-detail { display: none; padding: 12px 14px; }
.mention.expanded .mention-detail { display: block; }
.mention-detail .full-text {
  white-space: pre-wrap; padding: 12px; background: var(--bg-subtle);
  border-radius: 4px; font-size: 0.88rem; margin-bottom: 12px;
  line-height: 1.55;
}
.mention-detail .classifier-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
}
@media (max-width: 700px) { .mention-detail .classifier-grid { grid-template-columns: 1fr; } }
.classifier-block {
  padding: 10px 12px; background: var(--bg-subtle); border-radius: 4px;
  font-size: 0.85rem;
}
.classifier-block .cl-label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-mute); margin-bottom: 4px;
}
.classifier-block .cl-value { font-weight: 500; margin-bottom: 4px; }
.classifier-block .cl-rationale {
  color: var(--ink-soft); font-size: 0.85rem; line-height: 1.5;
}

.mention-detail .source-link {
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
  margin-top: 12px; word-break: break-all;
}

/* Filter chips */
.filter-bar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin-bottom: 20px; padding: 12px 16px;
  background: var(--bg-subtle); border-radius: 6px;
}
.filter-bar input[type="search"] {
  flex: 1; min-width: 200px;
  padding: 6px 10px; font: inherit; font-size: 0.88rem;
  border: 1px solid var(--rule); border-radius: 4px;
  background: var(--bg-card); color: var(--ink);
}
.chip-group { display: flex; gap: 4px; flex-wrap: wrap; }
.chip-group .label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--ink-mute); padding: 4px 6px;
}
.chip {
  padding: 4px 10px; font-size: 0.8rem;
  background: var(--bg-card); border: 1px solid var(--rule);
  border-radius: 12px; cursor: pointer; user-select: none;
  transition: all 120ms;
}
.chip:hover { border-color: var(--ink-soft); }
.chip.active { background: var(--ink); color: var(--bg); border-color: var(--ink); }

/* Misc */
.note { font-size: 0.85rem; color: var(--ink-soft); font-style: italic; }
.warn-marker { color: var(--warn); margin-left: 4px; }
.hidden { display: none !important; }
.empty {
  padding: 40px; text-align: center; color: var(--ink-mute);
  font-style: italic; font-family: 'Fraunces', serif;
}

/* Print */
@media print {
  body { font-size: 11pt; }
  .brand-tabs, .section-tabs, .filter-bar { display: none; }
  .panel { display: block !important; page-break-after: always; }
  .cluster, .mention { break-inside: avoid; }
  .cluster-body, .mention-detail { display: block !important; }
}
</style>
</head>
<body>
<div class="shell">

<header class="top">
  <div class="meta">
    <span>Voice-of-Customer Monitor</span>
    <span>Generated {{GENERATED_AT}}</span>
  </div>
  <div class="title-row">
    <h1>Weekly Brand Mentions Report</h1>
    <span class="subtitle">{{TITLE}}</span>
  </div>
</header>

<nav class="brand-tabs" id="brand-tabs">
  <button class="brand-tab active" data-brand="__comparison__">Comparison</button>
  {{BRAND_TAB_BUTTONS}}
</nav>

<main id="panels">
  <section class="panel active" data-brand="__comparison__">
    {{COMPARISON_PANEL}}
  </section>
  {{BRAND_PANELS}}
</main>

</div>

<script>
const DATA = {{DATA_JSON}};

// Tab switching
document.querySelectorAll('.brand-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const brand = btn.dataset.brand;
    document.querySelectorAll('.brand-tab').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('[data-brand]').forEach(p => {
      if (p.tagName === 'SECTION') {
        p.classList.toggle('active', p.dataset.brand === brand);
        p.classList.toggle('hidden', p.dataset.brand !== brand);
      }
    });
  });
});

// Section sub-tabs
document.querySelectorAll('.section-tabs').forEach(group => {
  group.querySelectorAll('.section-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const section = btn.dataset.section;
      const panel = btn.closest('.panel');
      panel.querySelectorAll('.section-tab').forEach(b => b.classList.toggle('active', b === btn));
      panel.querySelectorAll('[data-section]').forEach(s => {
        if (s.tagName === 'DIV' && s.classList.contains('section-content')) {
          s.classList.toggle('hidden', s.dataset.section !== section);
        }
      });
    });
  });
});

// Cluster expand/collapse
document.querySelectorAll('.cluster-header').forEach(h => {
  h.addEventListener('click', () => h.parentElement.classList.toggle('expanded'));
});

// Mention expand/collapse
document.querySelectorAll('.mention-summary').forEach(s => {
  s.addEventListener('click', () => s.parentElement.classList.toggle('expanded'));
});

// Filter / search
document.querySelectorAll('[data-filter-bar]').forEach(bar => {
  const brand = bar.dataset.filterBar;
  const panel = bar.closest('.panel');
  const searchInput = bar.querySelector('input[type="search"]');
  const chips = bar.querySelectorAll('.chip');
  const filters = { source: null, sentiment: null, topic: null, origin: null };

  function apply() {
    const q = (searchInput.value || '').toLowerCase().trim();
    const mentions = panel.querySelectorAll('[data-section="mentions"] .mention');
    let visible = 0;
    mentions.forEach(el => {
      const m = JSON.parse(el.dataset.mention);
      let show = true;
      if (filters.source && m.source !== filters.source) show = false;
      if (filters.sentiment && m.sentiment.label !== filters.sentiment) show = false;
      if (filters.topic && m.topic.label !== filters.topic) show = false;
      if (filters.origin && m.origin.label !== filters.origin) show = false;
      if (q) {
        const blob = (m.text + ' ' + m.title + ' ' + m.author).toLowerCase();
        if (blob.indexOf(q) === -1) show = false;
      }
      el.classList.toggle('hidden', !show);
      if (show) visible++;
    });
    const counter = panel.querySelector('[data-mention-count]');
    if (counter) counter.textContent = `${visible} of ${mentions.length} mentions`;
  }

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const dim = chip.dataset.dim;
      const val = chip.dataset.val;
      if (filters[dim] === val) {
        filters[dim] = null;
        chip.classList.remove('active');
      } else {
        filters[dim] = val;
        bar.querySelectorAll(`[data-dim="${dim}"]`).forEach(c =>
          c.classList.toggle('active', c.dataset.val === val)
        );
      }
      apply();
    });
  });
  searchInput.addEventListener('input', apply);
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers for building the markup
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _md_inline(s: str) -> str:
    """Render a fragment with minimal markdown (bold + italic) into safe HTML.

    The deterministic exec-read bullets contain `**bold**` markdown to highlight
    metric labels (e.g. "**Organic conversation**: ..."). HTML-escape first,
    then turn the escaped `**foo**` and `*foo*` back into <strong>/<em>.
    """
    escaped = html.escape(s or "", quote=True)
    # Bold: **text**
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    # Italic: *text* (single asterisks, only after bold pass)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def _severity_class(severity: float) -> str:
    """Map 0-1 severity to a stepped CSS class."""
    bucket = max(10, min(100, int(round(severity * 10) * 10)))
    return f"s{bucket}"


def _pct(num: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{num / total * 100:.1f}%"


# ---------------------------------------------------------------------------
# Per-brand panel
# ---------------------------------------------------------------------------

# Default ACTIONABLE_TOPICS — used when no domain is specified. Domain-specific
# brand panels read from the domain registry instead.
DEFAULT_ACTIONABLE_TOPICS = {
    "fraud_claim", "outage_or_service_issue", "product_complaint",
    "fee_dispute", "competitor_comparison",
}

# Topics that count as actionable ONLY when polarity is negative.
# Neutral discussion ("NFCU vs USAA, what do you think?") is conversation, not action.
# Negative discussion ("I'm switching to USAA because NFCU keeps...") IS action.
SOFT_ACTIONABLE_TOPICS = {
    "competitor_comparison",
    # higher-ed: rankings_or_reputation neutral chatter is just news; negative
    # framing is the actionable case
    "rankings_or_reputation",
    "alumni_or_career_outcomes",
}


def _actionable_topics_for_brand(brand_data: dict) -> set[str]:
    """Get the actionable-topic set for this brand from the domain registry."""
    domain_name = brand_data.get("domain", "financial_services")
    try:
        from .domains import get_domain
        return get_domain(domain_name).actionable_topics
    except (ValueError, ImportError):
        return DEFAULT_ACTIONABLE_TOPICS


def _is_actionable_cluster(cluster: dict, actionable_topics: set[str]) -> bool:
    """A cluster is actionable when its topic is in the actionable set AND
    (it's a hard-actionable topic OR its polarity is negative).

    `cluster` is a serialized Cluster dict. The `cluster_id` encodes polarity:
    `topic:neg`, `topic:neu`, `topic:pos`. We pull polarity from the suffix.
    """
    topic = cluster.get("topic_label") or cluster.get("topic")
    if topic not in actionable_topics:
        return False
    if topic not in SOFT_ACTIONABLE_TOPICS:
        return True  # hard-actionable: any polarity counts
    # Soft-actionable: only negative polarity counts
    cluster_id = cluster.get("cluster_id", "")
    return cluster_id.endswith(":neg")


def _render_brand_panel(brand: str, brand_data: dict) -> str:
    report = brand_data["report"]
    mentions = brand_data["mentions"]

    n = report["total_mentions"]
    n_customer = report.get("total_customer_mentions", 0) or \
                 report.get("origin_distribution", {}).get("customer", 0)
    sd = report["sentiment_distribution"]  # customer-voice in v1.3
    od = report["origin_distribution"]
    td = report["topic_distribution"]  # customer-voice
    vd = report["validity_claim_distribution"]  # customer-voice

    cust_n = n_customer
    neg_n = sd.get("negative", 0) + sd.get("very_negative", 0)
    risk_n = sum(td.get(t, 0) for t in
                 ("fraud_claim", "outage_or_service_issue", "product_complaint", "fee_dispute"))

    parts = [f'<section class="panel hidden" data-brand="{_esc(brand)}">']

    # Section sub-tabs
    parts.append("""
<div class="section-tabs">
  <button class="section-tab active" data-section="overview">Overview</button>
  <button class="section-tab" data-section="clusters">Clusters</button>
  <button class="section-tab" data-section="mentions">All mentions</button>
  <button class="section-tab" data-section="methodology">Methodology</button>
</div>
""")

    # ---- Overview ----
    parts.append('<div class="section-content" data-section="overview">')

    # Headline stat row — denominators are customer-voice for the actionable stats
    parts.append('<div class="grid-4">')
    for label, val, sub in [
        ("Total mentions", str(n),
         f"window: {report['week_start'][:10]} → {report['week_end'][:10]}"),
        ("Customer-voice mentions", str(cust_n),
         f"{_pct(cust_n, n)} of total"),
        ("Negative share (customer voice)", _pct(neg_n, cust_n),
         f"{neg_n} negative of {cust_n} customer-voice"),
        ("Risk-topic share (customer voice)", _pct(risk_n, cust_n),
         f"{risk_n} fraud/outage/complaint/fee of {cust_n}"),
    ]:
        parts.append(f"""
<div class="stat">
  <div class="label">{_esc(label)}</div>
  <div class="value">{_esc(val)}</div>
  <div class="sub">{_esc(sub)}</div>
</div>""")
    parts.append('</div>')

    # Exec summary
    if report.get("executive_summary"):
        parts.append('<h2>Executive Summary</h2>')
        parts.append('<div class="exec-summary card"><ul>')
        for bullet in report["executive_summary"]:
            parts.append(f'<li>{_esc(bullet)}</li>')
        parts.append('</ul></div>')

    # Risk signals
    parts.append('<h2>Risk Signals</h2>')
    if not report.get("risk_signals"):
        parts.append('<p class="note">No risk signals triggered this week.</p>')
    else:
        for sig in report["risk_signals"]:
            parts.append(f"""
<div class="card">
  <h4>{_esc(sig['kind'])} (severity {sig['severity']:.2f})</h4>
  <p><strong>Owner:</strong> <code>{_esc(sig['recommended_owner'])}</code> &middot;
     <strong>Confidence:</strong> {sig['confidence']*100:.0f}%</p>
  <p>{_esc(sig['recommended_action'])}</p>
</div>""")

    # Distributions (customer voice)
    parts.append('<h2>Distributions <span class="note" style="font-weight:normal">— customer voice only</span></h2>')
    parts.append(f'<p class="note">Computed from the {cust_n} customer-voice mentions, '
                 f'excluding brand-owned, journalism, employee-personal, and partner content.</p>')
    parts.append('<div class="grid-2">')
    parts.append('<div class="card"><h4>Sentiment</h4>' + _dist_table(sd, cust_n) + '</div>')
    parts.append('<div class="card"><h4>Topic</h4>' + _dist_table(td, cust_n) + '</div>')
    parts.append('<div class="card"><h4>Origin (full corpus)</h4>' + _dist_table(od, n) + '</div>')
    parts.append('<div class="card"><h4>Claim type (Habermasian)</h4>' +
                 _validity_table(vd, cust_n) + '</div>')
    parts.append('</div>')

    parts.append("""
<p class="note"><strong>Reading the claim types.</strong>
<em>Fact claims (truth)</em> assert facts; <em>norm claims (rightness)</em> assert what should be;
<em>experience claims (sincerity)</em> express personal feelings;
<em>meaning claims (comprehensibility)</em> ask about understanding.
Healthy customer-voice corpora blend all four.</p>
""")

    parts.append('</div>')  # /overview

    # ---- Clusters ----
    parts.append('<div class="section-content hidden" data-section="clusters">')
    parts.append('<h2>Actionable Clusters</h2>')

    # Build mention lookup for cluster expansion
    mentions_by_id = {m["id"]: m for m in mentions}

    actionable_topics_set = _actionable_topics_for_brand(brand_data)
    actionable = [c for c in report["clusters"]
                  if _is_actionable_cluster(c, actionable_topics_set)]
    context_clusters = [c for c in report["clusters"]
                        if not _is_actionable_cluster(c, actionable_topics_set)]

    if not actionable:
        parts.append('<p class="note">No actionable clusters this week — no fraud, outage, '
                     'complaint, fee, or competitor-comparison clusters reached the threshold.</p>')
    else:
        for cl in actionable:
            parts.append(_render_cluster(cl, mentions_by_id, actionable=True))

    parts.append('<h2>Context Clusters</h2>')
    parts.append('<p class="note">Lower-priority clusters: praise, employment, generic discussion. '
                 'Useful for ambient context, not action items.</p>')
    if not context_clusters:
        parts.append('<p class="note">None this week.</p>')
    else:
        for cl in context_clusters:
            parts.append(_render_cluster(cl, mentions_by_id, actionable=False))

    parts.append('</div>')  # /clusters

    # ---- All mentions ----
    parts.append('<div class="section-content hidden" data-section="mentions">')
    parts.append('<h2>All Mentions</h2>')

    # Filter bar
    sources = sorted({m["source"] for m in mentions})
    sentiments = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    topics = sorted({m["topic"]["label"] for m in mentions})
    origins = sorted({m["origin"]["label"] for m in mentions})

    parts.append(f'<div class="filter-bar" data-filter-bar="{_esc(brand)}">')
    parts.append('<input type="search" placeholder="Search mentions…">')
    for label, dim, vals in [
        ("Source", "source", sources),
        ("Sentiment", "sentiment", sentiments),
        ("Topic", "topic", topics),
        ("Origin", "origin", origins),
    ]:
        parts.append(f'<div class="chip-group"><span class="label">{_esc(label)}</span>')
        for v in vals:
            parts.append(f'<span class="chip" data-dim="{_esc(dim)}" data-val="{_esc(v)}">'
                         f'{_esc(v)}</span>')
        parts.append('</div>')
    parts.append('</div>')

    parts.append(f'<p class="note" data-mention-count>{len(mentions)} of {len(mentions)} mentions</p>')

    for m in mentions:
        parts.append(_render_mention(m))

    parts.append('</div>')  # /mentions

    # ---- Methodology ----
    parts.append('<div class="section-content hidden" data-section="methodology">')
    parts.append("""
<h2>Methodology</h2>
<div class="card">
<h4>Pipeline</h4>
<p>Mentions are discovered via Google search (Serper API) across Reddit, news,
LinkedIn, Trustpilot, BBB, YouTube, industry press, and general web. High-priority
mentions are enriched with full-text fetch. Each mention runs through four
classifiers in parallel: sentiment (5-level + intensity), topic (9 categories),
validity-claim (Habermasian: truth / rightness / sincerity / comprehensibility),
and origin (customer / brand_owned / employee_personal / journalism / partner).</p>

<h4>Customer-voice filter</h4>
<p>Clusters are built from mentions classified as customer voice only. Brand-owned,
employee-personal, journalism, and partner mentions appear in distributions but
are excluded from actionable cluster construction — these are signal about the
brand's own messaging or third-party coverage, not customer voice.</p>

<h4>Severity scoring</h4>
<p>Cluster severity = topic_weight × polarity_weight × volume_factor.
Topic weights: fraud_claim=1.0, outage_or_service_issue=0.9, product_complaint=0.8,
fee_dispute=0.7, competitor_comparison=0.5, recruitment_or_employment=0.4,
praise=0.3, generic_discussion=0.15. Polarity: negative=1.0, neutral=0.5, positive=0.4.</p>

<h4>Risk signal thresholds</h4>
<p>fraud_spike: ≥5 fraud_claim mentions. outage_cluster: ≥4
outage_or_service_issue mentions. sentiment_drift: negative-share rose ≥10pts WoW.</p>

<h4>Habermasian validity claims</h4>
<p>The framework distinguishes four kinds of claims people make in discourse:</p>
<ul>
  <li><strong>Fact claims</strong> (truth) — speaker asserts something objectively.</li>
  <li><strong>Norm claims</strong> (rightness) — speaker asserts what should be.</li>
  <li><strong>Experience claims</strong> (sincerity) — speaker asserts personal feelings.</li>
  <li><strong>Meaning claims</strong> (comprehensibility) — speaker asks about understanding.</li>
</ul>
<p>Brand corpora that skew heavily toward one type indicate discourse genre.
Fact-heavy = newsworthy/informational. Experience-heavy = relational/emotional.
Norm-heavy = active customer pushback. Meaning-heavy is rare and usually
indicates a classifier issue.</p>
</div>
""")
    parts.append('</div>')  # /methodology

    parts.append('</section>')
    return "\n".join(parts)


def _render_cluster(cl: dict, mentions_by_id: dict, actionable: bool) -> str:
    cls_actionable = "cluster-actionable" if actionable else ""
    sev_class = _severity_class(cl["severity"])
    topic_label = cl.get("topic_label") or cl["topic"]

    parts = [f'<div class="cluster {cls_actionable}">']
    parts.append(f"""
<div class="cluster-header">
  <div>
    <div class="cluster-title">{_esc(cl['theme'])}</div>
    <div class="cluster-meta">{len(cl['mention_ids'])} mentions &middot; severity
      <span class="severity {sev_class}"></span> {cl['severity']:.2f}
      &middot; topic <code>{_esc(topic_label)}</code>
    </div>
  </div>
  <div class="cluster-meta">click to expand</div>
</div>
""")

    parts.append('<div class="cluster-body">')

    # Render every mention in the cluster (full data, progressive disclosure per mention)
    for mid in cl["mention_ids"]:
        m = mentions_by_id.get(mid)
        if m:
            parts.append(_render_mention(m))

    parts.append('</div></div>')
    return "\n".join(parts)


def _render_mention(m: dict) -> str:
    sentiment_cls = f"s-{m['sentiment']['label']}"
    topic_cls = f"t-{m['topic']['label']}"
    origin_cls = f"o-{m['origin']['label']}"

    text_preview = m['title'] or (m['text'][:150] + ('...' if len(m['text']) > 150 else ''))

    # JSON for the filter system
    filter_data = {
        "source": m["source"],
        "title": m.get("title", ""),
        "text": m["text"][:500],  # truncated for filter purposes
        "author": m.get("author", ""),
        "sentiment": {"label": m["sentiment"]["label"]},
        "topic": {"label": m["topic"]["label"]},
        "origin": {"label": m["origin"]["label"]},
    }
    filter_json = html.escape(json.dumps(filter_data), quote=True)

    return f"""
<div class="mention" data-mention="{filter_json}">
  <div class="mention-summary">
    <span class="mention-source">{_esc(m['source'])}</span>
    <span class="mention-text">{_esc(text_preview)}</span>
    <span class="mention-tags">
      <span class="tag {sentiment_cls}">{_esc(m['sentiment']['label'])}</span>
      <span class="tag {topic_cls}">{_esc(m['topic']['label'])}</span>
      <span class="tag {origin_cls}">{_esc(m['origin']['label'])}</span>
    </span>
  </div>
  <div class="mention-detail">
    <div class="full-text">{_esc(m['text'])}</div>
    <div class="classifier-grid">
      <div class="classifier-block">
        <div class="cl-label">Sentiment</div>
        <div class="cl-value">{_esc(m['sentiment']['label'])} (intensity {m['sentiment']['intensity']:.2f})</div>
        <div class="cl-rationale">{_esc(m['sentiment']['rationale'])}</div>
      </div>
      <div class="classifier-block">
        <div class="cl-label">Topic</div>
        <div class="cl-value">{_esc(m['topic']['label'])} (conf {m['topic']['confidence']:.2f})</div>
        <div class="cl-rationale">{_esc(m['topic']['rationale'])}</div>
      </div>
      <div class="classifier-block">
        <div class="cl-label">Claim type (Habermasian)</div>
        <div class="cl-value">{_esc(m['validity']['label'])} (conf {m['validity']['confidence']:.2f})</div>
        <div class="cl-rationale">{_esc(m['validity']['rationale'])}</div>
      </div>
      <div class="classifier-block">
        <div class="cl-label">Origin</div>
        <div class="cl-value">{_esc(m['origin']['label'])} (conf {m['origin']['confidence']:.2f})</div>
        <div class="cl-rationale">{_esc(m['origin']['rationale'])}</div>
      </div>
    </div>
    <div class="source-link">
      <strong>{_esc(m.get('author', ''))} &middot; {m['timestamp'][:10]}</strong> &middot;
      <a href="{_esc(m['url'])}" target="_blank" rel="noopener">{_esc(m['url'])}</a>
    </div>
  </div>
</div>"""


def _dist_table(dist: dict, total: int) -> str:
    rows = sorted([(k, v) for k, v in dist.items() if v > 0],
                  key=lambda kv: -kv[1])
    if not rows:
        return '<p class="note">No data</p>'
    parts = ['<table>']
    for label, count in rows:
        parts.append(f'<tr><td>{_esc(label)}</td>'
                     f'<td class="num">{count}</td>'
                     f'<td class="num">{_pct(count, total)}</td></tr>')
    parts.append('</table>')
    return "\n".join(parts)


def _validity_table(vd: dict, total: int) -> str:
    paraphrase = {
        "truth": "fact claims",
        "rightness": "norm claims",
        "sincerity": "experience claims",
        "comprehensibility": "meaning claims",
    }
    rows = sorted([(k, v) for k, v in vd.items() if v > 0],
                  key=lambda kv: -kv[1])
    if not rows:
        return '<p class="note">No data</p>'
    parts = ['<table>']
    for label, count in rows:
        para = paraphrase.get(label, label)
        parts.append(f'<tr><td>{_esc(para)} <span class="mono" style="color:var(--ink-mute)">'
                     f'({_esc(label)})</span></td>'
                     f'<td class="num">{count}</td>'
                     f'<td class="num">{_pct(count, total)}</td></tr>')
    parts.append('</table>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Comparison panel (renders the comparison data inline; no markdown dependency)
# ---------------------------------------------------------------------------

def _render_comparison_panel(brand_data: dict[str, dict], target: str | None = None) -> str:
    """Render the cross-brand comparison panel.

    If `target` is provided and matches a brand, the panel becomes target-framed:
    - leads with an executive read narrating where the target stands
    - puts the target first in every table with a (target) marker
    - shows a "Where {target} stands" table with direction-aware judgments
    - surfaces cross-cohort risk signals prominently
    """
    if not brand_data:
        return '<p class="note">No data.</p>'

    # If target provided, reorder so target is first
    if target and target in brand_data:
        ordered = {target: brand_data[target]}
        for b, d in sorted(brand_data.items()):
            if b != target:
                ordered[b] = d
        brand_data = ordered

    parts = []

    # ---- Title ----
    if target and target in brand_data:
        parts.append(f'<h2>{_esc(target.upper())} — Cross-Brand Comparison</h2>')
        peers = [b for b in brand_data if b != target]
        parts.append(f'<p class="note">Compared against: {", ".join(_esc(p) for p in peers)}.</p>')
    else:
        parts.append('<h2>Cross-Brand Comparison</h2>')

    # ---- Target-framed analysis layer ----
    if target and target in brand_data:
        try:
            # Reconstruct WeeklyReport objects to feed the standings engine
            from .compare import compute_standings, deterministic_exec_read, interpret
            from .schema import WeeklyReport

            reports_obj: dict[str, WeeklyReport] = {}
            for b, d in brand_data.items():
                reports_obj[b] = WeeklyReport.model_validate(d["report"])

            standings = compute_standings(target, reports_obj)
            bullets = deterministic_exec_read(target, standings)

            # Executive read card
            parts.append('<div class="card">')
            parts.append('<h4>Executive Read</h4>')
            if bullets:
                parts.append('<ul class="exec-summary-list">')
                for b in bullets:
                    parts.append(f'<li>{_md_inline(b)}</li>')
                parts.append('</ul>')
            else:
                parts.append('<p class="note">Insufficient sample size or spread for confident '
                             'comparative claims this week.</p>')
            parts.append('</div>')

            # Standings table — target-centered, direction-aware judgments
            parts.append('<div class="card">')
            parts.append(f'<h4>Where {_esc(target.upper())} stands</h4>')
            parts.append('<p class="note">Per-metric rank within the cohort '
                         '(rank 1 = highest value), with the target\'s value, the cohort median, '
                         'and a direction-aware judgment.</p>')
            parts.append('<table>')
            parts.append('<tr><th>Metric</th><th>Direction</th>'
                         '<th class="num">Target</th><th class="num">Cohort median</th>'
                         '<th class="num">Rank</th><th>Judgment</th><th class="num">n</th></tr>')
            for s in standings:
                confident_marker = '' if s.confident else ' <span class="warn-marker">⚠</span>'
                judgment = interpret(s)
                direction_short = {
                    'higher_is_better': '↑ better',
                    'lower_is_better': '↓ better',
                    'neutral': '—',
                }.get(s.direction, s.direction)
                parts.append(
                    f'<tr><td>{_esc(s.metric)}{confident_marker}</td>'
                    f'<td>{_esc(direction_short)}</td>'
                    f'<td class="num">{s.target_value*100:.1f}%</td>'
                    f'<td class="num">{s.median*100:.1f}%</td>'
                    f'<td class="num">{s.rank}/{s.cohort_size}</td>'
                    f'<td>{_esc(judgment)}</td>'
                    f'<td class="num">{s.sample_n}</td></tr>'
                )
            parts.append('</table>')
            parts.append('<p class="note">Rank 1 means highest value of the metric. The '
                         '<strong>Direction</strong> column says whether higher or lower is better; '
                         'the <strong>Judgment</strong> column reads position-against-direction in '
                         'plain language. ⚠ marks metrics where the target\'s sample size is below 30; '
                         'rankings are unreliable.</p>')
            parts.append('</div>')
        except Exception as e:
            parts.append(f'<p class="note">Executive analysis unavailable: {_esc(str(e))}</p>')

    # ---- Risk signals across the cohort ----
    any_signals = any(d["report"].get("risk_signals") for d in brand_data.values())
    if any_signals:
        parts.append('<div class="card">')
        parts.append('<h4>Risk Signals This Week</h4>')
        parts.append('<p class="note">Signals that fired in any brand\'s pipeline this week, '
                     'with the recommended owner. These are the highest-priority items in the cohort.</p>')
        for brand, data in brand_data.items():
            sigs = data["report"].get("risk_signals", [])
            if not sigs:
                continue
            target_marker = ' <em>(target)</em>' if brand == target else ''
            for sig in sigs:
                parts.append(
                    f'<div style="margin: 12px 0; padding: 10px 14px; '
                    f'background: var(--bg-subtle); border-left: 3px solid var(--accent); border-radius: 4px;">'
                    f'<strong>{_esc(brand.upper())}{target_marker}</strong> &middot; '
                    f'<code>{_esc(sig["kind"])}</code> &middot; '
                    f'severity {sig["severity"]:.2f} &middot; '
                    f'owner <code>{_esc(sig["recommended_owner"])}</code><br>'
                    f'<span style="color: var(--ink-soft);">{_esc(sig["recommended_action"])}</span>'
                    f'</div>'
                )
        parts.append('</div>')

    # ---- Headline table ----
    parts.append('<div class="card"><h4>Headline</h4>')
    parts.append('<p class="note">Negative share and risk-topic share are computed against '
                 'customer-voice mentions only — they describe how customers talk about each brand, '
                 'not the full corpus including brand-owned posts and journalism.</p>')
    parts.append('<table>')
    parts.append('<tr><th>Brand</th><th class="num">Total</th>'
                 '<th class="num">Customer voice (n)</th>'
                 '<th class="num">Negative share</th>'
                 '<th class="num">Risk topics</th><th class="num">Risk signals</th></tr>')
    for brand, data in brand_data.items():
        r = data["report"]
        n = r["total_mentions"]
        cust_n = r.get("total_customer_mentions", 0) or \
                 r.get("origin_distribution", {}).get("customer", 0)
        warn = '<span class="warn-marker">⚠</span>' if cust_n < 20 else ''
        target_marker = ' <em>(target)</em>' if brand == target else ''
        neg = r["sentiment_distribution"].get("negative", 0) + \
              r["sentiment_distribution"].get("very_negative", 0)
        risk = sum(r["topic_distribution"].get(t, 0) for t in
                   ("fraud_claim", "outage_or_service_issue", "product_complaint", "fee_dispute"))
        parts.append(f'<tr><td><strong>{_esc(brand)}</strong>{warn}{target_marker}</td>'
                     f'<td class="num">{n}</td>'
                     f'<td class="num">{cust_n}</td>'
                     f'<td class="num">{_pct(neg, cust_n)} <span class="mono" style="color:var(--ink-mute)">({neg})</span></td>'
                     f'<td class="num">{_pct(risk, cust_n)} <span class="mono" style="color:var(--ink-mute)">({risk})</span></td>'
                     f'<td class="num">{len(r.get("risk_signals", []))}</td></tr>')
    parts.append('</table>')
    parts.append('<p class="note">⚠ = fewer than 20 customer-voice mentions; sample-size confidence is low.</p>')
    parts.append('</div>')

    # Habermas / claim types side-by-side
    parts.append('<div class="card"><h4>Claim Types (Habermasian, customer voice only)</h4>')
    parts.append("""<p class="note"><em>Fact claims</em> assert facts;
<em>norm claims</em> assert what should be;
<em>experience claims</em> express personal feelings;
<em>meaning claims</em> ask about understanding. Computed from customer-voice mentions only.</p>""")
    parts.append('<table><tr><th>Brand</th><th class="num">n</th><th class="num">Fact</th>'
                 '<th class="num">Norm</th><th class="num">Experience</th>'
                 '<th class="num">Meaning</th></tr>')
    for brand, data in brand_data.items():
        vd = data["report"]["validity_claim_distribution"]
        n = sum(vd.values()) or 1
        parts.append(f'<tr><td><strong>{_esc(brand)}</strong></td>'
                     f'<td class="num">{n}</td>'
                     f'<td class="num">{_pct(vd.get("truth", 0), n)}</td>'
                     f'<td class="num">{_pct(vd.get("rightness", 0), n)}</td>'
                     f'<td class="num">{_pct(vd.get("sincerity", 0), n)}</td>'
                     f'<td class="num">{_pct(vd.get("comprehensibility", 0), n)}</td></tr>')
    parts.append('</table></div>')

    # Origin mix
    parts.append('<div class="card"><h4>Origin Mix</h4>')
    parts.append("""<p class="note">How much of each brand's discoverable conversation
is the brand itself vs actual customer voice. High brand-owned share usually
means heavy corporate posting; high customer share means more organic
conversation; high journalism share means the brand is currently newsworthy.</p>""")
    parts.append('<table><tr><th>Brand</th><th class="num">Customer</th>'
                 '<th class="num">Brand-owned</th><th class="num">Employee</th>'
                 '<th class="num">Journalism</th><th class="num">Partner</th></tr>')
    for brand, data in brand_data.items():
        od = data["report"]["origin_distribution"]
        n = sum(od.values()) or 1
        parts.append(f'<tr><td><strong>{_esc(brand)}</strong></td>'
                     f'<td class="num">{_pct(od.get("customer", 0), n)}</td>'
                     f'<td class="num">{_pct(od.get("brand_owned", 0), n)}</td>'
                     f'<td class="num">{_pct(od.get("employee_personal", 0), n)}</td>'
                     f'<td class="num">{_pct(od.get("journalism", 0), n)}</td>'
                     f'<td class="num">{_pct(od.get("partner", 0), n)}</td></tr>')
    parts.append('</table></div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_html(
    brand_data: dict[str, dict],
    title: str = "",
    target: str | None = None,
) -> str:
    """Render the full self-contained HTML report.

    brand_data: same shape as assemble_html_data input.
    title: shown in the header (e.g. "Week of 2026-05-02")
    target: brand name to center the comparison panel around. The brand tabs
            put the target second (right after Comparison) and the comparison
            panel adds the target-framed analysis layer (executive read,
            "where {target} stands" table). If None, comparison panel is
            descriptive-only.
    """
    sample = next(iter(brand_data.values()), None)
    if sample and not title:
        ws = sample["report"]["week_start"][:10]
        title = f"Week of {ws}"

    # Reorder brand_data so target appears first in tabs (after Comparison)
    if target and target in brand_data:
        ordered = {target: brand_data[target]}
        for b, d in sorted(brand_data.items()):
            if b != target:
                ordered[b] = d
        brand_data = ordered

    brand_tabs = "\n".join(
        f'<button class="brand-tab" data-brand="{_esc(b)}">{_esc(b.upper())}</button>'
        for b in brand_data.keys()
    )

    brand_panels = "\n".join(_render_brand_panel(b, d) for b, d in brand_data.items())

    comparison_panel = _render_comparison_panel(brand_data, target=target)

    data_json = json.dumps({"brands": list(brand_data.keys())})

    out = (_TEMPLATE
           .replace("{{TITLE}}", _esc(title))
           .replace("{{GENERATED_AT}}", datetime.utcnow().isoformat(timespec="seconds") + "Z")
           .replace("{{BRAND_TAB_BUTTONS}}", brand_tabs)
           .replace("{{BRAND_PANELS}}", brand_panels)
           .replace("{{COMPARISON_PANEL}}", comparison_panel)
           .replace("{{DATA_JSON}}", data_json))

    return out
