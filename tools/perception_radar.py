"""
Perception Radar — single-page HTML view of cohort positioning.

Reads from the SQLite store, computes per-brand perception-dimension scores
via `compare.compute_perception_dimensions`, and emits a self-contained HTML
file rendering a radar chart with real polygon vertices.

Usage:
    python -m tools.perception_radar --target nfcu \\
        --brands nfcu bofa chase usaa penfed \\
        --output-path reports/perception_radar_banks.html

Produces a single HTML file with embedded SVG radar plus horizontal
dimension bars showing the target brand's standing on each axis relative
to the peer cohort. No external dependencies; opens in any browser.
"""

from __future__ import annotations
import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compare import (
    compute_perception_dimensions,
    cohort_median_dimensions,
    PerceptionDimension,
)
from src.store import Store


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Radar canvas
CX, CY = 220, 220
MAX_R = 170


def hex_vertex(angle_deg: float, value: float) -> tuple[float, float]:
    """Vertex for a single axis at `angle_deg` (0 = top), with score `value` (0-1)."""
    angle_rad = math.radians(angle_deg)
    x = CX + value * MAX_R * math.sin(angle_rad)
    y = CY - value * MAX_R * math.cos(angle_rad)
    return x, y


def polygon_points(values: list[float], angles: list[float]) -> str:
    """Build the SVG `points` attribute string from per-axis values."""
    pts = []
    for v, a in zip(values, angles):
        x, y = hex_vertex(a, v)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def grid_hexagon(value: float) -> str:
    """Hexagon at a given value (used for radar gridlines)."""
    angles = [0, 60, 120, 180, 240, 300]
    return polygon_points([value] * 6, angles)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_radar_svg(
    target_brand: str,
    perceptions: dict[str, list[PerceptionDimension]],
    medians: list[tuple[str, float]],
) -> str:
    """Build the SVG block for the radar chart with real data."""
    target_perception = perceptions[target_brand]
    axes = [d.dimension for d in target_perception]

    # Six axes evenly spaced
    if len(axes) != 6:
        raise ValueError(
            f"Radar expects exactly 6 axes; got {len(axes)}: {axes}"
        )
    angles = [0, 60, 120, 180, 240, 300]

    target_values = [d.score for d in target_perception]
    target_polygon = polygon_points(target_values, angles)

    median_values = [m[1] for m in medians]
    median_polygon = polygon_points(median_values, angles)

    # Target vertex dots
    target_dots_svg = "\n        ".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1d4ed8" />'
        for x, y in (hex_vertex(a, v) for a, v in zip(angles, target_values))
    )

    # Axis labels — placement at ~r=200 (just outside the max grid hex)
    label_r = 200
    label_block_lines: list[str] = []
    for axis_name, angle_deg in zip(axes, angles):
        rad = math.radians(angle_deg)
        x = CX + label_r * math.sin(rad)
        y = CY - label_r * math.cos(rad)
        # Adjust label vertical offset slightly so they sit cleanly
        if angle_deg == 0:
            y -= 8  # top
        elif angle_deg == 180:
            y += 14  # bottom
        else:
            y += 4
        label_block_lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'font-family="system-ui, -apple-system, sans-serif" '
            f'font-size="11" font-weight="600" fill="#1f2937">'
            f'{axis_name.upper()}</text>'
        )
    label_block = "\n        ".join(label_block_lines)

    # 5 concentric grid hexagons at value=0.2, 0.4, 0.6, 0.8, 1.0
    grid_polys = "\n          ".join(
        f'<polygon points="{grid_hexagon(v)}" />'
        for v in [1.0, 0.8, 0.6, 0.4, 0.2]
    )

    # Axis spokes from center to each axis tip (at value=1.0)
    axis_lines = []
    for angle_deg in angles:
        x, y = hex_vertex(angle_deg, 1.0)
        axis_lines.append(
            f'<line x1="{CX}" y1="{CY}" x2="{x:.1f}" y2="{y:.1f}" />'
        )
    axis_lines_block = "\n          ".join(axis_lines)

    return f"""
    <svg class="radar-svg" viewBox="-30 -10 520 470" width="520" height="470"
         aria-label="Cohort perception radar">
      <g>

        <!-- Grid rings (5 concentric hexagons) -->
        <g stroke="#e2e8f0" fill="none" stroke-width="1">
          {grid_polys}
        </g>

        <!-- Axis spokes -->
        <g stroke="#cbd5e1" stroke-width="1">
          {axis_lines_block}
        </g>

        <!-- Cohort median polygon (dashed gray) -->
        <polygon points="{median_polygon}"
                fill="rgba(100, 116, 139, 0.10)"
                stroke="#64748b" stroke-width="1.5"
                stroke-dasharray="5,4" />

        <!-- Target brand polygon -->
        <polygon points="{target_polygon}"
                fill="rgba(29, 78, 216, 0.20)"
                stroke="#1d4ed8" stroke-width="2.5" />

        <!-- Target vertex dots -->
        {target_dots_svg}

        <!-- Axis labels -->
        {label_block}

      </g>
    </svg>
"""


def render_dimension_bars(
    target_brand: str,
    perceptions: dict[str, list[PerceptionDimension]],
) -> str:
    """Build the horizontal dimension-bar block."""
    target_perception = perceptions[target_brand]
    peer_brands = [b for b in perceptions if b != target_brand]

    rows = []
    for axis_idx, target_dim in enumerate(target_perception):
        axis_name = target_dim.dimension
        target_score = target_dim.score
        peer_scores = [perceptions[p][axis_idx].score for p in peer_brands]

        # Cohort median (including target)
        all_scores = sorted([target_score] + peer_scores)
        n = len(all_scores)
        if n % 2 == 1:
            median = all_scores[n // 2]
        else:
            median = (all_scores[n // 2 - 1] + all_scores[n // 2]) / 2

        # Verdict
        if target_score > median + 0.15:
            verdict_class = "good"
            verdict_text = "Above cohort"
        elif target_score < median - 0.15:
            verdict_class = "bad"
            verdict_text = "Below cohort"
        elif abs(target_score - median) <= 0.05:
            verdict_class = ""
            verdict_text = "At cohort median"
        elif target_score < median:
            verdict_class = "warn"
            verdict_text = "Slightly below cohort"
        else:
            verdict_class = ""
            verdict_text = "Slightly above cohort"

        # Peer dot positions (left-to-right by score, 0-100%)
        peer_dots_html = "".join(
            f'<div class="dim-peer" style="left: {p*100:.1f}%;"></div>'
            for p in peer_scores
        )

        # Build the row. Use a fade indicator if target sample is low.
        confidence_marker = ""
        if target_dim.sample_n < 30:
            confidence_marker = f' <span class="low-n">(n={target_dim.sample_n})</span>'

        rows.append(f"""
        <div class="dim-row">
          <div>
            <div class="dim-name">{axis_name}{confidence_marker}</div>
            <div class="dim-desc">{target_dim.rationale}</div>
          </div>
          <div class="dim-bar-container">
            <div class="dim-bar-track"></div>
            <div class="dim-median" style="left: {median*100:.1f}%;"></div>
            {peer_dots_html}
            <div class="dim-target" style="left: {target_score*100:.1f}%;"></div>
            <div class="dim-bar-axis-labels">
              <span>Worse</span>
              <span>Better</span>
            </div>
          </div>
          <div class="dim-verdict {verdict_class}">
            {verdict_text}
          </div>
        </div>
""")
    return "".join(rows)


def render_reading_paragraph(
    target_brand: str,
    perceptions: dict[str, list[PerceptionDimension]],
    medians: list[tuple[str, float]],
) -> str:
    """Generate a one-paragraph reading of the target's standing."""
    target_perception = perceptions[target_brand]
    median_map = dict(medians)

    below = []
    above = []
    at = []
    for d in target_perception:
        m = median_map[d.dimension]
        if d.score < m - 0.15:
            below.append(d.dimension)
        elif d.score > m + 0.15:
            above.append(d.dimension)
        else:
            at.append(d.dimension)

    parts: list[str] = []
    target_name = target_brand.upper()
    if below:
        parts.append(
            f"{target_name} stands below cohort on "
            f"{_format_list(below)}"
        )
    if above:
        verb = "stands above cohort on" if not below else "; above cohort on"
        parts.append(f"{verb} {_format_list(above)}")
    if at and not (below or above):
        parts.append(f"{target_name} sits at or near cohort median across all dimensions")
    if not parts:
        parts.append(f"{target_name} positioning is mixed relative to peers")

    return ". ".join(parts).strip() + "."


def _format_list(items: list[str]) -> str:
    """Comma-separated list with 'and' before last item."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0].lower()
    return ", ".join(i.lower() for i in items[:-1]) + ", and " + items[-1].lower()


def render_full_html(
    target_brand: str,
    perceptions: dict[str, list[PerceptionDimension]],
    medians: list[tuple[str, float]],
    week_label: str,
    cohort_size: int,
) -> str:
    """Assemble the complete self-contained HTML page."""
    radar_svg = render_radar_svg(target_brand, perceptions, medians)
    dim_bars = render_dimension_bars(target_brand, perceptions)
    reading = render_reading_paragraph(target_brand, perceptions, medians)

    target_perception = perceptions[target_brand]
    sample_n = target_perception[0].sample_n

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{target_brand.upper()} — Cohort Perception Standing — Week of {week_label}</title>
<style>
  :root {{
    --ink: #0f172a;
    --ink-2: #1f2937;
    --ink-3: #475569;
    --ink-muted: #64748b;
    --rule: #e2e8f0;
    --rule-soft: #f1f5f9;
    --bg: #f8fafc;
    --card: #ffffff;
    --accent: #1d4ed8;
    --warn: #b45309;
    --good: #15803d;
    --bad: #b91c1c;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{ max-width: 1080px; margin: 0 auto; padding: 36px 28px 80px; }}

  header.page-head {{
    border-bottom: 2px solid var(--ink);
    padding-bottom: 18px;
    margin-bottom: 28px;
  }}
  .page-kicker {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin-bottom: 6px;
  }}
  .page-title {{
    font-size: 26px;
    font-weight: 600;
    margin: 0 0 6px;
    letter-spacing: -0.3px;
  }}
  .page-subtitle {{
    font-size: 14px;
    color: var(--ink-muted);
    margin: 0;
  }}

  .section-header {{
    margin: 36px 0 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--rule);
  }}
  .section-header h2 {{
    margin: 0;
    font-size: 19px;
    font-weight: 600;
    letter-spacing: -0.2px;
  }}

  .radar-wrap {{
    background: var(--card);
    border: 1px solid var(--rule);
    padding: 28px 28px 24px;
    margin: 0 0 18px;
  }}
  .radar-grid {{
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 32px;
    align-items: center;
  }}
  .radar-svg-container {{
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  .radar-svg {{
    max-width: 100%;
    height: auto;
  }}
  .radar-legend h4 {{
    font-size: 12px;
    font-weight: 600;
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 14px;
  }}
  .radar-legend-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 13.5px;
    color: var(--ink-2);
  }}
  .legend-swatch {{
    width: 24px;
    height: 12px;
    border-radius: 2px;
  }}
  .legend-swatch.target {{
    background: rgba(29, 78, 216, 0.20);
    border: 2px solid var(--accent);
  }}
  .legend-swatch.cohort {{
    background: rgba(100, 116, 139, 0.10);
    border: 2px dashed var(--ink-muted);
  }}
  .radar-reading {{
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--rule);
    font-size: 13.5px;
    color: var(--ink-2);
    line-height: 1.55;
  }}

  /* Dimension bars */
  .dim-row {{
    background: var(--card);
    border: 1px solid var(--rule);
    padding: 16px 22px;
    margin-bottom: 8px;
    display: grid;
    grid-template-columns: 220px 1fr 140px;
    gap: 22px;
    align-items: center;
  }}
  .dim-name {{
    font-size: 14px;
    font-weight: 600;
    color: var(--ink);
  }}
  .dim-name .low-n {{
    font-size: 11px;
    font-weight: 400;
    color: var(--warn);
  }}
  .dim-desc {{
    font-size: 11.5px;
    color: var(--ink-muted);
    font-weight: 400;
    margin-top: 4px;
    line-height: 1.4;
  }}
  .dim-bar-container {{
    position: relative;
    height: 34px;
  }}
  .dim-bar-track {{
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    left: 0;
    right: 0;
    height: 8px;
    background: var(--rule-soft);
    border-radius: 4px;
  }}
  .dim-bar-axis-labels {{
    position: absolute;
    bottom: -4px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: space-between;
    font-size: 9.5px;
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .dim-median {{
    position: absolute;
    top: 6px;
    bottom: 6px;
    width: 2px;
    background: var(--ink-muted);
  }}
  .dim-median::after {{
    content: "median";
    position: absolute;
    top: -13px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }}
  .dim-peer {{
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--ink-muted);
    opacity: 0.5;
    border: 2px solid var(--card);
  }}
  .dim-target {{
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--accent);
    border: 3px solid var(--card);
    box-shadow: 0 0 0 1.5px var(--accent);
    z-index: 2;
  }}
  .dim-verdict {{
    font-size: 12px;
    color: var(--ink-3);
    text-align: right;
    font-weight: 500;
  }}
  .dim-verdict.bad {{ color: var(--bad); font-weight: 600; }}
  .dim-verdict.warn {{ color: var(--warn); font-weight: 600; }}
  .dim-verdict.good {{ color: var(--good); font-weight: 600; }}

  .footer-note {{
    margin-top: 36px;
    padding-top: 18px;
    border-top: 1px solid var(--rule);
    font-size: 12px;
    color: var(--ink-muted);
    line-height: 1.6;
  }}

  @media (max-width: 880px) {{
    .radar-grid {{ grid-template-columns: 1fr; }}
    .dim-row {{ grid-template-columns: 1fr; gap: 10px; }}
    .dim-verdict {{ text-align: left; }}
  }}
</style>
</head>
<body>
<div class="container">

  <header class="page-head">
    <div class="page-kicker">Cohort Perception Standing · Week of {week_label}</div>
    <h1 class="page-title">{target_brand.upper()}</h1>
    <p class="page-subtitle">Position relative to {cohort_size - 1} peer institutions across six perception dimensions. {sample_n} customer-voice mentions back the {target_brand.upper()} scoring this week.</p>
  </header>

  <div class="section-header">
    <h2>Six-axis view</h2>
  </div>

  <div class="radar-wrap">
    <div class="radar-grid">
      <div class="radar-svg-container">
        {radar_svg}
      </div>
      <div class="radar-legend">
        <h4>This week's standing</h4>
        <div class="radar-legend-row">
          <span class="legend-swatch target"></span>
          <span>{target_brand.upper()}</span>
        </div>
        <div class="radar-legend-row">
          <span class="legend-swatch cohort"></span>
          <span>Peer cohort median ({cohort_size} brands)</span>
        </div>
        <div class="radar-reading">
          {reading}
        </div>
      </div>
    </div>
  </div>

  <div class="section-header">
    <h2>Per-dimension detail</h2>
  </div>

  {dim_bars}

  <div class="footer-note">
    <p>Scores are normalized cohort-relative: 0 maps to the worst score across the {cohort_size} brands on that dimension, 1 maps to the best. Cohort median is the median computed across all {cohort_size} brands. Each dimension's underlying measurement is shown in the per-dimension description.</p>
  </div>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target brand")
    parser.add_argument("--brands", nargs="*", help="Cohort brands (default: all in store)")
    parser.add_argument("--week", help="ISO date for the week start (default: most recent)")
    parser.add_argument("--store-path", default="data/mentions.db")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

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

    if args.target not in brands_to_load:
        print(f"[perception_radar] Target '{args.target}' must be in brand list; "
              f"got {brands_to_load}")
        return

    reports = {}
    for brand in brands_to_load:
        report = store.get_prior_report(cutoff, brand=brand)
        if report is None:
            print(f"[perception_radar] No report for brand={brand}; skipping.")
            continue
        reports[brand] = report

    if args.target not in reports:
        print(f"[perception_radar] No report for target '{args.target}'. Run weekly first.")
        return
    if len(reports) < 2:
        print(f"[perception_radar] Need at least 2 brands for cohort comparison; "
              f"got {len(reports)}.")
        return

    perceptions = compute_perception_dimensions(reports)
    medians = cohort_median_dimensions(perceptions)

    sample_report = reports[args.target]
    week_label = sample_report.week_start.date().isoformat()

    html_content = render_full_html(
        target_brand=args.target,
        perceptions=perceptions,
        medians=medians,
        week_label=week_label,
        cohort_size=len(reports),
    )

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = (Path(__file__).parent.parent / "reports" /
                       f"perception_radar_{args.target}_{week_label}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    print(f"[perception_radar] Written to {output_path}")
    print(f"[perception_radar] Target: {args.target}; cohort: {list(reports.keys())}")
    print(f"[perception_radar] Open: file://{output_path.resolve()}")

    store.close()


if __name__ == "__main__":
    main()