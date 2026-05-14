# Voice-of-Customer Monitor

Multi-brand, multi-domain monitoring pipeline for public mentions across the open web. Applies discourse-analysis methods (Habermasian validity-claim typology, sentiment, topic clustering, origin classification) to produce weekly reports, cross-brand comparisons, and a shareable HTML dashboard.

The methodology is brand-agnostic and domain-pluggable — each brand declares a domain (currently `financial_services` or `higher_education`) which controls the topic taxonomy, severity weights, classifier prompts, and risk-signal rules. Default config includes NFCU, USAA, PenFed, Chase, BofA (financial services) and GMU, UVA, Virginia Tech, ODU, Georgetown, GWU (higher education).

See `ETHICS.md` for data-source decisions and the reasoning behind them.

## Architecture

Six-stage pipeline, each stage with a clean interface:

1. **Ingest** — Serper-based multi-source discovery (Reddit, news, LinkedIn, Trustpilot, BBB, YouTube, industry press, general web)
2. **Filter** — relevance check + dedup
3. **Enrich** — full-text web fetch for high-priority mentions
4. **Classify** — four parallel classifiers per mention: sentiment, topic (domain-specific), validity-claim, origin
5. **Aggregate** — weekly window, customer-voice clusters, risk-signal detection (rules per domain)
6. **Report** — per-brand markdown, cross-brand comparison, HTML dashboard

Provider-agnostic LLM adapter: `gemini` (personal), `azure_openai` (enterprise), `anthropic` (parity testing). Switch via config; nothing else changes.

## Domain pluggability

Each brand declares a `domain` in config.yaml. The domain registry (`src/domains.py`) defines per-domain:

- **Topic schema** — categories the classifier can emit
- **Actionable topics** — subset that the report's actionable section pulls from
- **Severity weights** — per-topic weights used in cluster scoring
- **Risk-signal rules** — what topic + threshold triggers an action (and who owns it)
- **Brand-noun phrasing** — used in classifier prompts ("the financial institution" vs. "the university")

Universal topics (`praise`, `competitor_comparison`, `generic_discussion`, `unrelated`) work across domains. The pipeline is otherwise domain-agnostic — to add healthcare or retail domains, add a new `DomainConfig` entry; nothing in the orchestrator, classifier, aggregator, or HTML renderer changes.

## What customer-voice filtering does

Mentions are classified into one of five origins:

- `customer` — actual customer or general user voice (the actionable signal). For higher education, this includes students, prospective students, alumni, and parents — they are consumers of the institution.
- `brand_owned` — the brand itself posting (corporate accounts, official press releases, university communications offices)
- `employee_personal` — employee personal posts. For higher education, faculty and staff personal posts.
- `journalism` — third-party reporting and industry press. For higher education, this includes student newspapers (Cavalier Daily, Hatchet, Hoya, etc.), which operate as journalistic outlets.
- `partner` — partner organizations promoting the brand

**All actionable distributions (sentiment, topic, validity-claim) are computed from customer voice only.** Brand-owned and journalism mentions describe corpus shape, not customer signal — including them in the denominator suppresses negative-share metrics in proportion to how heavily a brand posts about itself. Reports clearly label customer-voice scope; full-corpus distributions are kept as a transparency appendix.

## Layout

```
src/
  schema.py            # Pydantic contracts: Mention, Classification, WeeklyReport, etc.
  domains.py           # Domain registry: per-domain topics, weights, prompts, risk rules
  llm.py               # Provider-agnostic adapter with retry-with-backoff
  ingest/
    base.py            # Collector interface
    serper.py          # Serper-based multi-source collector (primary)
    reddit.py          # PRAW-based (only if non-commercial Reddit API approved)
  filter.py            # Brand-parametric relevance + dedup
  enrich.py            # WebFetchEnricher — full-text fetching for high-priority mentions
  origin.py            # Heuristic-first origin detection with LLM fallback
  classify.py          # Four classifiers: sentiment, topic (domain-aware), validity-claim, origin
  aggregate.py         # Customer-voice clustering, severity scoring (domain weights), risk-signal detection
  report.py            # Per-brand markdown rendering (domain-aware actionable filtering)
  compare.py           # Cross-brand comparison + multi-week trend reports
  html_report.py       # HTML dashboard renderer (single self-contained file)
  html_dashboard.py    # CLI entry point for HTML generation
  store.py             # SQLite persistence (multi-brand, schema migrations)
  weekly.py            # Top-level orchestrator
config/
  config.example.yaml  # Multi-brand, multi-source config template
data/
  mentions.db          # SQLite store (gitignored)
reports/
  {brand}_{date}.md          # Per-brand markdown reports
  comparison_{date}.md       # Cross-brand comparison snapshot
  trend_{N}w_{date}.md       # Multi-week trend report
  dashboard_{date}.html      # Shareable HTML dashboard
tools/
  audit_classifications.py   # Filter-based audit (read mentions by topic/sentiment/origin)
  audit_validity_claims.py   # Stratified-sample audit for human verification
tests/
  test_pipeline_smoke.py     # End-to-end with mocked LLM
  test_serper_unit.py        # Serper collector unit tests
ETHICS.md                    # Data-source decisions and reasoning
.env.example                 # Required environment variables (real values go in .env)
```

## Run

```bash
pip install -r requirements.txt

# 1. Set up credentials
cp .env.example .env
nano .env                              # paste GEMINI_API_KEY, SERPER_API_KEY

# 2. Set up config
cp config/config.example.yaml config/config.yaml

# 3. Dry run first — see volume before spending tokens
python -m src.weekly --brand nfcu --days 2 --dry-run

# 4. If volume looks right, do a small classification slice
python -m src.weekly --brand nfcu --days 2 --max-classify 25

# 5. Full week, single brand
python -m src.weekly --brand nfcu

# 6. All brands (NFCU, USAA, PenFed, Chase, BofA per default config)
python -m src.weekly

# 7. Cross-brand comparison snapshot (current week)
python -m src.compare

# 8. Multi-week trend (after multiple weekly runs accumulate)
python -m src.compare --weeks 4

# 9. Shareable HTML dashboard
python -m src.html_dashboard
```

The dashboard reads from existing classifications — no LLM cost. Open `reports/dashboard_*.html` in any browser; single self-contained file (~600KB), attachable to email or Slack.

## Auditing the classifier

Two tools for verifying the classifier is doing real work:

```bash
# Filter-based — read mentions matching a specific topic/origin/sentiment
python -m tools.audit_classifications --brand nfcu --origin brand_owned
python -m tools.audit_classifications --brand nfcu --topics fraud_claim product_complaint

# Stratified sampling — N random mentions per (brand, validity-claim) cell
python -m tools.audit_validity_claims --n 5
python -m tools.audit_validity_claims --labels sincerity --n 10
```

The validity-claim audit is especially useful before publishing comparative claims — it lets you verify that classifier rationales are coherent on real text.

## Credentials & secret hygiene

- `.env` holds real API keys; gitignored, must never be committed
- `config/config.yaml` references env vars via `${VAR_NAME}` — no secrets
- `config/config.example.yaml` is the committed template
- `.env.example` is the committed template for required env vars
- `.env` is loaded automatically by the orchestrator at startup
- Existing shell env vars take precedence over `.env`

If a real key is committed: rotate it immediately at the provider, force-push if not yet propagated, and treat it as leaked regardless. Scrapers index public commits within minutes.

## Sources covered (via Serper)

- Reddit (via Google index)
- News (Google News)
- LinkedIn public posts
- Trustpilot
- Better Business Bureau
- YouTube discovery
- Industry press (American Banker, CU Today, CU Times, Banking Dive)
- General web

## Versioning notes

- v1.0 — single-brand Reddit pipeline (now superseded)
- v1.1 — Serper-based multi-source ingestion, retry-with-backoff
- v1.2 — origin classification (customer-voice filtering), competitor brands, plain-language Habermas
- v1.3 — customer-voice denominators throughout, sample-size annotations, multi-week trends, HTML dashboard, validity-claim audit tooling
- v1.4 — domain-pluggable topic schemas (financial_services + higher_education), 6 universities added, faculty/staff vs student origin disambiguation, domain-specific risk-signal rules
- v1.5 — target-brand framing in comparison reports and HTML dashboard (executive read, "where target stands" table, risk signals surfaced cross-cohort), Play Store / App Store URLs handled correctly as customer voice, error-page detection in filter, soft-actionable topics (competitor_comparison neutral demoted to context), exec-summary prompt tightened for risk-signal contradiction

## Known issues

These are flagged for next iteration:

- **Same-week WoW phantom deltas**: re-running the same brand on the same week creates a fake "WoW change" against the earlier same-week run. Should compare against last calendar week, not last report.
- **`.env` not auto-loaded by compare.py**: `python -m src.compare --llm` requires manual env export. Fix is to mirror weekly.py's auto-load.
- **"Generic discussion is largest" bullet still occasionally appears**: prompt rule forbids it but LLM treats it as soft suggestion; may need stronger negative example in the prompt or a post-generation filter.
