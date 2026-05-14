"""
Domain registry — per-domain topic schemas, severity weights, and classifier prompts.

Each domain (financial_services, higher_education, ...) declares:
- topics: list of allowed topic labels for this domain
- actionable_topics: subset that the report's "Actionable" section pulls from
- topic_weights: per-topic severity weight (0-1, used in cluster scoring)
- topic_definitions: short descriptions used to build the classifier prompt
- topic_signals: short keyword/concept hints for the classifier

The classifier always emits `unrelated`, `generic_discussion`, and `praise` —
these are universal. Domain-specific actionable categories vary.

Adding a new domain = adding a new entry to DOMAINS. Brands point at a domain
via brand_cfg['domain'] in config.yaml; if absent, defaults to financial_services
for backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    name: str
    description: str
    # Ordered list of all topic labels this domain uses (excluding the universals)
    domain_topics: list[str]
    # Topic labels treated as actionable for this domain (subset of all topics)
    actionable_topics: set[str]
    # Severity weight per topic (0-1). Universal topics get sensible defaults.
    topic_weights: dict[str, float]
    # Short definitions used to build the classifier prompt
    topic_definitions: dict[str, str]
    # The brand reference noun used in classifier prompts ("NFCU", "the university", etc.)
    brand_noun: str = "the brand"
    # v1.6: intent-keyword queries for the multi-query ingest strategy.
    # Each entry is (intent_label, OR-clause appended to the brand query).
    # Used by SerperCollector when intent_queries=True. Empty list = no intent
    # queries fire for this domain (collector falls back to brand-only).
    intent_queries: list[tuple[str, str]] = field(default_factory=list)


# Universal topics — always available across domains
UNIVERSAL_TOPICS = ["praise", "competitor_comparison", "generic_discussion", "unrelated"]

UNIVERSAL_DEFINITIONS = {
    "praise": "positive experience, recommendation, or gratitude expressed about the brand",
    "competitor_comparison": "comparing the brand to another similar institution or service",
    "generic_discussion": "the brand mentioned in passing or in neutral information sharing",
    "unrelated": "the search term refers to something else, or the mention is too thin to classify",
}


# ---------------------------------------------------------------------------
# Financial services
# ---------------------------------------------------------------------------

FINANCIAL_SERVICES = DomainConfig(
    name="financial_services",
    description="Banks, credit unions, fintech — consumer financial institutions.",
    brand_noun="the financial institution",
    domain_topics=[
        "fraud_claim",
        "outage_or_service_issue",
        "fee_dispute",
        "product_complaint",
        "recruitment_or_employment",
    ],
    actionable_topics={
        "fraud_claim",
        "outage_or_service_issue",
        "product_complaint",
        "fee_dispute",
        "competitor_comparison",
    },
    topic_weights={
        "fraud_claim": 1.0,
        "outage_or_service_issue": 0.9,
        "product_complaint": 0.8,
        "fee_dispute": 0.7,
        "competitor_comparison": 0.5,
        "recruitment_or_employment": 0.4,
        "praise": 0.3,
        "generic_discussion": 0.15,
        "unrelated": 0.0,
    },
    topic_definitions={
        "fraud_claim": (
            "user alleges fraud, scam, account compromise, or that the institution mishandled a fraud case"
        ),
        "outage_or_service_issue": (
            "app down, branch closed, can't log in, hold times, transfer broken, or other service-availability issues"
        ),
        "fee_dispute": (
            "overdraft fee, late fee, NSF, ATM fee, monthly maintenance fee complaint"
        ),
        "product_complaint": (
            "loan denial, rate change, credit card issue, account product issue (not fee or outage)"
        ),
        "recruitment_or_employment": (
            "hiring posts, working at the institution, employee experience"
        ),
    },
    intent_queries=[
        # Each tuple: (intent_label, additional OR-clause appended to brand query).
        # Mapped roughly onto the FS topic schema so the intent label can be
        # cross-validated against the topic classifier's output.
        ("complaint",  '(complaint OR complaints OR problem)'),
        ("review",     '(review OR reviews OR experience)'),
        ("fee",        '(fee OR fees OR charged OR "hidden fee")'),
        ("scam",       '(scam OR fraud OR "unauthorized charge")'),
        ("outage",     '(outage OR down OR "not working" OR error)'),
        ("denied",     '(denied OR declined OR rejected)'),
        ("vs",         '(vs OR "better than" OR "switching from")'),
        ("praise",     '(love OR "highly recommend" OR "best bank")'),
    ],
)


# ---------------------------------------------------------------------------
# Higher education
# ---------------------------------------------------------------------------

HIGHER_EDUCATION = DomainConfig(
    name="higher_education",
    description="Colleges and universities — student, alumni, faculty, and journalism discourse.",
    brand_noun="the university",
    domain_topics=[
        "admissions",
        "academic_experience",
        "campus_life",
        "athletics",
        "rankings_or_reputation",
        "alumni_or_career_outcomes",
        "safety_or_incident",
        "system_or_service_issue",
        "faculty_or_employment",
    ],
    actionable_topics={
        "safety_or_incident",
        "system_or_service_issue",
        "academic_experience",
        "rankings_or_reputation",
        "competitor_comparison",
    },
    topic_weights={
        "safety_or_incident": 1.0,
        "system_or_service_issue": 0.85,
        "academic_experience": 0.7,
        "rankings_or_reputation": 0.6,
        "admissions": 0.5,
        "competitor_comparison": 0.5,
        "campus_life": 0.4,
        "alumni_or_career_outcomes": 0.4,
        "athletics": 0.35,
        "faculty_or_employment": 0.3,
        "praise": 0.3,
        "generic_discussion": 0.15,
        "unrelated": 0.0,
    },
    topic_definitions={
        "admissions": (
            "decisions, acceptance rates, application questions, deferrals, financial aid, "
            "enrollment, prospective-student discussion"
        ),
        "academic_experience": (
            "course quality, teaching, advising, grade disputes, curriculum, rigor, workload, "
            "academic policies, classroom-facing complaints or praise about instruction"
        ),
        "campus_life": (
            "housing, dining, social life, clubs, residence-hall issues, student services, "
            "campus events, transportation"
        ),
        "athletics": (
            "game results, coaching, recruiting, athletic department, conference performance, "
            "NIL deals, athlete conduct"
        ),
        "rankings_or_reputation": (
            "US News and similar rankings, employer perception, prestige discussion, "
            "ROI commentary, comparisons of academic standing"
        ),
        "alumni_or_career_outcomes": (
            "job placement, graduate outcomes, alumni network, salary data, career services, "
            "internships, post-graduation paths"
        ),
        "safety_or_incident": (
            "campus crime, scandals, protests, leadership controversy, Title IX, hazing, "
            "discrimination claims, or other safety/integrity incidents"
        ),
        "system_or_service_issue": (
            "registration system down, financial aid portal broken, IT outage, dorm maintenance "
            "issue, dining hall closed, parking system problems"
        ),
        "faculty_or_employment": (
            "faculty hiring, staff jobs, working at the university, faculty governance, "
            "labor disputes, academic-employment posts"
        ),
    },
    intent_queries=[
        # Higher-ed intents — mapped onto the HE topic schema (admissions,
        # academic_experience, athletics, faculty_or_employment, etc.).
        ("complaint",   '(complaint OR complaints OR problem)'),
        ("admissions",  '(admitted OR rejected OR waitlist OR transfer OR acceptance)'),
        ("tuition",     '(tuition OR "financial aid" OR scholarship OR "student loan")'),
        ("dorm",        '(dorm OR housing OR roommate OR cafeteria OR "campus food")'),
        ("professor",   '(professor OR faculty OR advisor OR "office hours")'),
        ("rankings",    '(ranking OR rankings OR reputation OR US News)'),
        ("safety",      '(safety OR incident OR assault OR scandal OR investigation)'),
        ("alumni",      '(alumni OR "class of" OR graduate OR hired OR internship)'),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAINS: dict[str, DomainConfig] = {
    "financial_services": FINANCIAL_SERVICES,
    "higher_education": HIGHER_EDUCATION,
}


def get_domain(name: str) -> DomainConfig:
    """Look up a domain by name. Raises if unknown."""
    if name not in DOMAINS:
        raise ValueError(
            f"Unknown domain '{name}'. Available: {list(DOMAINS.keys())}"
        )
    return DOMAINS[name]


def all_topics_for_domain(name: str) -> list[str]:
    """Universal + domain-specific topics, the full enum the classifier may emit."""
    d = get_domain(name)
    # Universal first so the LLM sees them at the top of the list
    return UNIVERSAL_TOPICS + d.domain_topics


def all_definitions_for_domain(name: str) -> dict[str, str]:
    d = get_domain(name)
    return {**UNIVERSAL_DEFINITIONS, **d.topic_definitions}
