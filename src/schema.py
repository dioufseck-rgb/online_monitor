"""
Schema contracts for the mentions pipeline.

Every stage of the pipeline reads and writes these types. Keeping them in
one file makes the data flow explicit and makes provider/source swaps
mechanical: a new source just needs to emit Mention records.

Discipline: structured output is enforced at every LLM boundary. We learned
this the hard way in cc-trading v4.1 — silent failures hide in unstructured
strings.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Source-side: what a collector emits
# -----------------------------------------------------------------------------

class Source(str, Enum):
    REDDIT = "reddit"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    NEWS = "news"
    YOUTUBE = "youtube"
    TRUSTPILOT = "trustpilot"
    BBB = "bbb"
    INDUSTRY_PRESS = "industry_press"
    YELP = "yelp"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    THREADS = "threads"
    FACEBOOK = "facebook"
    GENERAL_WEB = "general_web"


class Mention(BaseModel):
    """Normalized mention record. Every collector outputs this shape."""

    id: str = Field(..., description="Globally unique: {source}:{native_id}")
    source: Source
    brand: str = Field(..., description="Which brand this mention is about (nfcu, nike, etc.)")
    domain: str = Field(default="financial_services",
                        description="Domain registry key — controls topic schema and prompts")
    author_handle: str
    author_metadata: dict = Field(default_factory=dict)  # karma, follower count, etc.
    timestamp: datetime
    text: str  # full text if available, else snippet
    snippet: Optional[str] = None  # search-result snippet (Serper-sourced)
    full_text_fetched: bool = False  # True if text was enriched via web_fetch
    title: Optional[str] = None
    url: str
    parent_id: Optional[str] = None  # for threaded sources (Reddit, YouTube)
    engagement: dict = Field(default_factory=dict)  # upvotes, replies, etc.
    raw_payload: dict = Field(default_factory=dict)  # source-specific extras

    # v1.6: intent-driven query metadata. When the mention was surfaced via an
    # intent-keyword query (complaint/fee/outage/etc.), the intent is recorded
    # here. Used to (a) cross-validate the topic classifier (intent=outage
    # mentions should classify as outage_or_service_issue at high rates,
    # otherwise audit), and (b) potentially as a topic prior for the classifier.
    # First-discovery wins: if a mention surfaces from multiple queries, the
    # intent of the first query that found it is kept.
    intent: Optional[str] = Field(default=None,
                                  description="Intent keyword that surfaced this mention "
                                              "(e.g. 'complaint', 'fee', 'outage'). None if "
                                              "the mention came from a brand-only query.")


# -----------------------------------------------------------------------------
# Classification outputs — three parallel classifiers per mention
# -----------------------------------------------------------------------------

class Sentiment(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Topic(str, Enum):
    """Coarse business-relevant categories. Refine after seeing v1 output."""
    FRAUD_CLAIM = "fraud_claim"
    OUTAGE_OR_SERVICE_ISSUE = "outage_or_service_issue"
    FEE_DISPUTE = "fee_dispute"
    PRODUCT_COMPLAINT = "product_complaint"
    PRAISE = "praise"
    RECRUITMENT_OR_EMPLOYMENT = "recruitment_or_employment"
    COMPETITOR_COMPARISON = "competitor_comparison"
    GENERIC_DISCUSSION = "generic_discussion"
    UNRELATED = "unrelated"  # filter survivor that's actually noise


class ValidityClaim(str, Enum):
    """
    Habermasian validity-claim typology.

    - TRUTH: factual assertion about NFCU ('they charged me $35')
    - RIGHTNESS: normative claim ('NFCU shouldn't do X to military families')
    - SINCERITY: personal experience/feeling ('I love my NFCU rep')
    - COMPREHENSIBILITY: meta-discussion ('what does NFCU mean by...')
    """
    TRUTH = "truth"
    RIGHTNESS = "rightness"
    SINCERITY = "sincerity"
    COMPREHENSIBILITY = "comprehensibility"


class MentionOrigin(str, Enum):
    """Where the mention is coming from in terms of voice/perspective.

    For VoC research we care primarily about CUSTOMER voice. Brand-owned
    content (the company talking about itself) is filtered into a separate
    section — it's signal about the brand's own messaging, not what
    customers think.
    """
    CUSTOMER = "customer"                     # actual customer voice — what we care about
    BRAND_OWNED = "brand_owned"               # the brand itself posting (corporate, official)
    EMPLOYEE_PERSONAL = "employee_personal"   # employee personal post (LinkedIn "grateful to work here")
    JOURNALISM = "journalism"                 # third-party reporting
    PARTNER = "partner"                       # partner organizations (Blue Star Families, NHL etc.)
    UNKNOWN = "unknown"


class SentimentResult(BaseModel):
    label: Sentiment
    intensity: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=1000)


class TopicResult(BaseModel):
    label: str  # domain-specific topic label; validated against domain registry
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=1000)


class ValidityClaimResult(BaseModel):
    label: ValidityClaim
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=1000)


class OriginResult(BaseModel):
    label: MentionOrigin
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=1000)


class Classification(BaseModel):
    """All classifier outputs bound to one mention."""
    mention_id: str
    sentiment: SentimentResult
    topic: TopicResult
    validity_claim: ValidityClaimResult
    origin: OriginResult
    classified_at: datetime


# -----------------------------------------------------------------------------
# Aggregation outputs
# -----------------------------------------------------------------------------

class Cluster(BaseModel):
    """A group of mentions sharing topic + theme during the window."""
    cluster_id: str
    topic: Topic  # Legacy FS Topic enum; for non-FS domains, see topic_label
    topic_label: Optional[str] = None  # Real domain-specific topic label (preferred over topic.value)
    theme: str  # short LLM-generated label
    mention_ids: list[str]
    severity: float = Field(..., ge=0.0, le=1.0)  # volume * negativity * reach
    reach_estimate: int  # sum of engagement signals
    representative_examples: list[str]  # 2-3 mention texts (truncated)


class RiskSignal(BaseModel):
    """A cluster elevated to action-required status."""
    signal_id: str
    kind: str  # 'fraud_spike', 'outage_cluster', 'sentiment_drift', etc.
    cluster_id: str
    severity: float
    recommended_owner: str  # 'fraud_ops', 'service_desk', 'comms', 'product'
    recommended_action: str
    confidence: float


class WeeklyReport(BaseModel):
    """Top-level artifact. Everything in the markdown comes from this."""
    week_start: datetime
    week_end: datetime
    total_mentions: int
    total_customer_mentions: int = 0  # mentions classified as customer voice
    mentions_by_source: dict[str, int]
    # Primary (customer-voice) distributions — these drive the report
    sentiment_distribution: dict[str, int]
    topic_distribution: dict[str, int]
    validity_claim_distribution: dict[str, int]
    # Explicit customer aliases (same data, named for clarity)
    sentiment_distribution_customer: dict[str, int] = Field(default_factory=dict)
    topic_distribution_customer: dict[str, int] = Field(default_factory=dict)
    validity_claim_distribution_customer: dict[str, int] = Field(default_factory=dict)
    # Overall distributions across the full corpus — transparency
    sentiment_distribution_overall: dict[str, int] = Field(default_factory=dict)
    topic_distribution_overall: dict[str, int] = Field(default_factory=dict)
    validity_claim_distribution_overall: dict[str, int] = Field(default_factory=dict)
    origin_distribution: dict[str, int] = Field(default_factory=dict)
    wow_volume_delta: Optional[float] = None  # None for first run
    wow_sentiment_delta: Optional[float] = None
    clusters: list[Cluster]
    risk_signals: list[RiskSignal]
    executive_summary: list[str]  # 3-5 bullets, LLM-generated last