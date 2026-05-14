"""
Classification stage. Three classifiers per mention:

1. Sentiment — five-level + intensity + rationale
2. Topic — coarse business-relevant category
3. Validity claim — Habermasian typology

Each runs as a separate LLM call with strict structured output. We parallelize
across mentions (not across classifiers per mention) because the bottleneck
is mention volume, not per-mention latency.

The Habermasian classifier is the one to watch in v1 results. The typology
is theoretically clean but the boundary between TRUTH and SINCERITY blurs
in user-generated content ('they charged me $35' is both a factual claim
and a personal grievance). The rationale field is your debugging signal —
when distributions look weird, read the rationales.
"""

from __future__ import annotations
import concurrent.futures
from datetime import datetime, timezone
from typing import Callable

from .llm import LLMAdapter
from .schema import (
    Classification,
    Mention,
    SentimentResult,
    TopicResult,
    ValidityClaimResult,
)


# -----------------------------------------------------------------------------
# Prompts. Keep these terse and explicit. Versioning matters — when you tune
# a prompt, bump the suffix and store it on Classification (TODO v1.1).
# -----------------------------------------------------------------------------

SENTIMENT_SYSTEM = """You classify the sentiment of social-media mentions about a brand or institution.

Output a SentimentResult with:
- label: very_negative | negative | neutral | positive | very_positive
- intensity: float 0.0-1.0 (how strongly the sentiment is expressed)
- rationale: <=1000 chars, what in the text drove the call

Rules:
- Sarcasm and frustration count as negative even if phrased politely
- Neutral means the post discusses the brand without expressing feeling, not 'mildly positive'
- Intensity is independent of label; 'I HATE THEM' is very_negative, intensity 0.95"""


def _build_topic_prompt(domain_name: str, brand: str) -> str:
    """Build the topic classifier prompt for a specific domain."""
    from .domains import get_domain, all_definitions_for_domain, all_topics_for_domain
    d = get_domain(domain_name)
    topics = all_topics_for_domain(domain_name)
    definitions = all_definitions_for_domain(domain_name)

    lines = [
        f"You categorize social-media mentions about {d.brand_noun} (brand: {brand}).",
        "",
        "Output a TopicResult with:",
        f"- label: one of [{', '.join(topics)}]",
        "- confidence: 0.0-1.0",
        "- rationale: <=1000 chars, what signaled the category",
        "",
        "Definitions:",
    ]
    for label in topics:
        defn = definitions.get(label, "")
        lines.append(f"- {label}: {defn}")
    lines.extend([
        "",
        "Pick the dominant category. If multiple apply, pick the one driving the post.",
        "Use 'unrelated' only when the mention isn't actually about the brand.",
        "Use 'generic_discussion' for in-passing or neutral information sharing.",
    ])
    return "\n".join(lines)


VALIDITY_CLAIM_SYSTEM = """You classify the type of validity claim a social-media mention is making about a brand or institution, using Habermas's typology.

Output a ValidityClaimResult with:
- label: truth | rightness | sincerity | comprehensibility
- confidence: 0.0-1.0
- rationale: <=1000 chars

Definitions:
- TRUTH: claim about objective facts ('they charged me $35', 'the system went down at 2pm', news reports of events, factual descriptions of products/services)
- RIGHTNESS: normative claim about how the institution should act ('they shouldn't do this to military families', 'they ought to refund', moral/policy demands)
- SINCERITY: expression of personal experience or feeling ('I love my advisor', 'frustrated and exhausted', 'so grateful', personal testimony)
- COMPREHENSIBILITY: meta-discussion about meaning or understanding — RARE. Only use when the post is explicitly asking what something means, debating definitions, or questioning whether language is being used correctly. Examples that ARE comprehensibility: "what does 'pending' mean here?", "is this term used differently elsewhere?", "I don't understand the wording of this notice".

CRITICAL RULES:
- COMPREHENSIBILITY is RARE. Most posts are not meta-discussions about meaning. If you find yourself defaulting to COMPREHENSIBILITY because nothing else fits, you are wrong — pick the dominant claim from TRUTH, RIGHTNESS, or SINCERITY instead.
- News articles, press releases, and product descriptions are TRUTH (they assert facts about the world).
- Employee testimonials, customer complaints, and praise posts are SINCERITY (they assert personal experience).
- Demands, criticism of policy, and "they should" statements are RIGHTNESS.
- A factual complaint with emotional content ('they charged me $35 and I'm furious') is TRUTH if the factual assertion is what the speaker most needs accepted, SINCERITY if the emotional expression is. When in doubt between TRUTH and SINCERITY, pick based on what gets quoted if someone retells the post: "they charged him $35" → TRUTH; "he was furious" → SINCERITY.
- A post can mix claim types; pick the DOMINANT one — what the speaker most needs accepted for the post to land.
- If the text is too thin to classify (e.g., a navigation snippet, a page title alone), still pick TRUTH/SINCERITY/RIGHTNESS based on apparent genre, not COMPREHENSIBILITY."""


# -----------------------------------------------------------------------------
# Single-mention classifiers
# -----------------------------------------------------------------------------

def _classify_sentiment(llm: LLMAdapter, mention: Mention) -> SentimentResult:
    user = f"Mention text:\n---\n{mention.text}\n---\n\nClassify."
    return llm.classify_structured(SENTIMENT_SYSTEM, user, SentimentResult)


def _classify_topic(llm: LLMAdapter, mention: Mention) -> TopicResult:
    system = _build_topic_prompt(mention.domain, mention.brand)
    user = f"Mention text:\n---\n{mention.text}\n---\n\nCategorize."
    return llm.classify_structured(system, user, TopicResult)


def _classify_validity_claim(llm: LLMAdapter, mention: Mention) -> ValidityClaimResult:
    user = f"Mention text:\n---\n{mention.text}\n---\n\nClassify the validity claim."
    return llm.classify_structured(VALIDITY_CLAIM_SYSTEM, user, ValidityClaimResult)


def classify_mention(llm: LLMAdapter, mention: Mention, origin_rules=None) -> Classification:
    """Run all classifiers on a single mention. Sequential — parallelism
    is at the mention level, not the classifier level."""
    from .origin import detect_origin, DEFAULT_RULES

    sentiment = _classify_sentiment(llm, mention)
    topic = _classify_topic(llm, mention)
    validity = _classify_validity_claim(llm, mention)

    # Origin: heuristic first, LLM fallback. If no rules for this brand,
    # fall back to a no-op rule set that defers to LLM for everything.
    rules = origin_rules or DEFAULT_RULES.get(mention.brand)
    if rules is None:
        from .origin import BrandOriginRules
        rules = BrandOriginRules(brand_name=mention.brand)
    origin = detect_origin(mention, rules, llm)

    return Classification(
        mention_id=mention.id,
        sentiment=sentiment,
        topic=topic,
        validity_claim=validity,
        origin=origin,
        classified_at=datetime.now(tz=timezone.utc),
    )


# -----------------------------------------------------------------------------
# Batch
# -----------------------------------------------------------------------------

def classify_batch(
    llm: LLMAdapter,
    mentions: list[Mention],
    max_workers: int = 4,
    on_error: Callable[[Mention, Exception], None] | None = None,
) -> list[Classification]:
    """Classify many mentions concurrently. Errors don't stop the batch —
    failed mentions are skipped and reported via on_error.

    max_workers tuned conservatively. For Gemini Flash and Haiku you can push
    higher, but provider rate limits bite before throughput plateaus.
    """
    results: list[Classification] = []

    def _run(m: Mention) -> Classification | None:
        try:
            return classify_mention(llm, m)
        except Exception as e:
            if on_error:
                on_error(m, e)
            else:
                print(f"[classify] failed mention {m.id}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for cls in ex.map(_run, mentions):
            if cls is not None:
                results.append(cls)

    return results
