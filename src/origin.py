"""
Mention origin detection.

Determines whether a mention is customer voice, brand-owned messaging,
employee personal posts, journalism, or partner content. This matters
because VoC research cares about customer voice — the brand talking
about itself or partners promoting the brand are different signals
that belong in their own section.

Strategy: heuristic first (cheap, deterministic), LLM fallback only
when the heuristics don't make a confident call.

Per-brand configuration via BrandOriginRules:
- official_domains: URLs from these are BRAND_OWNED
- official_handles: authors matching these are BRAND_OWNED
- partner_domains: URLs from these are PARTNER (until proven otherwise)
- journalism_domains: URLs from these are JOURNALISM
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .llm import LLMAdapter
from .schema import Mention, MentionOrigin, OriginResult, Source


@dataclass
class BrandOriginRules:
    """Per-brand patterns for cheap origin detection."""
    brand_name: str
    # Official brand domains (corporate site, press release wires, brand-owned channels)
    official_domains: set[str] = field(default_factory=set)
    # Official author handles on social platforms
    official_handles: set[str] = field(default_factory=set)
    # Partner organizations frequently co-posting with the brand
    partner_domains: set[str] = field(default_factory=set)
    partner_handles: set[str] = field(default_factory=set)


# Rule sets for the brands we care about. Add more as we expand coverage.
DEFAULT_RULES: dict[str, BrandOriginRules] = {
    "nfcu": BrandOriginRules(
        brand_name="nfcu",
        official_domains={
            "navyfederal.org",
            "navyfederal.com",
            "nfcu.org",
            "businesswire.com",  # most BW results for NFCU are NFCU-issued press releases
            "prnewswire.com",
        },
        official_handles={
            "navy federal credit union",
            "navyfederal",
            "navy federal",
            "nfcu",
        },
        partner_domains={
            "bluestarfam.org",
            "unitedheroesleague.org",
        },
    ),
    "usaa": BrandOriginRules(
        brand_name="usaa",
        official_domains={
            "usaa.com",
            "communities.usaa.com",
        },
        official_handles={"usaa", "united services automobile association"},
    ),
    "penfed": BrandOriginRules(
        brand_name="penfed",
        official_domains={
            "penfed.org",
            "penfedfoundation.org",
        },
        official_handles={"penfed", "pentagon federal credit union", "penfed credit union"},
    ),
    "chase": BrandOriginRules(
        brand_name="chase",
        official_domains={"chase.com", "jpmorganchase.com"},
        official_handles={"chase", "jpmorgan chase", "chase bank"},
    ),
    "bofa": BrandOriginRules(
        brand_name="bofa",
        official_domains={"bankofamerica.com", "newsroom.bankofamerica.com"},
        official_handles={"bank of america", "bofa", "bofa newsroom"},
    ),

    # ---- Higher education ----
    "gmu": BrandOriginRules(
        brand_name="gmu",
        official_domains={
            "gmu.edu", "george.mason.edu",
            "gomason.com",  # athletics
            "today.gmu.edu", "news.gmu.edu",
        },
        official_handles={
            "george mason university", "gmu", "george mason",
            "mason nation",
        },
        partner_domains={"masonalumni.org"},
    ),
    "uva": BrandOriginRules(
        brand_name="uva",
        official_domains={
            "virginia.edu", "news.virginia.edu",
            "virginiasports.com",
            "alumni.virginia.edu",
        },
        official_handles={
            "university of virginia", "uva",
            "uva sports", "virginia cavaliers", "uva athletics",
        },
        partner_domains={"uvafoundation.com"},
    ),
    "vt": BrandOriginRules(
        brand_name="vt",
        official_domains={
            "vt.edu", "news.vt.edu",
            "hokiesports.com",
            "alumni.vt.edu",
        },
        official_handles={
            "virginia tech", "vt", "hokies", "virginia tech hokies",
        },
    ),
    "odu": BrandOriginRules(
        brand_name="odu",
        official_domains={
            "odu.edu", "ww2.odu.edu",
            "odusports.com",
        },
        official_handles={
            "old dominion university", "odu", "old dominion",
            "odu monarchs",
        },
    ),
    "georgetown": BrandOriginRules(
        brand_name="georgetown",
        official_domains={
            "georgetown.edu", "alumni.georgetown.edu",
            "guhoyas.com",
        },
        official_handles={
            "georgetown university", "georgetown", "hoyas",
            "georgetown hoyas",
        },
    ),
    "gwu": BrandOriginRules(
        brand_name="gwu",
        official_domains={
            "gwu.edu", "alumni.gwu.edu",
            "gwsports.com",
        },
        official_handles={
            "george washington university", "gwu", "gw",
            "the george washington university",
        },
    ),
}


# Journalism / industry-press domains — these are JOURNALISM regardless of brand
_JOURNALISM_DOMAINS = {
    # Financial press
    "americanbanker.com",
    "cutimes.com",
    "cutoday.info",
    "bankingdive.com",
    "wsj.com",
    "nytimes.com",
    "reuters.com",
    "bloomberg.com",
    "cnn.com",
    "nbcnews.com",
    "wfla.com",
    "wjla.com",
    "washingtonpost.com",
    "forbes.com",
    "marketwatch.com",
    "investopedia.com",
    "nerdwallet.com",
    "bankrate.com",
    "thepointsguy.com",
    "wallethub.com",
    # Higher-education press and rankings
    "chronicle.com",                # Chronicle of Higher Education
    "insidehighered.com",
    "thecrimson.com",               # Harvard Crimson but covers HE broadly
    "highereddive.com",
    "usnews.com",
    "niche.com",
    "collegefactual.com",
    "studentlifestyles.com",
    "cnbc.com",                     # often covers HE rankings/cost
    "axios.com",
    "politico.com",
    "wtop.com",                     # local DC-area news, relevant for VA/DC schools
    "wric.com",                     # Virginia
    "wavy.com",                     # Norfolk/ODU area
    "richmond.com",
    "roanoke.com",
    "cavalierdaily.com",            # UVA student paper
    "collegiatetimes.com",          # VT student paper
    "gwhatchet.com",                # GWU student paper
    "thehoya.com",                  # Georgetown student paper
    "fourthestate.gmu.edu",         # GMU student paper
    "maceandcrown.com",             # ODU student paper
}


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


def detect_origin_heuristic(
    mention: Mention,
    rules: BrandOriginRules,
) -> Optional[OriginResult]:
    """Return an origin if heuristic is confident, else None (defer to LLM)."""
    domain = _domain(mention.url)
    url_lower = mention.url.lower()
    handle_lower = (mention.author_handle or "").lower().strip()
    text_lower = (mention.text or "").lower()

    # 0) App-store-listing URLs — Google Play and Apple App Store
    # These pages are HYBRID: the brand authors the marketing description at
    # the top, but the bulk of the page is user reviews. Serper's snippets
    # virtually always pull from the user-review section, so these are
    # CUSTOMER voice for our purposes. The marketing-blurb edge case is rare
    # and would need to be caught separately if it shows up.
    is_play_store = ("play.google.com/store/apps" in url_lower)
    is_apple_app_store = ("apps.apple.com" in url_lower)
    if is_play_store or is_apple_app_store:
        store_name = "Google Play" if is_play_store else "Apple App Store"
        return OriginResult(
            label=MentionOrigin.CUSTOMER,
            confidence=0.85,
            rationale=(
                f"{store_name} listing URL — Serper snippets from these pages "
                f"surface user-review content rather than brand marketing copy."
            ),
        )

    # 1) Official brand domain — high confidence brand-owned
    if any(domain.endswith(d) for d in rules.official_domains):
        return OriginResult(
            label=MentionOrigin.BRAND_OWNED,
            confidence=0.95,
            rationale=f"Domain {domain} is on {rules.brand_name} official-domain list.",
        )

    # 2) Official handle — same
    if any(h in handle_lower for h in rules.official_handles):
        return OriginResult(
            label=MentionOrigin.BRAND_OWNED,
            confidence=0.9,
            rationale=f"Author handle '{mention.author_handle}' matches {rules.brand_name} official-handle list.",
        )

    # 3) LinkedIn /company/ URLs — corporate page
    if mention.source == Source.LINKEDIN and "/company/" in mention.url.lower():
        return OriginResult(
            label=MentionOrigin.BRAND_OWNED,
            confidence=0.85,
            rationale="LinkedIn company-page URL.",
        )

    # 4) Journalism domains — high confidence
    if any(domain.endswith(d) for d in _JOURNALISM_DOMAINS):
        return OriginResult(
            label=MentionOrigin.JOURNALISM,
            confidence=0.85,
            rationale=f"Domain {domain} is a recognized journalism source.",
        )

    # 5) Partner handles/domains
    if any(domain.endswith(d) for d in rules.partner_domains):
        return OriginResult(
            label=MentionOrigin.PARTNER,
            confidence=0.8,
            rationale=f"Domain {domain} is a recognized {rules.brand_name} partner.",
        )

    # 6) LinkedIn personal posts containing strong employee-positivity language —
    #    cheap heuristic. The LLM will validate borderline cases.
    #    For higher-ed: faculty/staff get tagged EMPLOYEE_PERSONAL but students
    #    do NOT — students are CUSTOMER voice (they consume the service).
    if mention.source == Source.LINKEDIN and "/posts/" in mention.url.lower():
        # Universal employee markers (works for any brand)
        employee_markers = (
            "grateful to work",
            "proud to be part",
            "joined the team",
            "my journey at",
            "i've been with",
            "team member at",
            "honored to work",
        )
        # Higher-ed-specific faculty/staff markers
        faculty_markers = (
            "joining the faculty",
            "joined the faculty",
            "my colleagues at",
            "as a faculty member",
            "professor of",
            "teaching at",
            "as faculty at",
            "thrilled to join",  # often paired with academic appointment language
            "delighted to announce my",
        )
        # Markers that indicate STUDENT voice — these signal we should NOT
        # mark as EMPLOYEE_PERSONAL even if other markers fire
        student_markers = (
            "class of 20",
            "freshman",
            "sophomore",
            "junior year",
            "senior year",
            "as a student at",
            "graduating from",
            "alumni",
            "alum of",
        )

        is_student = any(m in text_lower for m in student_markers)
        is_employee = any(m in text_lower for m in employee_markers + faculty_markers)

        if is_employee and not is_student:
            return OriginResult(
                label=MentionOrigin.EMPLOYEE_PERSONAL,
                confidence=0.7,
                rationale="LinkedIn personal post with employee/faculty positivity language.",
            )

    return None  # defer to LLM


# -----------------------------------------------------------------------------
# LLM fallback for ambiguous cases
# -----------------------------------------------------------------------------

ORIGIN_SYSTEM = """You determine the origin/voice of a brand mention. Output an OriginResult.

Categories:
- CUSTOMER: an actual customer or general user talking about the brand from the outside (complaints, praise, questions, opinions). For higher education, this includes students (current and prospective), alumni, and parents — they are consumers of the institution, not employees. The default for VoC research.
- BRAND_OWNED: the brand itself posting (corporate accounts, official press releases, marketing content authored by the company; for universities: the institution's official news, admissions office, communications office).
- EMPLOYEE_PERSONAL: a current employee's personal post about working at the brand. For universities, this includes faculty and staff personal posts ("thrilled to join the faculty," "my colleagues in the chemistry department"). Distinct from corporate posts.
- JOURNALISM: third-party reporting, news articles, industry analysis. For higher education, this includes student newspapers (Cavalier Daily, Hoya, Hatchet, etc.) — they operate as journalistic outlets.
- PARTNER: a partner organization promoting the brand (sponsorships, co-branded events, foundations the brand supports, alumni associations posting institutional news).

Rules:
- A customer complaint or praise on Reddit is CUSTOMER, even if it includes factual claims about the brand.
- A press release on a wire service (BusinessWire, PRNewswire) is BRAND_OWNED — the brand authored it, the wire just distributed it.
- A LinkedIn post by a named individual saying "grateful to work at [brand]" is EMPLOYEE_PERSONAL.
- A LinkedIn post by the company's official account is BRAND_OWNED.
- A news article in American Banker, Inside Higher Ed, or Chronicle of Higher Education is JOURNALISM.
- A Trustpilot review is CUSTOMER.
- An app-store listing page (Google Play, Apple App Store) is CUSTOMER when the snippet shows review content (user complaints, ratings, "I", "the app", frustration with updates). These pages are technically run by the brand but the searchable content is user reviews. Only tag BRAND_OWNED if the snippet is unambiguously the brand's marketing description ("Bank easy with our mobile app...").
- A Reddit post in r/[University] complaining about the registrar is CUSTOMER (student voice).
- A YouTube video tutorial about how the brand's products work is usually CUSTOMER (third-party content creator) unless explicitly produced by the brand.
- For higher education: a student's LinkedIn post about being accepted, graduating, or studying somewhere is CUSTOMER — students are not employees.
- When uncertain between CUSTOMER and JOURNALISM, look at whether the speaker is reporting facts (JOURNALISM) or expressing personal experience/opinion (CUSTOMER)."""


def detect_origin_llm(llm: LLMAdapter, mention: Mention) -> OriginResult:
    user = (
        f"Mention from {mention.source.value} ({mention.url}):\n"
        f"Author: {mention.author_handle}\n"
        f"Title: {mention.title or '(none)'}\n"
        f"Text: {mention.text[:1500]}\n\n"
        f"Classify the origin."
    )
    return llm.classify_structured(ORIGIN_SYSTEM, user, OriginResult)


def detect_origin(
    mention: Mention,
    rules: BrandOriginRules,
    llm: LLMAdapter,
) -> OriginResult:
    """Heuristic first, LLM fallback. Always returns an OriginResult."""
    heuristic = detect_origin_heuristic(mention, rules)
    if heuristic is not None:
        return heuristic
    try:
        return detect_origin_llm(llm, mention)
    except Exception as e:
        # On LLM failure, default to UNKNOWN — better than dropping the mention
        return OriginResult(
            label=MentionOrigin.UNKNOWN,
            confidence=0.0,
            rationale=f"Heuristic ambiguous, LLM origin call failed: {str(e)[:200]}",
        )
