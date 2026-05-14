"""
Unit tests for SerperCollector. No real API calls — uses a stub for requests.

Run from repo root:
    python -m tests.test_serper_unit
"""

from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.serper import SerperCollector, classify_url
from src.schema import Source


def test_url_classification():
    cases = [
        ("https://www.reddit.com/r/personalfinance/comments/abc", Source.REDDIT),
        ("https://twitter.com/someone/status/123", Source.TWITTER),
        ("https://x.com/someone/status/123", Source.TWITTER),
        ("https://www.linkedin.com/posts/foo", Source.LINKEDIN),
        ("https://youtube.com/watch?v=xyz", Source.YOUTUBE),
        ("https://youtu.be/xyz", Source.YOUTUBE),
        ("https://www.trustpilot.com/review/nfcu", Source.TRUSTPILOT),
        ("https://www.bbb.org/us/va/vienna/profile/foo", Source.BBB),
        ("https://www.americanbanker.com/news/foo", Source.INDUSTRY_PRESS),
        ("https://www.cutoday.info/foo", Source.INDUSTRY_PRESS),
        ("https://random-blog.example.com/post", Source.GENERAL_WEB),
    ]
    for url, expected in cases:
        actual = classify_url(url)
        assert actual == expected, f"{url}: expected {expected}, got {actual}"
    print(f"[ok] URL classification: {len(cases)} cases pass")


def test_brand_or_clause():
    """v1.6: brand terms consolidate into a single OR-clause."""
    c = SerperCollector(
        api_key="fake", brand="nfcu",
        brand_terms=["navy federal", "navyfederal", "nfcu"],
    )
    clause = c._brand_or_clause()
    assert clause == '("navy federal" OR navyfederal OR nfcu)', \
        f"unexpected: {clause}"
    print(f"[ok] Brand OR-clause: {clause}")


def test_compose_query():
    """v1.6: query composition with optional intent + suffix."""
    c = SerperCollector(api_key="fake", brand="nfcu", brand_terms=["nfcu"])
    brand_or = "(nfcu)"

    q = c._compose_query(brand_or, suffix="site:reddit.com", intent_clause=None)
    assert q == "(nfcu) site:reddit.com", f"unexpected: {q}"

    q = c._compose_query(brand_or, suffix="site:reddit.com",
                         intent_clause="(complaint OR problem)")
    assert q == "(nfcu) (complaint OR problem) site:reddit.com", f"unexpected: {q}"

    q = c._compose_query(brand_or, suffix="", intent_clause=None)
    assert q == "(nfcu)", f"unexpected: {q}"
    print("[ok] Query composition: 3 cases pass")


def test_window_plan_dayrange_off():
    c = SerperCollector(
        api_key="fake", brand="nfcu", brand_terms=["nfcu"],
        time_window="qdr:w", dayrange=False,
    )
    plan = c._build_window_plan()
    assert plan == ["qdr:w"], f"unexpected: {plan}"
    print("[ok] Window plan, dayrange=off: single weekly window")


def test_window_plan_dayrange_on():
    c = SerperCollector(
        api_key="fake", brand="nfcu", brand_terms=["nfcu"],
        dayrange=True, dayrange_days=7,
    )
    plan = c._build_window_plan()
    assert len(plan) == 7, f"expected 7 windows, got {len(plan)}"
    for tbs in plan:
        assert tbs.startswith("cdr:1,cd_min:"), f"unexpected: {tbs}"
        assert "cd_max:" in tbs
    print(f"[ok] Window plan, dayrange=on: {len(plan)} day-specific windows")


def test_intent_queries_resolve_from_domain():
    c_fs = SerperCollector(
        api_key="fake", brand="nfcu", brand_terms=["nfcu"],
        domain="financial_services", intent_queries=True,
    )
    assert len(c_fs._intent_specs) > 0, "FS should have intents"
    intent_labels = [label for label, _ in c_fs._intent_specs]
    assert "complaint" in intent_labels
    assert "fee" in intent_labels

    c_he = SerperCollector(
        api_key="fake", brand="gmu", brand_terms=["gmu"],
        domain="higher_education", intent_queries=True,
    )
    assert len(c_he._intent_specs) > 0
    he_labels = [label for label, _ in c_he._intent_specs]
    assert "admissions" in he_labels
    assert "professor" in he_labels

    assert set(intent_labels) != set(he_labels), \
        "FS and HE should have different intent sets"
    print(f"[ok] Intent resolution: FS={len(intent_labels)}, HE={len(he_labels)} intents")


def test_intent_queries_disabled_means_no_specs():
    c = SerperCollector(
        api_key="fake", brand="nfcu", brand_terms=["nfcu"],
        domain="financial_services", intent_queries=False,
    )
    assert c._intent_specs == []
    print("[ok] Intent disabled: empty intent specs")


def test_collect_with_mocked_serper():
    fake_response = {
        "organic": [
            {
                "title": "NFCU charged me an overdraft fee",
                "link": "https://www.reddit.com/r/personalfinance/comments/abc",
                "snippet": "Posted 3 days ago. NFCU charged a $35 overdraft fee...",
                "date": "2026-05-06",
            },
            {
                "title": "Navy Federal Credit Union review",
                "link": "https://www.trustpilot.com/review/navyfederal",
                "snippet": "1-star review from a customer...",
                "date": "2026-05-07",
            },
            {
                "title": "Random unrelated page",
                "link": "https://example.com/something-else",
                "snippet": "no brand mention here",
                "date": "2026-05-05",
            },
            {
                "title": "NFCU charged me an overdraft fee",
                "link": "https://www.reddit.com/r/personalfinance/comments/abc",
                "snippet": "duplicate",
                "date": "2026-05-06",
            },
        ]
    }

    with patch("src.ingest.serper.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        c = SerperCollector(
            api_key="fake", brand="nfcu", brand_terms=["navy federal"],
            query_templates=[{"name": "test", "q_suffix": "", "endpoint": "search",
                              "intent_eligible": False, "dayrange_eligible": False}],
            rate_limit_sleep=0,
        )

        until = datetime.now(tz=timezone.utc) + timedelta(days=1)
        since = datetime(2026, 5, 1, tzinfo=timezone.utc)
        mentions = list(c.collect(since, until))

    assert len(mentions) == 3, f"expected 3, got {len(mentions)}"
    sources = {m.source for m in mentions}
    assert Source.REDDIT in sources
    assert Source.TRUSTPILOT in sources
    assert Source.GENERAL_WEB in sources
    assert all(m.brand == "nfcu" for m in mentions)
    assert all(m.id and m.url for m in mentions)
    assert all(m.intent is None for m in mentions), \
        "brand-only run should have no intent tagging"
    print(f"[ok] Mocked Serper collect: {len(mentions)} mentions, sources={sources}")


def test_intent_tagging_first_discovery_wins():
    """v1.6: when a URL surfaces from both brand-only and intent queries,
    first discovery wins. Brand-only runs first, so the URL should have
    intent=None even if it ALSO appears in the intent query."""
    brand_only_response = {
        "organic": [
            {"title": "Page 1", "link": "https://example.com/page1",
             "snippet": "...", "date": "2026-05-06"},
        ]
    }
    intent_response = {
        "organic": [
            {"title": "Page 1", "link": "https://example.com/page1",
             "snippet": "...", "date": "2026-05-06"},
            {"title": "Page 2 only via intent",
             "link": "https://example.com/page2",
             "snippet": "complaint...", "date": "2026-05-06"},
        ]
    }

    call_count = [0]
    def mock_post_fn(*args, **kwargs):
        call_count[0] += 1
        mock_resp = MagicMock()
        mock_resp.json.return_value = (
            brand_only_response if call_count[0] == 1 else intent_response
        )
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    with patch("src.ingest.serper.requests.post", side_effect=mock_post_fn):
        c = SerperCollector(
            api_key="fake", brand="nfcu", brand_terms=["nfcu"],
            domain="financial_services",
            query_templates=[{"name": "test", "q_suffix": "", "endpoint": "search",
                              "intent_eligible": True, "dayrange_eligible": False}],
            intent_queries=True,
            rate_limit_sleep=0,
        )

        until = datetime.now(tz=timezone.utc) + timedelta(days=1)
        since = datetime(2026, 5, 1, tzinfo=timezone.utc)
        mentions = list(c.collect(since, until))

    by_url = {m.url: m for m in mentions}
    page1 = by_url.get("https://example.com/page1")
    page2 = by_url.get("https://example.com/page2")
    assert page1 is not None
    assert page1.intent is None, \
        f"page1 was in brand-only first, intent should be None, got {page1.intent!r}"
    if page2:
        assert page2.intent is not None, \
            "page2 only came from intent query, should have an intent"
    print(f"[ok] Intent tagging first-discovery wins: "
          f"page1.intent={page1.intent}, page2.intent={page2.intent if page2 else 'absent'}")


def main():
    print("=" * 60)
    print("SerperCollector unit tests (v1.6)")
    print("=" * 60)
    test_url_classification()
    test_brand_or_clause()
    test_compose_query()
    test_window_plan_dayrange_off()
    test_window_plan_dayrange_on()
    test_intent_queries_resolve_from_domain()
    test_intent_queries_disabled_means_no_specs()
    test_collect_with_mocked_serper()
    test_intent_tagging_first_discovery_wins()
    print("=" * 60)
    print("All tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
