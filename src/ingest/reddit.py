"""
Reddit collector using PRAW.

Strategy:
- Sitewide search for query terms (catches everything)
- Plus targeted scans of subreddits where NFCU comes up frequently:
  r/personalfinance, r/CreditUnions, r/MilitaryFinance, r/USMC,
  r/navy, r/AirForce, r/army
- Pulls submissions AND top-level comments matching terms

Window semantics: Reddit's search isn't perfectly time-bounded, so we
over-fetch and filter by created_utc. Idempotency is at the orchestrator
level (the store dedups by mention id).

Rate limits: PRAW handles backoff. With OAuth credentials you get
~600 requests / 10 min, comfortable for weekly cadence.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterator

from ..schema import Mention, Source
from .base import Collector


# Search terms — kept broad on purpose. Filter.py drops the noise.
SEARCH_TERMS = [
    "navy federal",
    "navyfederal",
    "nfcu",
]

# Subreddits where NFCU lives. Add more as you discover them.
TARGET_SUBREDDITS = [
    "personalfinance",
    "CreditUnions",
    "MilitaryFinance",
    "USMC",
    "navy",
    "AirForce",
    "army",
    "Veterans",
    "MilitarySpouse",
]


class RedditCollector(Collector):
    """PRAW-based Reddit collector."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        brand: str = "nfcu",
        username: str | None = None,
        password: str | None = None,
        max_per_query: int = 250,
    ):
        try:
            import praw
        except ImportError as e:
            raise RuntimeError("praw not installed. pip install praw") from e

        kwargs = dict(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        if username and password:
            kwargs["username"] = username
            kwargs["password"] = password

        self._reddit = praw.Reddit(**kwargs)
        self._reddit.read_only = True
        self._max_per_query = max_per_query
        self._brand = brand

    @property
    def source_name(self) -> str:
        return "reddit"

    def collect(self, since: datetime, until: datetime) -> Iterator[Mention]:
        seen_ids: set[str] = set()

        since_ts = since.replace(tzinfo=timezone.utc).timestamp()
        until_ts = until.replace(tzinfo=timezone.utc).timestamp()

        # Sitewide search — covers anything we'd miss with subreddit-targeted scans
        for term in SEARCH_TERMS:
            yield from self._search_sitewide(term, since_ts, until_ts, seen_ids)

        # Subreddit-scoped search — sometimes returns hits the sitewide misses
        for subreddit in TARGET_SUBREDDITS:
            for term in SEARCH_TERMS:
                yield from self._search_subreddit(
                    subreddit, term, since_ts, until_ts, seen_ids
                )

    # ----- internals -----

    def _search_sitewide(self, term, since_ts, until_ts, seen_ids):
        try:
            results = self._reddit.subreddit("all").search(
                term,
                sort="new",
                time_filter="week",
                limit=self._max_per_query,
            )
            for submission in results:
                if submission.created_utc < since_ts or submission.created_utc >= until_ts:
                    continue
                yield from self._mentions_from_submission(submission, seen_ids)
        except Exception as e:
            # Log and continue — never let one source crash the whole run
            print(f"[reddit] sitewide search failed for '{term}': {e}")

    def _search_subreddit(self, subreddit_name, term, since_ts, until_ts, seen_ids):
        try:
            sub = self._reddit.subreddit(subreddit_name)
            results = sub.search(
                term,
                sort="new",
                time_filter="week",
                limit=self._max_per_query,
            )
            for submission in results:
                if submission.created_utc < since_ts or submission.created_utc >= until_ts:
                    continue
                yield from self._mentions_from_submission(submission, seen_ids)
        except Exception as e:
            print(f"[reddit] r/{subreddit_name} search failed for '{term}': {e}")

    def _mentions_from_submission(self, submission, seen_ids):
        # The post itself
        post_id = f"reddit:{submission.id}"
        if post_id not in seen_ids:
            seen_ids.add(post_id)
            yield Mention(
                id=post_id,
                source=Source.REDDIT,
                brand=self._brand,
                author_handle=str(submission.author) if submission.author else "[deleted]",
                author_metadata={
                    "author_karma": getattr(submission.author, "link_karma", None)
                                    if submission.author else None,
                },
                timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                text=f"{submission.title}\n\n{submission.selftext or ''}".strip(),
                url=f"https://reddit.com{submission.permalink}",
                parent_id=None,
                engagement={
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": submission.upvote_ratio,
                },
                raw_payload={
                    "subreddit": str(submission.subreddit),
                    "is_self": submission.is_self,
                },
            )

        # Top-level comments — limit depth to keep volume manageable for v1
        try:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:50]:
                comment_id = f"reddit:{comment.id}"
                if comment_id in seen_ids:
                    continue
                # Only include comments that mention the search terms — comment
                # threads can drift far from the post topic.
                body_lower = (comment.body or "").lower()
                if not any(t in body_lower for t in SEARCH_TERMS):
                    continue
                seen_ids.add(comment_id)
                yield Mention(
                    id=comment_id,
                    source=Source.REDDIT,
                    brand=self._brand,
                    author_handle=str(comment.author) if comment.author else "[deleted]",
                    author_metadata={
                        "author_karma": getattr(comment.author, "comment_karma", None)
                                        if comment.author else None,
                    },
                    timestamp=datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                    text=comment.body,
                    url=f"https://reddit.com{comment.permalink}",
                    parent_id=f"reddit:{submission.id}",
                    engagement={"score": comment.score},
                    raw_payload={"subreddit": str(submission.subreddit)},
                )
        except Exception as e:
            print(f"[reddit] comments fetch failed for {submission.id}: {e}")
