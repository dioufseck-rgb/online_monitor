"""
SQLite store. Three tables:

- mentions: id (PK), source, author, ts, text, url, parent_id, engagement_json,
            metadata_json, raw_payload_json, ingested_at
- classifications: mention_id (PK, FK), sentiment_label, sentiment_intensity,
                   topic_label, topic_confidence, validity_label, validity_confidence,
                   classifier_version, classified_at
- reports: week_start (PK), report_json, generated_at

We store the full report JSON because the schema may evolve; reads always
go through the Pydantic model (validate on load).

Idempotency: mentions and classifications use INSERT OR IGNORE — re-running
the same week never duplicates.
"""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import Classification, Mention, WeeklyReport


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mentions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    brand TEXT NOT NULL DEFAULT 'unknown',
    domain TEXT NOT NULL DEFAULT 'financial_services',
    intent TEXT,
    author TEXT,
    timestamp TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    snippet TEXT,
    full_text_fetched INTEGER NOT NULL DEFAULT 0,
    url TEXT,
    parent_id TEXT,
    engagement_json TEXT,
    metadata_json TEXT,
    raw_payload_json TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mentions_ts ON mentions(timestamp);
CREATE INDEX IF NOT EXISTS idx_mentions_source ON mentions(source);
CREATE INDEX IF NOT EXISTS idx_mentions_brand ON mentions(brand);

CREATE TABLE IF NOT EXISTS classifications (
    mention_id TEXT PRIMARY KEY,
    sentiment_label TEXT NOT NULL,
    sentiment_intensity REAL,
    sentiment_rationale TEXT,
    topic_label TEXT NOT NULL,
    topic_confidence REAL,
    topic_rationale TEXT,
    validity_label TEXT NOT NULL,
    validity_confidence REAL,
    validity_rationale TEXT,
    origin_label TEXT NOT NULL DEFAULT 'unknown',
    origin_confidence REAL DEFAULT 0,
    origin_rationale TEXT DEFAULT '',
    classified_at TEXT NOT NULL,
    FOREIGN KEY (mention_id) REFERENCES mentions(id)
);

CREATE TABLE IF NOT EXISTS reports (
    week_start TEXT NOT NULL,
    brand TEXT NOT NULL DEFAULT 'unknown',
    report_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (week_start, brand)
);
"""


class Store:
    def __init__(self, path: str | Path = "data/mentions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        """Lightweight schema migrations for fields added after initial design.
        Each step is idempotent (try-except OperationalError on duplicate column).
        """
        migrations = [
            "ALTER TABLE classifications ADD COLUMN origin_label TEXT NOT NULL DEFAULT 'unknown'",
            "ALTER TABLE classifications ADD COLUMN origin_confidence REAL DEFAULT 0",
            "ALTER TABLE classifications ADD COLUMN origin_rationale TEXT DEFAULT ''",
            "ALTER TABLE mentions ADD COLUMN domain TEXT NOT NULL DEFAULT 'financial_services'",
            "ALTER TABLE mentions ADD COLUMN intent TEXT",
        ]
        for sql in migrations:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ----- mentions -----

    def upsert_mentions(self, mentions: list[Mention]) -> int:
        """Insert mentions, ignoring duplicates by id. Returns rows inserted."""
        now = datetime.utcnow().isoformat()
        rows = [
            (
                m.id,
                m.source.value,
                m.brand,
                m.domain,
                m.intent,
                m.author_handle,
                m.timestamp.isoformat(),
                m.title,
                m.text,
                m.snippet,
                1 if m.full_text_fetched else 0,
                m.url,
                m.parent_id,
                json.dumps(m.engagement),
                json.dumps(m.author_metadata),
                json.dumps(m.raw_payload),
                now,
            )
            for m in mentions
        ]
        cur = self._conn.executemany(
            """INSERT OR REPLACE INTO mentions
               (id, source, brand, domain, intent, author, timestamp, title, text, snippet,
                full_text_fetched, url, parent_id, engagement_json, metadata_json, raw_payload_json,
                ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()
        return cur.rowcount

    def get_mentions_in_window(self, since: datetime, until: datetime, brand: str | None = None) -> list[Mention]:
        if brand:
            cur = self._conn.execute(
                """SELECT id, source, brand, domain, intent, author, timestamp, title, text, snippet,
                          full_text_fetched, url, parent_id, engagement_json, metadata_json,
                          raw_payload_json
                   FROM mentions
                   WHERE timestamp >= ? AND timestamp < ? AND brand = ?
                   ORDER BY timestamp""",
                (since.isoformat(), until.isoformat(), brand),
            )
        else:
            cur = self._conn.execute(
                """SELECT id, source, brand, domain, intent, author, timestamp, title, text, snippet,
                          full_text_fetched, url, parent_id, engagement_json, metadata_json,
                          raw_payload_json
                   FROM mentions
                   WHERE timestamp >= ? AND timestamp < ?
                   ORDER BY timestamp""",
                (since.isoformat(), until.isoformat()),
            )
        out: list[Mention] = []
        for row in cur:
            out.append(Mention(
                id=row[0],
                source=row[1],
                brand=row[2],
                domain=row[3] or "financial_services",
                intent=row[4],
                author_handle=row[5] or "",
                timestamp=datetime.fromisoformat(row[6]),
                title=row[7],
                text=row[8],
                snippet=row[9],
                full_text_fetched=bool(row[10]),
                url=row[11] or "",
                parent_id=row[12],
                engagement=json.loads(row[13] or "{}"),
                author_metadata=json.loads(row[14] or "{}"),
                raw_payload=json.loads(row[15] or "{}"),
            ))
        return out

    def get_unclassified_mention_ids(self, mention_ids: list[str]) -> list[str]:
        if not mention_ids:
            return []
        placeholders = ",".join("?" for _ in mention_ids)
        cur = self._conn.execute(
            f"""SELECT m.id FROM mentions m
                LEFT JOIN classifications c ON m.id = c.mention_id
                WHERE m.id IN ({placeholders}) AND c.mention_id IS NULL""",
            mention_ids,
        )
        return [row[0] for row in cur]

    # ----- classifications -----

    def upsert_classifications(self, classifications: list[Classification]) -> int:
        rows = [
            (
                c.mention_id,
                c.sentiment.label.value,
                c.sentiment.intensity,
                c.sentiment.rationale,
                c.topic.label,
                c.topic.confidence,
                c.topic.rationale,
                c.validity_claim.label.value,
                c.validity_claim.confidence,
                c.validity_claim.rationale,
                c.origin.label.value,
                c.origin.confidence,
                c.origin.rationale,
                c.classified_at.isoformat(),
            )
            for c in classifications
        ]
        cur = self._conn.executemany(
            """INSERT OR REPLACE INTO classifications
               (mention_id, sentiment_label, sentiment_intensity, sentiment_rationale,
                topic_label, topic_confidence, topic_rationale,
                validity_label, validity_confidence, validity_rationale,
                origin_label, origin_confidence, origin_rationale,
                classified_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.commit()
        return cur.rowcount

    def get_classifications(self, mention_ids: list[str]) -> list[Classification]:
        if not mention_ids:
            return []
        placeholders = ",".join("?" for _ in mention_ids)
        cur = self._conn.execute(
            f"""SELECT mention_id, sentiment_label, sentiment_intensity, sentiment_rationale,
                       topic_label, topic_confidence, topic_rationale,
                       validity_label, validity_confidence, validity_rationale,
                       origin_label, origin_confidence, origin_rationale,
                       classified_at
                FROM classifications WHERE mention_id IN ({placeholders})""",
            mention_ids,
        )
        out: list[Classification] = []
        for row in cur:
            out.append(Classification(
                mention_id=row[0],
                sentiment={"label": row[1], "intensity": row[2], "rationale": row[3] or ""},
                topic={"label": row[4], "confidence": row[5], "rationale": row[6] or ""},
                validity_claim={"label": row[7], "confidence": row[8], "rationale": row[9] or ""},
                origin={"label": row[10] or "unknown", "confidence": row[11] or 0,
                        "rationale": row[12] or ""},
                classified_at=datetime.fromisoformat(row[13]),
            ))
        return out

    # ----- reports -----

    def save_report(self, report: WeeklyReport, brand: str) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO reports (week_start, brand, report_json, generated_at)
               VALUES (?, ?, ?, ?)""",
            (
                report.week_start.isoformat(),
                brand,
                report.model_dump_json(),
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    def get_prior_report(self, before: datetime, brand: str) -> Optional[WeeklyReport]:
        cur = self._conn.execute(
            """SELECT report_json FROM reports
               WHERE week_start < ? AND brand = ?
               ORDER BY week_start DESC LIMIT 1""",
            (before.isoformat(), brand),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return WeeklyReport.model_validate_json(row[0])

    def close(self):
        self._conn.close()
