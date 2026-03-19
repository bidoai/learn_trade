"""
Event store — append-only SQLite log of all system events.

Every significant event (fills, orders, signals, risk alerts) is
written here. This enables:
  1. Post-mortem analysis: what happened and why
  2. Trade replay: reconstruct any trading session second-by-second
  3. Audit trail: prove what the system did

SQLite is opened in WAL mode so the dashboard can read concurrently
without blocking writes from the trading engine.

Schema: one table, events, with columns:
  id          INTEGER PRIMARY KEY
  event_type  TEXT NOT NULL
  timestamp   TEXT NOT NULL (ISO 8601)
  payload     TEXT NOT NULL (JSON)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from dataclasses import asdict
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


class EventStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """
        Create the database and enable WAL mode.
        WAL (Write-Ahead Logging) allows concurrent readers without
        blocking the writer — required for the dashboard to read
        while the trading engine writes.
        """
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster, still safe with WAL
        self._conn.executescript(_CREATE_TABLE)
        self._conn.commit()
        logger.info("event_store.initialized", db_path=self.db_path)

    def append(self, event: object) -> None:
        """
        Persist an event to the store.
        Events are frozen dataclasses — use dataclasses.asdict for serialization.
        Continues on SQLite errors (trading must not halt due to logging failure).
        """
        if self._conn is None:
            logger.error("event_store.not_initialized")
            return

        event_type = type(event).__name__
        timestamp = getattr(event, "timestamp", datetime.utcnow())

        try:
            # Serialize: convert dataclass to dict, handle nested dataclasses
            payload = json.dumps(
                _serialize(event),
                default=str,  # fallback for datetime and enums
            )
            self._conn.execute(
                "INSERT INTO events (event_type, timestamp, payload) VALUES (?, ?, ?)",
                (event_type, timestamp.isoformat(), payload),
            )
            self._conn.commit()
        except sqlite3.OperationalError as e:
            # Disk full, locked, corrupted — log but don't halt trading
            logger.critical(
                "event_store.write_failed",
                event_type=event_type,
                error=str(e),
            )

    def replay(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve events for replay or analysis.
        Returns raw dicts — callers reconstruct domain objects.
        """
        if self._conn is None:
            return []

        query = "SELECT event_type, timestamp, payload FROM events WHERE 1=1"
        params: list = []

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)

        query += " ORDER BY timestamp ASC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            {"event_type": r[0], "timestamp": r[1], "payload": json.loads(r[2])}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()


def _serialize(obj: object) -> dict:
    """Recursively convert a frozen dataclass to a JSON-serializable dict."""
    try:
        from dataclasses import fields, is_dataclass
        if is_dataclass(obj):
            result = {}
            for f in fields(obj):  # type: ignore
                value = getattr(obj, f.name)
                if is_dataclass(value):
                    result[f.name] = _serialize(value)
                elif hasattr(value, "value"):  # Enum
                    result[f.name] = value.value
                else:
                    result[f.name] = value
            return result
        return {"value": str(obj)}
    except Exception:
        return {"value": str(obj)}
