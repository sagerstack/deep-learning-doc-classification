"""SQLite-backed monitoring event persistence.

Uses stdlib sqlite3 only. No ORM dependency.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from app.src.monitoring.schema import PROB_COLUMN_NAMES, InferenceEvent

logger = logging.getLogger(__name__)

# ─── Schema DDL ───────────────────────────────────────────────────────────────

_PROB_COLS_DDL = "\n".join(
    f"    {col} REAL NOT NULL DEFAULT 0.0," for col in PROB_COLUMN_NAMES
)

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS inference_events (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id             TEXT    NOT NULL,
    timestamp              TEXT    NOT NULL,
    sample_type            TEXT    NOT NULL,
    sample_name            TEXT    NOT NULL,
    image_width            INTEGER NOT NULL,
    image_height           INTEGER NOT NULL,
    image_mode             TEXT    NOT NULL,
    model_id               TEXT    NOT NULL,
    model_display_name     TEXT    NOT NULL,
    model_version          TEXT    NOT NULL DEFAULT '1.0',
    predicted_label        TEXT    NOT NULL,
    predicted_index        INTEGER NOT NULL,
    confidence             REAL    NOT NULL,
{_PROB_COLS_DDL}
    total_time_ms          REAL    NOT NULL DEFAULT 0.0,
    feature_time_ms        REAL    NOT NULL DEFAULT 0.0,
    graph_time_ms          REAL    NOT NULL DEFAULT 0.0,
    model_time_ms          REAL    NOT NULL DEFAULT 0.0,
    ocr_available          INTEGER NOT NULL DEFAULT 0,
    text_density_available INTEGER NOT NULL DEFAULT 0,
    error_type             TEXT
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_inference_events_timestamp
    ON inference_events (timestamp)
"""

_CREATE_INDEX_MODEL_SQL = """
CREATE INDEX IF NOT EXISTS idx_inference_events_model_id
    ON inference_events (model_id)
"""


# ─── Public API ───────────────────────────────────────────────────────────────


def init_db(db_path: Path) -> None:
    """Create the SQLite database file and table if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.execute(_CREATE_INDEX_MODEL_SQL)
        conn.commit()
    logger.info("Monitoring DB ready: %s", db_path)


def log_inference_events(events: list[InferenceEvent], db_path: Path) -> None:
    """Persist a batch of InferenceEvents to SQLite.

    Safe to call with an empty list (no-op).
    """
    if not events:
        return

    rows = [_event_to_row(e) for e in events]
    columns = _column_list()
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO inference_events ({', '.join(columns)}) VALUES ({placeholders})"

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.executemany(sql, rows)
            conn.commit()
        logger.debug("Logged %d inference events (request_id=%s)", len(events), events[0].request_id)
    except sqlite3.Error as exc:
        logger.error("Failed to log inference events: %s", exc)


def query_events(
    db_path: Path,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 1000,
) -> list[dict]:
    """Fetch inference events as a list of dicts.

    Args:
        db_path: Path to the SQLite file.
        since: ISO-8601 timestamp lower bound (inclusive).
        until: ISO-8601 timestamp upper bound (inclusive).
        model_id: Filter to a specific model.
        limit: Maximum rows to return.

    Returns:
        List of row dicts, newest first.
    """
    conditions: list[str] = []
    params: list = []

    if since:
        conditions.append("timestamp >= ?")
        params.append(since)
    if until:
        conditions.append("timestamp <= ?")
        params.append(until)
    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM inference_events {where_clause} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.error("Failed to query inference events: %s", exc)
        return []


# ─── Internal helpers ──────────────────────────────────────────────────────────


def _column_list() -> list[str]:
    base = [
        "request_id", "timestamp",
        "sample_type", "sample_name",
        "image_width", "image_height", "image_mode",
        "model_id", "model_display_name", "model_version",
        "predicted_label", "predicted_index", "confidence",
    ]
    base.extend(PROB_COLUMN_NAMES)
    base.extend([
        "total_time_ms", "feature_time_ms", "graph_time_ms", "model_time_ms",
        "ocr_available", "text_density_available",
        "error_type",
    ])
    return base


def _event_to_row(e: InferenceEvent) -> tuple:
    row: list = [
        e.request_id, e.timestamp,
        e.sample_type, e.sample_name,
        e.image_width, e.image_height, e.image_mode,
        e.model_id, e.model_display_name, e.model_version,
        e.predicted_label, e.predicted_index, e.confidence,
    ]
    row.extend(e.probabilities)
    row.extend([
        e.total_time_ms, e.feature_time_ms, e.graph_time_ms, e.model_time_ms,
        int(e.ocr_available), int(e.text_density_available),
        e.error_type,
    ])
    return tuple(row)
