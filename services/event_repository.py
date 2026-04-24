"""Persistencia SQLite para eventos de deteccao (label, confianca, imagem)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from services.config import DB_PATH


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp DESC)"
        )


def save_event(label: str, confidence: float, image_path: str | None = None) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO events (label, confidence, timestamp, image_path) "
            "VALUES (?, ?, ?, ?)",
            (label, float(confidence), timestamp, image_path),
        )
        return int(cur.lastrowid)


def list_events(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, label, confidence, timestamp, image_path "
            "FROM events ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def count_events() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM events").fetchone()
    return int(row["total"]) if row else 0
