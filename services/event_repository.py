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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                temperature_c REAL,
                humidity_pct INTEGER,
                precipitation_mm REAL,
                wind_kmh REAL,
                condition_code INTEGER,
                condition_label TEXT,
                is_day INTEGER
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_weather_fetched ON weather_snapshots(fetched_at DESC)"
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


def save_weather_snapshot(snapshot: dict) -> int:
    """Persiste um snapshot de clima; ignora chaves fora da tabela (ex: is_stale)."""
    is_day = snapshot.get("is_day")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO weather_snapshots ("
            "fetched_at, latitude, longitude, temperature_c, humidity_pct, "
            "precipitation_mm, wind_kmh, condition_code, condition_label, is_day"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.get("fetched_at"),
                snapshot.get("latitude"),
                snapshot.get("longitude"),
                snapshot.get("temperature_c"),
                snapshot.get("humidity_pct"),
                snapshot.get("precipitation_mm"),
                snapshot.get("wind_kmh"),
                snapshot.get("condition_code"),
                snapshot.get("condition_label"),
                int(is_day) if is_day is not None else None,
            ),
        )
        return int(cur.lastrowid)
