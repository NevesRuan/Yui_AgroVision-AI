"""Listagem de imagens capturadas em static/captures/."""
from __future__ import annotations

from pathlib import Path

from services.config import SAVE_DIR

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_captures(limit: int = 20) -> list[str]:
    base = Path(SAVE_DIR)
    if not base.exists():
        return []
    files = [
        p
        for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED_SUFFIXES
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [f"/static/captures/{p.name}" for p in files[:limit]]
