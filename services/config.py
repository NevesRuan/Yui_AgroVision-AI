"""Fonte unica de leitura de variaveis de ambiente do projeto."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


def _get_camera_source(raw: str) -> str | int:
    # Permite numero inteiro para webcam (ex.: CAMERA_SOURCE=0).
    if raw.isdigit():
        return int(raw)
    return raw


# --- Ollama ---
OLLAMA_URL: str = _get_str("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL: str = _get_str("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT: int = _get_int("OLLAMA_TIMEOUT", 120)
OLLAMA_KEEP_ALIVE: str = _get_str("OLLAMA_KEEP_ALIVE", "30m")

# --- Agente ---
AGENT_EVENT_LIMIT: int = _get_int("AGENT_EVENT_LIMIT", 12)

# --- Camera / video ---
CAMERA_SOURCE: str | int = _get_camera_source(
    _get_str(
        "CAMERA_SOURCE",
        "https://wzmedia.dot.ca.gov/D11/C214_SB_5_at_Via_De_San_Ysidro.stream/playlist.m3u8",
    )
)
CAMERA_RECONNECT_SECONDS: int = _get_int("CAMERA_RECONNECT_SECONDS", 5)

# --- YOLO ---
MODEL_PATH: str = _get_str("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD: float = _get_float("CONFIDENCE_THRESHOLD", 0.5)
_TARGET_CLASSES_RAW: str = _get_str(
    "TARGET_CLASSES", "person,car,truck,bus,motorcycle,bicycle"
)
TARGET_CLASSES: tuple[str, ...] = tuple(
    c.strip() for c in _TARGET_CLASSES_RAW.split(",") if c.strip()
)
MIN_CONSECUTIVE_FRAMES: int = _get_int("MIN_CONSECUTIVE_FRAMES", 3)
ALERT_COOLDOWN_SECONDS: int = _get_int("ALERT_COOLDOWN_SECONDS", 15)

# --- Persistencia ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: str = _get_str("DB_PATH", str(BASE_DIR / "detections.db"))
SAVE_DIR: str = _get_str("SAVE_DIR", str(BASE_DIR / "static" / "captures"))

# --- Web scraping: clima ---
WEATHER_ENABLED: bool = _get_str("WEATHER_ENABLED", "1").strip() not in {"0", "", "false", "False"}
WEATHER_LATITUDE: float = _get_float("WEATHER_LATITUDE", 32.55)
WEATHER_LONGITUDE: float = _get_float("WEATHER_LONGITUDE", -117.04)
WEATHER_CACHE_TTL_SECONDS: int = _get_int("WEATHER_CACHE_TTL_SECONDS", 600)
WEATHER_REQUEST_TIMEOUT: int = _get_int("WEATHER_REQUEST_TIMEOUT", 5)

# --- Limites HTTP ---
# Teto de tamanho do corpo da requisicao. Acomoda o ChatRequest maximo (~84 KB:
# 20 mensagens de 4000 chars + pergunta de 2000) com folga e barra corpos abusivos.
MAX_REQUEST_BODY_BYTES: int = _get_int("MAX_REQUEST_BODY_BYTES", 262144)

# --- Debug ---
DEBUG: bool = _get_str("DEBUG", "0").strip() not in {"0", "", "false", "False"}
