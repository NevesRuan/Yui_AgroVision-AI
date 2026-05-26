"""Camada de web scraping: clima via Open-Meteo.

Fonte: https://open-meteo.com/ (API REST publica, sem API key, licenca CC-BY 4.0).

Arquitetura:
  - Service isolado: nao conhece SQLite, FastAPI, nem o agente.
  - Cache em memoria com TTL (limite de requisicoes implicito).
  - Em falha de rede ou parsing, retorna o ultimo snapshot bom marcado is_stale=True;
    se nunca houve sucesso, retorna None.
  - Dados sao validados contra ranges plausiveis (defesa contra valores absurdos).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from services.config import (
    WEATHER_CACHE_TTL_SECONDS,
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
    WEATHER_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Codigos WMO -> label legivel em pt-BR.
# Referencia: https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODES: dict[int, str] = {
    0: "Ceu limpo",
    1: "Predominantemente limpo",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Neblina",
    48: "Neblina com geada",
    51: "Garoa fraca",
    53: "Garoa moderada",
    55: "Garoa intensa",
    56: "Garoa congelante fraca",
    57: "Garoa congelante intensa",
    61: "Chuva fraca",
    63: "Chuva moderada",
    65: "Chuva forte",
    66: "Chuva congelante fraca",
    67: "Chuva congelante forte",
    71: "Neve fraca",
    73: "Neve moderada",
    75: "Neve forte",
    77: "Graos de neve",
    80: "Pancadas de chuva fracas",
    81: "Pancadas de chuva moderadas",
    82: "Pancadas de chuva violentas",
    85: "Pancadas de neve fracas",
    86: "Pancadas de neve fortes",
    95: "Tempestade",
    96: "Tempestade com granizo fraco",
    99: "Tempestade com granizo forte",
}


def _decode_weather_code(code: int | None) -> str:
    if code is None:
        return "Indisponivel"
    return _WEATHER_CODES.get(int(code), f"Codigo {code}")


def _validate_range(name: str, value: float | None, lo: float, hi: float) -> float | None:
    """Aceita None ou valor dentro de [lo, hi]. Fora do range: descarta com warning."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not (lo <= v <= hi):
        logger.warning("Valor %s fora do range plausivel: %s", name, v)
        return None
    return v


@dataclass(frozen=True)
class WeatherSnapshot:
    fetched_at: str  # ISO 8601 com timezone
    latitude: float
    longitude: float
    temperature_c: float | None
    humidity_pct: int | None
    precipitation_mm: float | None
    wind_kmh: float | None
    condition_code: int | None
    condition_label: str
    is_day: bool
    is_stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_response(data: dict[str, Any], lat: float, lon: float) -> WeatherSnapshot | None:
    """Aceita o payload do Open-Meteo (chave 'current') e devolve snapshot validado."""
    current = data.get("current") or {}
    if not current:
        logger.warning("Resposta Open-Meteo sem campo 'current': %s", data)
        return None

    temperature = _validate_range("temperature", current.get("temperature_2m"), -90, 60)
    humidity_raw = _validate_range("humidity", current.get("relative_humidity_2m"), 0, 100)
    humidity = int(humidity_raw) if humidity_raw is not None else None
    precipitation = _validate_range("precipitation", current.get("precipitation"), 0, 500)
    wind = _validate_range("wind", current.get("wind_speed_10m"), 0, 500)

    code_raw = current.get("weather_code")
    try:
        code = int(code_raw) if code_raw is not None else None
    except (TypeError, ValueError):
        code = None

    is_day_raw = current.get("is_day")
    is_day = bool(is_day_raw) if is_day_raw is not None else True

    return WeatherSnapshot(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        latitude=lat,
        longitude=lon,
        temperature_c=temperature,
        humidity_pct=humidity,
        precipitation_mm=precipitation,
        wind_kmh=wind,
        condition_code=code,
        condition_label=_decode_weather_code(code),
        is_day=is_day,
        is_stale=False,
    )


class WeatherScraper:
    """Coletor de clima com cache TTL e fallback para snapshot stale."""

    def __init__(
        self,
        latitude: float = WEATHER_LATITUDE,
        longitude: float = WEATHER_LONGITUDE,
        cache_ttl_seconds: int = WEATHER_CACHE_TTL_SECONDS,
        request_timeout: int = WEATHER_REQUEST_TIMEOUT,
    ) -> None:
        self._lat = float(latitude)
        self._lon = float(longitude)
        self._cache_ttl = max(60, int(cache_ttl_seconds))  # minimo 60s
        self._timeout = max(1, int(request_timeout))
        self._cached: WeatherSnapshot | None = None
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_current(self) -> WeatherSnapshot | None:
        """Retorna snapshot mais recente (fresh, cached ou stale)."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._cached and (now - self._cached_at) < self._cache_ttl:
                return self._cached

            try:
                snap = await self._fetch()
            except Exception as exc:
                logger.warning("Falha ao buscar clima: %s", exc)
                snap = None

            if snap is not None:
                self._cached = snap
                self._cached_at = now
                return snap

            # Falhou agora; se temos cache antigo, devolvemos marcado como stale.
            if self._cached is not None:
                stale = WeatherSnapshot(
                    fetched_at=self._cached.fetched_at,
                    latitude=self._cached.latitude,
                    longitude=self._cached.longitude,
                    temperature_c=self._cached.temperature_c,
                    humidity_pct=self._cached.humidity_pct,
                    precipitation_mm=self._cached.precipitation_mm,
                    wind_kmh=self._cached.wind_kmh,
                    condition_code=self._cached.condition_code,
                    condition_label=self._cached.condition_label,
                    is_day=self._cached.is_day,
                    is_stale=True,
                )
                return stale

            return None

    async def _fetch(self) -> WeatherSnapshot | None:
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "is_day",
                ]
            ),
            "wind_speed_unit": "kmh",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(_OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()
        return _parse_response(data, self._lat, self._lon)


# --- Singleton de modulo ---

_scraper: WeatherScraper | None = None


def _get_scraper() -> WeatherScraper:
    global _scraper
    if _scraper is None:
        _scraper = WeatherScraper()
    return _scraper


async def get_current() -> WeatherSnapshot | None:
    """Atalho de modulo: usa o singleton do WeatherScraper."""
    return await _get_scraper().get_current()
