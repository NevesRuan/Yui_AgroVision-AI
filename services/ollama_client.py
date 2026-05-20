"""Cliente assincrono para a API local do Ollama."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from services.config import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Falha de comunicacao com o Ollama (timeout, conexao recusada, etc)."""


def _tags_url() -> str:
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    return f"{base}/api/tags"


def _base_payload(messages: list[dict], stream: bool) -> dict:
    return {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(_tags_url())
            return r.status_code == 200
    except Exception as exc:
        logger.debug("Ollama indisponivel: %s", exc)
        return False


async def chat(messages: list[dict]) -> str:
    payload = _base_payload(messages, stream=False)
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(str(exc)) from exc
    message = data.get("message", {}) or {}
    return str(message.get("content", ""))


async def chat_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Streaming NDJSON: produz cada chunk de texto conforme chega."""
    payload = _base_payload(messages, stream=True)
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = (obj.get("message") or {}).get("content", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        break
    except httpx.HTTPError as exc:
        raise OllamaUnavailableError(str(exc)) from exc


async def warmup() -> None:
    """Dispara um request minimo pra carregar o modelo em RAM."""
    try:
        await chat([{"role": "user", "content": "ok"}])
        logger.info("Ollama warmup OK (modelo=%s)", OLLAMA_MODEL)
    except Exception as exc:
        logger.warning("Ollama warmup falhou: %s", exc)
