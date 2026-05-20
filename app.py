"""AgroVision AI - rotas FastAPI e orquestracao de services."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.config import AGENT_EVENT_LIMIT, DEBUG
from services.event_repository import init_db, list_events
from services.monitoring_agent import (
    AGENT_PROFILE,
    build_agent_messages,
    build_event_context,
)
from services import ollama_client
from services.ollama_client import OllamaUnavailableError
from services.schemas import ChatRequest, ChatResponse
from services.video_monitor import VideoMonitor

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrovision")

app = FastAPI(title="AgroVision AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

monitor = VideoMonitor()


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    monitor.start()
    asyncio.create_task(ollama_client.warmup())
    logger.info("AgroVision AI pronto")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    monitor.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> JSONResponse:
    available = await ollama_client.is_available()
    return JSONResponse(
        {"status": "ok", "service": "AgroVision AI", "ollama_available": available}
    )


@app.get("/events")
async def events() -> JSONResponse:
    return JSONResponse(list_events(AGENT_EVENT_LIMIT))


@app.get("/camera/status")
async def camera_status() -> JSONResponse:
    return JSONResponse(monitor.get_status())


@app.get("/agent/status")
async def agent_status() -> JSONResponse:
    events = list_events(AGENT_EVENT_LIMIT)
    return JSONResponse(
        {
            "name": AGENT_PROFILE.name,
            "role": AGENT_PROFILE.role,
            "goal": AGENT_PROFILE.goal,
            "events_in_context": len(events),
            "context_preview": build_event_context(events),
        }
    )


@app.get("/frame")
async def frame() -> Response:
    data = monitor.get_last_frame()
    if data is None:
        return Response(status_code=503, content="Nenhum frame disponivel")
    return Response(content=data, media_type="image/jpeg")


async def _mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        data = monitor.get_last_frame()
        if data is not None:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        await asyncio.sleep(0.05)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _wants_stream(request: Request) -> bool:
    if request.query_params.get("stream") in {"1", "true"}:
        return True
    accept = request.headers.get("accept", "")
    return "application/x-ndjson" in accept


@app.post("/chat")
async def chat(request: Request, payload: ChatRequest):
    events = list_events(AGENT_EVENT_LIMIT)
    messages = build_agent_messages(payload.question, payload.history, events)

    if not _wants_stream(request):
        try:
            answer = await ollama_client.chat(messages)
        except OllamaUnavailableError as exc:
            logger.warning("Falha na chamada ao Ollama: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Servico de IA temporariamente indisponivel",
            ) from exc
        return ChatResponse(answer=answer)

    async def event_stream():
        try:
            async for chunk in ollama_client.chat_stream(messages):
                yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except OllamaUnavailableError as exc:
            logger.warning("Falha no streaming do Ollama: %s", exc)
            yield json.dumps(
                {"error": "Ollama indisponivel no momento.", "done": True},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
