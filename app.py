"""AgroVision AI - rotas FastAPI e orquestracao de services."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services.config import AGENT_EVENT_LIMIT, DEBUG, WEATHER_ENABLED
from services.event_repository import init_db, list_events
from services.monitoring_agent import (
    AGENT_PROFILE,
    build_agent_messages,
    build_event_context,
)
from services import ollama_client
from services import weather_scraper
from services.ollama_client import OllamaUnavailableError
from services.schemas import ChatRequest, ChatResponse
from services.video_monitor import VideoMonitor

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrovision")

monitor = VideoMonitor()


async def _persist_weather_loop():
    from services.event_repository import save_weather_snapshot
    while True:
        try:
            if WEATHER_ENABLED:
                snap = await weather_scraper.get_current()
                if snap is not None and not snap.is_stale:
                    save_weather_snapshot(snap.to_dict())
                    logger.debug("Snapshot de clima persistido")
        except Exception as exc:
            logger.warning("Falha ao persistir clima: %s", exc)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    monitor.start()
    warmup_task = asyncio.create_task(ollama_client.warmup())
    app.state.warmup_task = warmup_task
    weather_task = asyncio.create_task(_persist_weather_loop())
    app.state.weather_task = weather_task
    logger.info("AgroVision AI pronto")
    try:
        yield
    finally:
        monitor.stop()
        for task in (warmup_task, weather_task):
            if not task.done():
                task.cancel()


app = FastAPI(title="AgroVision AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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


@app.get("/weather")
async def weather() -> JSONResponse:
    if not WEATHER_ENABLED:
        return JSONResponse({"enabled": False}, status_code=200)
    snap = await weather_scraper.get_current()
    if snap is None:
        return JSONResponse({"enabled": True, "available": False}, status_code=503)
    return JSONResponse({"enabled": True, "available": True, **snap.to_dict()})


@app.get("/agent/status")
async def agent_status() -> JSONResponse:
    recent_events = list_events(AGENT_EVENT_LIMIT)
    return JSONResponse(
        {
            "name": AGENT_PROFILE.name,
            "role": AGENT_PROFILE.role,
            "goal": AGENT_PROFILE.goal,
            "events_in_context": len(recent_events),
            "context_preview": build_event_context(recent_events),
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
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest):
    recent_events = list_events(AGENT_EVENT_LIMIT)
    weather_dict = None
    if WEATHER_ENABLED:
        snap = await weather_scraper.get_current()
        if snap is not None:
            weather_dict = snap.to_dict()
    messages = build_agent_messages(payload.question, payload.history, recent_events, weather_dict)

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
