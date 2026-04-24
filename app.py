"""AgroVision AI - rotas FastAPI e orquestracao de services."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.config import AGENT_EVENT_LIMIT
from services.event_repository import init_db, list_events
from services.video_monitor import VideoMonitor

logging.basicConfig(
    level=logging.INFO,
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
    logger.info("AgroVision AI pronto")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    monitor.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "AgroVision AI", "ollama_available": False}
    )


@app.get("/events")
async def events() -> JSONResponse:
    return JSONResponse(list_events(AGENT_EVENT_LIMIT))


@app.get("/frame")
async def frame() -> Response:
    data = monitor.get_last_frame()
    if data is None:
        return Response(status_code=503, content="Nenhum frame disponivel")
    return Response(content=data, media_type="image/jpeg")
