"""Captura de video + YOLO em thread daemon, com reconexao automatica."""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from services.config import (
    ALERT_COOLDOWN_SECONDS,
    CAMERA_RECONNECT_SECONDS,
    CAMERA_SOURCE,
    CONFIDENCE_THRESHOLD,
    MIN_CONSECUTIVE_FRAMES,
    MODEL_PATH,
    SAVE_DIR,
    TARGET_CLASSES,
)
from services.event_repository import save_event

logger = logging.getLogger(__name__)

_MAX_READ_FAILURES = 10
_JPEG_QUALITY = 80


def _detect_source_type(source: str | int) -> str:
    if isinstance(source, int):
        return "webcam"
    s = str(source).strip().lower()
    if s.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return "stream"
    return "file"


@dataclass
class _DetectionState:
    # Contador de frames consecutivos por label para evitar alertar em flicker.
    consecutive: dict[str, int] = field(default_factory=dict)
    last_alert_at: dict[str, float] = field(default_factory=dict)


def _should_alert(state: _DetectionState, label: str) -> bool:
    count = state.consecutive.get(label, 0)
    if count < MIN_CONSECUTIVE_FRAMES:
        return False
    now = time.time()
    last = state.last_alert_at.get(label, 0.0)
    if now - last < ALERT_COOLDOWN_SECONDS:
        return False
    state.last_alert_at[label] = now
    return True


def _draw_box(
    frame: "cv2.typing.MatLike",
    xyxy: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


class VideoMonitor:
    """Encapsula captura, inferencia YOLO e o frame mais recente em memoria."""

    def __init__(self) -> None:
        self._source: str | int = CAMERA_SOURCE
        self._source_type: str = _detect_source_type(CAMERA_SOURCE)
        self._model: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_frame: bytes | None = None
        self._last_frame_lock = threading.Lock()
        self._connected: bool = False
        self._state = _DetectionState()
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # --- API publica ---

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="video-monitor", daemon=True
        )
        self._thread.start()
        logger.info("VideoMonitor iniciado (fonte=%s)", self._source_type)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_last_frame(self) -> bytes | None:
        with self._last_frame_lock:
            return self._last_frame

    def get_status(self) -> dict:
        return {
            "online": self._thread is not None and self._thread.is_alive(),
            "connected": self._connected,
            "has_live_frame": self.get_last_frame() is not None,
            "source_type": self._source_type,
        }

    # --- Loop interno ---

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from ultralytics import YOLO  # import local para nao exigir no boot se faltar
        self._model = YOLO(MODEL_PATH)
        logger.info("Modelo YOLO carregado de %s", MODEL_PATH)
        return self._model

    def _open_capture(self) -> "cv2.VideoCapture | None":
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            logger.warning("Nao foi possivel abrir a fonte %s", self._source_type)
            cap.release()
            return None
        self._connected = True
        logger.info("Conexao com a camera estabelecida (%s)", self._source_type)
        return cap

    def _run_loop(self) -> None:
        try:
            model = self._load_model()
        except Exception as exc:  # pragma: no cover
            logger.exception("Falha ao carregar YOLO: %s", exc)
            return

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                self._connected = False
                time.sleep(CAMERA_RECONNECT_SECONDS)
                continue

            read_failures = 0
            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        read_failures += 1
                        if read_failures >= _MAX_READ_FAILURES:
                            logger.warning(
                                "Falhas consecutivas de leitura (%s). Reconectando.",
                                read_failures,
                            )
                            break
                        time.sleep(0.1)
                        continue
                    read_failures = 0
                    self._process_frame(model, frame)
            finally:
                cap.release()
                self._connected = False
                if not self._stop_event.is_set():
                    time.sleep(CAMERA_RECONNECT_SECONDS)

    def _process_frame(self, model: Any, frame: "cv2.typing.MatLike") -> None:
        try:
            results = model.predict(
                frame, conf=CONFIDENCE_THRESHOLD, verbose=False
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Falha na inferencia YOLO: %s", exc)
            self._update_frame(frame)
            return

        seen_labels: set[str] = set()
        if results:
            r = results[0]
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    label = names.get(cls_id, str(cls_id))
                    if TARGET_CLASSES and label not in TARGET_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    _draw_box(frame, (x1, y1, x2, y2), label, conf)
                    seen_labels.add(label)
                    self._state.consecutive[label] = (
                        self._state.consecutive.get(label, 0) + 1
                    )
                    if _should_alert(self._state, label):
                        self._persist_event(frame, label, conf)

        # Zera contadores de labels que nao apareceram neste frame.
        for label in list(self._state.consecutive.keys()):
            if label not in seen_labels:
                self._state.consecutive[label] = 0

        self._update_frame(frame)

    def _update_frame(self, frame: "cv2.typing.MatLike") -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
        )
        if not ok:
            return
        with self._last_frame_lock:
            self._last_frame = buf.tobytes()

    def _persist_event(
        self, frame: "cv2.typing.MatLike", label: str, confidence: float
    ) -> None:
        filename = f"{label}_{int(time.time())}.jpg"
        path = os.path.join(SAVE_DIR, filename)
        try:
            cv2.imwrite(path, frame)
            save_event(label, confidence, f"/static/captures/{filename}")
            logger.info(
                "Evento registrado: %s (%.2f) -> %s", label, confidence, filename
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Falha ao salvar evento: %s", exc)
