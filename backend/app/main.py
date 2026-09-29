"""FastAPI entry point.

Run from the repo root:
    .venv\\Scripts\\python -m uvicorn app.main:app --app-dir backend --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from app.core.config import get_settings
from app.core.gpu import gpu_snapshot
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_dir)
    logger.info(f"backend starting, profile={settings.profile}")
    yield
    logger.info("backend stopped")


app = FastAPI(title="AI Avatar Voice Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "profile": get_settings().profile}


@app.get("/health/gpu")
async def health_gpu() -> dict:
    return gpu_snapshot()


@app.websocket("/ws")
async def ws_echo(ws: WebSocket) -> None:
    """Phase 0 placeholder: echoes JSON messages back."""
    await ws.accept()
    try:
        while True:
            await ws.send_json({"type": "echo", "data": await ws.receive_json()})
    except WebSocketDisconnect:
        pass
