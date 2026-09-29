"""FastAPI entry point.

Run from the repo root:
    .venv\\Scripts\\python -m uvicorn app.main:app --app-dir backend --port 8000
or simply:  scripts\\run.ps1
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.ws import manager
from app.api.ws import router as ws_router
from app.core.config import REPO_ROOT, get_settings
from app.core.gpu import gpu_snapshot
from app.core.logging import setup_logging
from app.services import Services


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_dir)
    logger.info(f"backend starting, profile={settings.profile}")
    services = Services(settings)
    app.state.services = services
    await services.start()
    yield
    if manager.active is not None:
        await manager.active.close("shutdown")
    await services.stop()
    logger.info("backend stopped")


app = FastAPI(title="AI Avatar Voice Agent", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
async def health() -> dict:
    services: Services = app.state.services
    return {
        "status": "ok",
        "profile": services.settings.profile,
        "llm_server_running": services.llm_server.running or not services.settings.llm.manage_server,
        "llm_restarts": services.llm_server.restarts,
        "active_session": manager.active.id if manager.active else None,
        "avatar": {"connected": services.avatar_ready, "restarts": services.avatar.restarts,
                   "hello": services.avatar.hello, "stats": services.avatar.stats} if services.avatar else None,
        "load_report": services.load_report,
    }


@app.get("/health/gpu")
async def health_gpu() -> dict:
    return gpu_snapshot()


_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
