"""FastAPI entry point.

Run from the repo root:
    .venv\\Scripts\\python -m uvicorn app.main:app --app-dir backend --port 8000
or simply:  scripts\\run.ps1
"""
from __future__ import annotations

import json
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
        "llm_ready": services.llm_server.ready.is_set() or not services.settings.llm.manage_server,
        "active_session": manager.active.id if manager.active else None,
        "avatar": {"connected": services.avatar_ready, "restarts": services.avatar.restarts,
                   "hello": services.avatar.hello, "stats": services.avatar.stats} if services.avatar else None,
        "load_report": services.load_report,
        "warnings": services.warnings,
        "voice_ready": services.voice_ready,
        "tts_engine": services.settings.tts.engine if services.tts else None,
        "rag_chunks": services.rag.rows if services.rag else 0,
    }


@app.get("/health/gpu")
async def health_gpu() -> dict:
    snap = gpu_snapshot()
    if snap.get("available") and snap["vram_free_mib"] < 400:
        snap["warning"] = "less than 400 MiB VRAM free: Windows may spill to system RAM (very slow)"
    return snap


@app.get("/api/metrics")
async def api_metrics(last: int = 50) -> dict:
    """p50/p90 per pipeline stage over the last N completed turns (ms from end of user input)."""
    path = app.state.services.settings.log_dir / "metrics.jsonl"
    if not path.exists():
        return {"turns": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[-2000:] if line.strip()]
    rows = [r for r in rows if not r.get("interrupted")][-last:]
    stages: dict[str, list[float]] = {}
    for r in rows:
        for k, v in (r.get("ms_from_t0") or {}).items():
            if k != "t0" and v is not None:
                stages.setdefault(k, []).append(v)

    def pct(vals: list[float], p: float) -> float:
        vals = sorted(vals)
        return round(vals[min(len(vals) - 1, int(p * len(vals)))], 1)

    return {"turns": len(rows), "stages": {k: {"p50": pct(v, 0.5), "p90": pct(v, 0.9), "n": len(v)}
                                          for k, v in sorted(stages.items(), key=lambda kv: pct(kv[1], 0.5))}}


_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
