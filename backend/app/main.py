"""FastAPI entry point.

Run from the repo root:
    .venv\\Scripts\\python -m uvicorn app.main:app --app-dir backend --port 8000
or simply:  scripts\\run.ps1
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.ws import manager, token_matches
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
    services.conversation_busy = manager.busy    # background LLM jobs give way to the conversation
    app.state.services = services
    await services.start()
    if services.avatar is not None:
        # a restarted avatar worker numbers its frames from 0 again: the page starts its idle video over
        services.avatar.on_hello = lambda hello: manager.active and manager.active.avatar_restarted(hello)
    yield
    await manager.shutdown()                      # every session (also those waiting) and the manager's tasks
    await services.stop()                         # runs the memory/lead jobs those sessions queued first
    logger.info("backend stopped")


app = FastAPI(title="AI Avatar Voice Agent", lifespan=lifespan)
app.include_router(ws_router)


def _owner(request: Request) -> bool:
    """The detailed endpoints need the owner's token when one is set (remote access): through the tunnel
    they were open to anyone with the link's host name. Without a token (local use) they stay open."""
    token = app.state.services.settings.access_token
    return not token or token_matches(request.query_params.get("token", ""), token)


@app.get("/health")
async def health(request: Request) -> dict:
    services: Services = app.state.services
    if not _owner(request):
        return {"status": "ok"}
    return {
        "status": "ok",
        "profile": services.settings.profile,
        "llm_server_running": services.llm_server.running or not services.settings.llm.manage_server,
        "llm_restarts": services.llm_server.restarts,
        "llm_ready": services.llm_server.ready.is_set() or not services.settings.llm.manage_server,
        "active_session": manager.active.id if manager.active else None,
        "waiting_sessions": len(manager.waiting),
        "background_jobs": services.pending_background(),   # memory / lead jobs not finished yet
        "avatar": {"connected": services.avatar_ready, "restarts": services.avatar.restarts,
                   "hello": services.avatar.hello, "stats": services.avatar.stats} if services.avatar else None,
        "load_report": services.load_report,
        "warnings": services.warnings,
        "voice_ready": services.voice_ready,
        "tts_engine": services.settings.tts.engine if services.tts else None,
        "rag_chunks": services.rag.rows if services.rag else 0,
    }


@app.get("/health/gpu")
async def health_gpu(request: Request) -> dict:
    if not _owner(request):
        raise HTTPException(403, "owner token required")
    snap = gpu_snapshot()
    # our own processes: whole-GPU numbers include every other app on the desktop
    import torch

    if torch.cuda.is_initialized():
        snap["backend_torch_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    services = app.state.services
    if services.avatar_ready:
        snap["avatar_torch_reserved_mib"] = services.avatar.stats.get("worker_vram_reserved_mib")
    if snap.get("available") and snap["vram_free_mib"] < 400:
        snap["warning"] = "less than 400 MiB VRAM free: Windows may spill to system RAM (very slow)"
    return snap


@app.get("/api/metrics")
async def api_metrics(request: Request, last: int = 50) -> dict:
    """p50/p90 per pipeline stage over the last N completed turns (ms from end of user input)."""
    if not _owner(request):
        raise HTTPException(403, "owner token required")
    last = max(1, min(last, 2000))
    path = app.state.services.settings.log_dir / "metrics.jsonl"
    if not path.exists():
        return {"turns": 0}
    rows = await asyncio.to_thread(_read_metrics, path)     # (the file grows for ever: not on the event loop)
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


def _read_metrics(path, keep: int = 2000) -> list[dict]:
    """The last ``keep`` records of metrics.jsonl, reading only its end."""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - keep * 4096))               # a record is ~1-2 KB
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    if size > keep * 4096:
        lines = lines[1:]                                # (the first one is cut)
    rows = []
    for line in lines[-keep:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
