"""WebSocket endpoint. Only one active session at a time (single GPU):
a new connection takes over and the previous one is closed."""
from __future__ import annotations

import asyncio
import json
import secrets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.session.session import Session

router = APIRouter()


class SessionManager:
    def __init__(self) -> None:
        self.active: Session | None = None
        self._lock = asyncio.Lock()

    async def attach(self, session: Session) -> None:
        async with self._lock:
            previous, self.active = self.active, session
            if session.svc.avatar is not None:
                session.svc.avatar.frame_sink = session.on_avatar_frame
        if previous is not None:
            await previous.close("replaced by a new connection")
            try:
                await previous.ws.close(code=4001, reason="replaced")
            except Exception:  # noqa: BLE001
                pass

    async def detach(self, session: Session) -> None:
        async with self._lock:
            if self.active is session:
                self.active = None
                if session.svc.avatar is not None:
                    session.svc.avatar.frame_sink = None
        await session.close("disconnected")


manager = SessionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    services = ws.app.state.services
    token = services.settings.access_token
    await ws.accept()   # accept first: a close before accept reaches the browser only as code 1006
    if token and not secrets.compare_digest(ws.query_params.get("token", ""), token):
        await ws.close(code=4003, reason="invalid or missing access token")
        logger.warning(f"rejected websocket from {ws.client}: bad token")
        return
    session = Session(ws, services)
    await manager.attach(session)
    await session.start()
    logger.info(f"session {session.id} connected from {ws.client}")
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.on_binary(message["bytes"])
            elif message.get("text") is not None:
                try:
                    await session.on_json(json.loads(message["text"]))
                except json.JSONDecodeError:
                    logger.warning("ignored non-JSON text frame")
    except WebSocketDisconnect:
        pass
    finally:
        await manager.detach(session)
