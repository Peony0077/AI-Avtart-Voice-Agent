"""WebSocket endpoint. One conversation at a time (one GPU renders one avatar).

Two links (2026-10-05, customers in the US and Canada use the page too):
  * owner link (?token=access_token): everything - settings, leads, voice preview
  * customer link (?token=customer_token): only the conversation; each customer has their own memory
Who gets the conversation: the owner's new page takes over from the owner's old one (a reload); a
customer's reconnect (same browser, ?visitor=) takes over their own session; anyone else waits in
line ({"type":"waiting","position":n}) and starts when the conversation before them ends - a customer
is never cut off by someone else. If people are waiting and the current conversation (also the owner's)
has been inactive for IDLE_HANDOVER_S, it ends (code 4002) so an abandoned tab cannot block the line.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.session.session import Session
from app.session.state_machine import State

router = APIRouter()
IDLE_HANDOVER_S = 90.0
BUSY_STATES = (State.END_OF_TURN, State.THINKING, State.SPEAKING)


class SessionManager:
    def __init__(self) -> None:
        self.active: Session | None = None
        self.waiting: list[Session] = []
        self._lock = asyncio.Lock()
        self._watch: asyncio.Task | None = None

    @staticmethod
    def takes_over(new: Session, current: Session | None) -> bool:
        if current is None:
            return True
        if new.role == "owner" and current.role == "owner":
            return True
        return bool(new.visitor_id) and new.visitor_id == current.visitor_id

    async def attach(self, session: Session) -> bool:
        """True: the session is active now (started); False: it waits in line."""
        async with self._lock:
            previous = self.active
            if not self.takes_over(session, previous):
                self.waiting.append(session)
                self._ensure_watch()
            else:
                self.active = session
                if session.svc.avatar is not None:
                    session.svc.avatar.frame_sink = session.on_avatar_frame
        if self.active is not session:
            await self._tell_positions()
            return False
        if previous is not None:
            await previous.close("replaced by a new connection")
            try:
                await previous.ws.close(code=4001, reason="replaced")
            except Exception:  # noqa: BLE001
                pass
        await session.start()
        return True

    async def detach(self, session: Session) -> None:
        promote: Session | None = None
        async with self._lock:
            if session in self.waiting:
                self.waiting.remove(session)        # never started: nothing to close
            elif self.active is session:
                self.active = None
                if session.svc.avatar is not None:
                    session.svc.avatar.frame_sink = None
                if self.waiting:
                    promote = self.waiting.pop(0)
                    self.active = promote
                    if promote.svc.avatar is not None:
                        promote.svc.avatar.frame_sink = promote.on_avatar_frame
        if session.started:
            await session.close("disconnected")
        if promote is not None:
            logger.info(f"session {promote.id}: the conversation before ended, starting ({len(self.waiting)} waiting)")
            await promote.start()
            if promote.pending_hello is not None:
                await promote.on_json(promote.pending_hello)
        await self._tell_positions()

    def is_active(self, session: Session) -> bool:
        return self.active is session

    def busy(self) -> bool:
        """The conversation is using (or about to use) the LLM: background LLM jobs wait (services.py)."""
        s = self.active
        if s is None or not s.started:
            return False
        return s.sm.state in BUSY_STATES or time.time() - s.last_activity < 2.0

    async def _tell_positions(self) -> None:
        for n, s in enumerate(list(self.waiting), start=1):
            try:
                await s.ws.send_text(json.dumps({"type": "waiting", "position": n}))
            except Exception:  # noqa: BLE001 - it leaves the line when its socket closes
                pass

    def _ensure_watch(self) -> None:
        if self._watch is None or self._watch.done():
            self._watch = asyncio.get_running_loop().create_task(self._idle_handover(), name="idle-handover")

    async def _idle_handover(self) -> None:
        while self.waiting:
            await asyncio.sleep(5)
            current = self.active
            if (current is not None and self.waiting
                    and time.time() - current.last_activity > IDLE_HANDOVER_S):
                logger.info(f"session {current.id}: inactive for {IDLE_HANDOVER_S:.0f} s with customers waiting: handed over")
                try:
                    await current.ws.close(code=4002, reason="inactive")
                except Exception:  # noqa: BLE001
                    pass


manager = SessionManager()


def role_for(token: str, owner_token: str | None, customer_token: str | None) -> str | None:
    """"owner", "customer" or None (refused). Without an owner token (local use) everyone is the owner."""
    if not owner_token:
        return "owner"
    if secrets.compare_digest(token, owner_token):
        return "owner"
    if customer_token and secrets.compare_digest(token, customer_token):
        return "customer"
    return None


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    services = ws.app.state.services
    services.conversation_busy = manager.busy
    await ws.accept()   # accept first: a close before accept reaches the browser only as code 1006
    role = role_for(ws.query_params.get("token", ""), services.settings.access_token, services.settings.customer_token)
    if role is None:
        await ws.close(code=4003, reason="invalid or missing access token")
        logger.warning(f"rejected websocket from {ws.client}: bad token")
        return
    visitor = ws.query_params.get("visitor", "")[:64] or None
    session = Session(ws, services, role=role, visitor_id=visitor)
    logger.info(f"session {session.id} connected from {ws.client} as {role} "
                f"({ws.headers.get('user-agent', '?')[:300]})")
    active = await manager.attach(session)
    if not active:
        logger.info(f"session {session.id}: waiting ({len(manager.waiting)} in line)")
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if not manager.is_active(session):
                if message.get("text") and '"hello"' in message["text"]:
                    session.pending_hello = json.loads(message["text"])   # applied when its turn comes
                continue                     # waiting in line: nothing to do yet
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
