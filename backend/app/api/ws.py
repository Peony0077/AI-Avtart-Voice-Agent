"""WebSocket endpoint. One conversation at a time (one GPU renders one avatar).

Two links (2026-10-05, customers in the US and Canada use the page too):
  * owner link (?token=access_token): everything - settings, leads, voice preview
  * customer link (?token=customer_token): only the conversation; each customer has their own memory
Who gets the conversation: the owner's new page takes over from the owner's old one (a reload); a
customer's reconnect (same browser, ?visitor=, same link) takes over their own session; anyone else waits
in line ({"type":"waiting","position":n}) and starts when the conversation before them ends - a customer
is never cut off by someone else. If people are waiting and the current conversation (also the owner's)
has been inactive for IDLE_HANDOVER_S, it ends (code 4002) so an abandoned tab cannot block the line.

Dropped connections (2026-10-10): a tunnel hiccup closes the socket and the page reconnects half a second
later - by then the next one in line had the conversation, and the customer went to the back of the line.
Now a connection that drops (any close but the page's own normal close) keeps its place for GRACE_S: the
conversation's slot, or its place in the line, waits for the same visitor's reconnect before the next one
moves up. At most MAX_WAITING wait; more are refused with 1013 (try again later: the page retries).
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from starlette.websockets import WebSocketState

from app.session.session import Session
from app.session.state_machine import State

router = APIRouter()
IDLE_HANDOVER_S = 90.0
GRACE_S = 10.0              # a dropped connection's place is kept this long for its reconnect
GOING_AWAY = 1001           # the tab was closed or reloaded (browsers send the same code): a reload is back in
RELOAD_GRACE_S = 4.0        # ~1-2 s, so the next customer is not kept waiting 10 s after a closed tab
MAX_WAITING = 20            # a flood of tabs cannot grow the line (and the memory) without bound
NORMAL_CLOSE = 1000         # the page closed the connection itself: nothing to wait for
# What a page sends while it waits, applied in this order when its turn comes (the latest of each type).
# The rest - typed text, the microphone - is dropped: the page offers neither while it waits.
KEPT_WHILE_WAITING = ("hello", "visible", "voice", "speaker", "greet")
BUSY_STATES = (State.END_OF_TURN, State.THINKING, State.SPEAKING)


def identity(session) -> tuple[str, str] | None:
    """Who may take a session's place: any owner page the owner's, the same browser on the same link a
    customer's (the owner's browser opening the customer link is a customer). None: nobody."""
    if session.role == "owner":
        return ("owner", "")
    return ("customer", session.visitor_id) if session.visitor_id else None


def connected(session) -> bool:
    """Its socket is still open (a page that left is about to be detached by its own loop)."""
    ws = session.ws
    return (getattr(ws, "client_state", None) is not WebSocketState.DISCONNECTED
            and getattr(ws, "application_state", None) is not WebSocketState.DISCONNECTED)


class SessionManager:
    def __init__(self) -> None:
        self.active: Session | None = None
        self.waiting: list[Session] = []
        self._lock = asyncio.Lock()
        self._watch: asyncio.Task | None = None
        self._held: Session | None = None           # left with the conversation; its slot is kept (GRACE_S)
        self._grace: asyncio.Task | None = None
        self._places: dict[tuple[str, str], tuple[float, int]] = {}   # left the line: identity -> (until, index)
        self._handed_over: Session | None = None    # ended for inactivity (4002): its place is not kept

    @staticmethod
    def takes_over(new, current) -> bool:
        if current is None:
            return True
        mine = identity(new)
        return mine is not None and mine == identity(current)

    async def attach(self, session: Session) -> bool | None:
        """True: the session is active now (started); False: it waits in line; None: refused (line full)."""
        async with self._lock:
            previous = self.active
            current = previous if previous is not None else self._held
            mine = self.takes_over(session, current)
            if not mine:
                place = self._place_for(session)
                if place is None:
                    return None
                self.waiting.insert(place, session)
                self._ensure_watch()
            else:
                if current is not None and current is self._held:
                    logger.info(f"session {session.id}: reconnected, continues in the place of {current.id}")
                    self._end_hold()
                self._activate(session)
        if not mine:
            await self._tell_positions()
            return False
        if previous is not None:
            await previous.close("replaced by a new connection")
            try:
                await previous.ws.close(code=4001, reason="replaced")
            except Exception:  # noqa: BLE001
                pass
        return await self._start(session)

    async def detach(self, session: Session, code: int | None = None) -> None:
        """``session``'s socket closed (``code``: its close code)."""
        promote: Session | None = None
        async with self._lock:
            keep = code != NORMAL_CLOSE and session is not self._handed_over and identity(session) is not None
            grace = RELOAD_GRACE_S if code == GOING_AWAY else GRACE_S
            if self._handed_over is session:
                self._handed_over = None
            if session in self.waiting:
                index = self.waiting.index(session)
                self.waiting.remove(session)        # never started: nothing to close
                if keep:
                    self._places[identity(session)] = (time.time() + grace, index)
            elif self.active is session:
                self.active = None
                if session.svc.avatar is not None:
                    session.svc.avatar.frame_sink = None
                if keep and session.started:
                    self._hold(session, grace)       # its reconnect continues; the next in line waits a moment
                else:
                    if keep:                         # its turn had just come: first in line again
                        self._places[identity(session)] = (time.time() + grace, 0)
                    promote = self._next()
        if session.started:
            await session.close("disconnected")
        if promote is not None:
            await self._promote(promote)
        await self._tell_positions()

    def is_active(self, session: Session) -> bool:
        return self.active is session

    def busy(self) -> bool:
        """The conversation is using (or about to use) the LLM: background LLM jobs wait (services.py)."""
        s = self.active
        if s is None or not s.started:
            return False
        return s.sm.state in BUSY_STATES or time.time() - s.last_activity < 2.0

    async def shutdown(self) -> None:
        """Server stopping: every conversation ends (its memory and lead jobs are queued, services.stop
        runs them) and the pages waiting in line are told."""
        for task in (self._watch, self._grace):
            if task is not None and not task.done():
                task.cancel()
        async with self._lock:
            sessions = [s for s in (self.active, *self.waiting) if s is not None]
            self.active, self.waiting, self._held = None, [], None
        for s in sessions:
            if s.started:
                await s.close("shutdown")
            try:
                await s.ws.close(code=1012, reason="server restarting")
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- internals (under the lock)
    def _activate(self, session: Session) -> None:
        self.active = session
        if session.svc.avatar is not None:
            session.svc.avatar.frame_sink = session.on_avatar_frame

    def _place_for(self, session: Session) -> int | None:
        """Where a newcomer joins the line: at its old place if it left less than GRACE_S ago, else at the
        end; None if the line is full."""
        now = time.time()
        self._places = {k: v for k, v in self._places.items() if v[0] > now}
        kept = self._places.pop(key, None) if (key := identity(session)) is not None else None
        if kept is not None:
            return min(kept[1], len(self.waiting))
        return len(self.waiting) if len(self.waiting) < MAX_WAITING else None

    def _next(self) -> Session | None:
        """The first one in line whose page is still there becomes active (not started yet: _promote)."""
        for s in self.waiting:
            if connected(s):
                self.waiting.remove(s)
                self._activate(s)
                return s
        return None

    def _hold(self, session: Session, grace: float = GRACE_S) -> None:
        self._held = session
        self._grace = asyncio.get_running_loop().create_task(self._grace_over(session, grace), name="reconnect-grace")
        logger.info(f"session {session.id}: connection lost; its place is kept {grace:.0f} s for a reconnect")

    def _end_hold(self) -> None:
        self._held = None
        if self._grace is not None and not self._grace.done():
            self._grace.cancel()
        self._grace = None

    # ---------------------------------------------------------------- internals (tasks)
    async def _grace_over(self, session: Session, grace: float = GRACE_S) -> None:
        await asyncio.sleep(grace)
        async with self._lock:
            if self._held is not session:
                return
            self._held, self._grace = None, None
            promote = self._next()
        logger.info(f"session {session.id}: no reconnect within {grace:.0f} s")
        if promote is not None:
            await self._promote(promote)
        await self._tell_positions()

    async def _promote(self, session: Session) -> None:
        logger.info(f"session {session.id}: the conversation before ended, starting ({len(self.waiting)} waiting)")
        if not await self._start(session):
            logger.info(f"session {session.id}: left (or was replaced) before its turn came")

    async def _start(self, session: Session) -> bool:
        """Start ``session`` if the conversation is still its own and its page still there: while the one
        before was being closed it may have left (its own detach moved the line on) or been replaced. Then
        what its page sent while it waited; its messages are handled only after that (``ready``)."""
        if self.active is not session or not connected(session):
            return False
        await session.start()
        while session.pending and self.active is session:
            kind = next(k for k in KEPT_WHILE_WAITING if k in session.pending)
            try:
                await session.on_json(session.pending.pop(kind))
            except Exception:  # noqa: BLE001
                logger.exception(f"session {session.id}: message failed")
        session.ready = True
        return self.active is session

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
            # (not started yet: its last_activity is from when it joined the line)
            if (current is not None and current.started and self.waiting
                    and time.time() - current.last_activity > IDLE_HANDOVER_S):
                logger.info(f"session {current.id}: inactive for {IDLE_HANDOVER_S:.0f} s with customers waiting: handed over")
                self._handed_over = current
                try:
                    await current.ws.close(code=4002, reason="inactive")
                except Exception:  # noqa: BLE001
                    pass


manager = SessionManager()


def keep_for_later(session: Session, message: dict) -> None:
    """A message from a page that is not in the conversation yet (waiting in line, or being started):
    the latest of each KEPT_WHILE_WAITING type is applied when its turn comes."""
    text = message.get("text")
    if not text or len(text) > 8192:
        return
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(msg, dict) and msg.get("type") in KEPT_WHILE_WAITING:
        session.pending[msg["type"]] = msg


def token_matches(given: str, expected: str) -> bool:
    """Constant-time comparison; as bytes, so a non-ASCII ?token= is a mismatch instead of a TypeError."""
    return secrets.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def role_for(token: str, owner_token: str | None, customer_token: str | None) -> str | None:
    """"owner", "customer" or None (refused). Without an owner token (local use) everyone is the owner -
    unless a customer token is set: that configuration is refused at start-up (config.py), and here too."""
    if not owner_token:
        return None if customer_token else "owner"
    if token_matches(token, owner_token):
        return "owner"
    if customer_token and token_matches(token, customer_token):
        return "customer"
    return None


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    services = ws.app.state.services
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
    if active is None:
        logger.warning(f"session {session.id}: refused, {MAX_WAITING} already waiting")
        await ws.close(code=1013, reason="the line is full, please try again in a few minutes")
        return
    if not active:
        logger.info(f"session {session.id}: waiting ({len(manager.waiting)} in line)")
    code: int | None = None
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                code = message.get("code")
                break
            if not (manager.is_active(session) and session.ready):
                keep_for_later(session, message)     # waiting in line, or not started yet
                continue
            try:
                if message.get("bytes") is not None:
                    await session.on_binary(message["bytes"])
                elif message.get("text") is not None:
                    msg = json.loads(message["text"])
                    if isinstance(msg, dict):
                        await session.on_json(msg)
            except json.JSONDecodeError:
                logger.warning("ignored non-JSON text frame")
            except WebSocketDisconnect:
                raise
            except Exception:  # noqa: BLE001 - one bad message used to close the socket (1011) and lose the session
                logger.exception(f"session {session.id}: message failed")
    except WebSocketDisconnect as exc:
        code = exc.code
    finally:
        await manager.detach(session, code)
