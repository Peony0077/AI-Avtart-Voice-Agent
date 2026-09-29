"""Single-writer outbound queue for one WebSocket.

Starlette WebSockets must not be written from several tasks at once, and stale
data from an interrupted turn must never reach the browser. Every item carries
the turn it belongs to; data items of a turn that is no longer current are
dropped right before sending. Control messages (``force=True``) always go out.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import WebSocket
from loguru import logger


@dataclass
class _Item:
    payload: dict | bytes
    turn_id: int
    force: bool


class Outbox:
    def __init__(self, ws: WebSocket, is_current: Callable[[int], bool]) -> None:
        self._ws = ws
        self._is_current = is_current
        self._queue: asyncio.Queue[_Item | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.sent_bytes = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="outbox")

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    def json(self, msg: dict, turn_id: int = 0, force: bool = False) -> None:
        self._queue.put_nowait(_Item(msg, turn_id, force))

    def binary(self, data: bytes, turn_id: int = 0, force: bool = False) -> None:
        self._queue.put_nowait(_Item(data, turn_id, force))

    @property
    def backlog(self) -> int:
        return self._queue.qsize()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if item.turn_id and not item.force and not self._is_current(item.turn_id):
                self.dropped += 1
                continue
            try:
                if isinstance(item.payload, bytes):
                    await self._ws.send_bytes(item.payload)
                    self.sent_bytes += len(item.payload)
                else:
                    await self._ws.send_json(item.payload)
            except Exception as exc:  # noqa: BLE001 - socket closed; stop quietly
                logger.debug(f"outbox stopped: {exc!r}")
                return
