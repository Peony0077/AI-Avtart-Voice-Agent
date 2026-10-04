"""Backend side of the avatar worker (P3): process supervision + WebSocket bridge.

* starts ``avatar/musetalk/.venv`` → ``worker.py`` (separate venv/process: MuseTalk deps
  must not touch the backend environment) and restarts it if it dies
* forwards avatar video frames to the active browser session
* ``speak()`` sends a turn's first TTS clause and resolves when its first lip-synced frames exist,
  so the session can hold the reply's audio until the video is ready (A/V sync at the start);
  ``send_audio()`` sends the later clauses without waiting (they play after queued audio)
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import Callable

import numpy as np
import websockets
from loguru import logger

from app.core.config import Settings
from app.core.procjob import JOB, free_port
from app.transport.protocol import HEADER, KIND_AUDIO, KIND_IDLE_VIDEO

FrameSink = Callable[[bytes, int, int], None]  # (packet, kind, turn_id)


class AvatarClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.avatar
        self.proc: subprocess.Popen | None = None
        self.ws = None
        self.connected = False
        self.hello: dict = {}
        self.frame_sink: FrameSink | None = None
        self._waiters: dict[tuple[int, int], asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._stopping = False
        self.restarts = 0
        self.stats = {"frames_in": 0, "idle_in": 0, "first_frame_ms": [], "segment_fps": [], "skipped_frames": 0, "strided_frames": 0,
                      "gestures": {}}

    # ------------------------------------------------------------------ lifecycle
    async def start(self, timeout_s: float = 240.0) -> None:
        if self.cfg.manage_worker:
            self._spawn()
        await self._connect(timeout_s)
        self._watchdog = asyncio.create_task(self._watch(), name="avatar-watchdog")

    def _spawn(self) -> None:
        py = self.settings.repo_path(self.cfg.worker_python)
        script = self.settings.repo_path("avatar/musetalk/worker.py")
        port = self.cfg.worker_url.rsplit(":", 1)[1].split("/")[0]
        cmd = [str(py), str(script), "--avatar", self.cfg.avatar_id, "--port", port,
               "--quality", str(self.cfg.jpeg_quality), "--gate-frames", str(self.cfg.gate_frames), "--decoder", self.cfg.decoder,
               "--warm-burst-ms", str(self.cfg.warm_burst_ms), "--late-ms", str(self.cfg.late_ms)]
        free_port(int(port), "worker.py")
        log = (self.settings.log_dir / "avatar-worker.log").open("ab")
        logger.info(f"starting avatar worker: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, cwd=script.parent, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        JOB.adopt(self.proc)

    async def _connect(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"avatar worker exited with {self.proc.returncode}; see logs/avatar-worker.log")
            try:
                self.ws = await websockets.connect(self.cfg.worker_url, max_size=None, ping_interval=20)
                self.hello = json.loads(await asyncio.wait_for(self.ws.recv(), 10))
                self.connected = True
                self._reader = asyncio.create_task(self._read(), name="avatar-reader")
                logger.info(f"avatar worker connected: {self.hello}")
                return
            except (OSError, websockets.WebSocketException, TimeoutError):
                await asyncio.sleep(1.0)
        raise TimeoutError("avatar worker did not come up")

    async def _watch(self) -> None:
        while not self._stopping:
            await asyncio.sleep(3.0)
            dead = self.proc is not None and self.proc.poll() is not None
            if (dead or not self.connected) and not self._stopping:
                self.restarts += 1
                logger.error(f"avatar worker down (restart #{self.restarts})")
                self.connected = False
                try:
                    if dead and self.cfg.manage_worker:
                        self._spawn()
                    await self._connect(240.0)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"avatar restart failed: {exc!r}")

    async def stop(self) -> None:
        self._stopping = True
        for task in (self._watchdog, self._reader):
            if task:
                task.cancel()
        if self.ws is not None:
            await self.ws.close()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ------------------------------------------------------------------ io
    async def _read(self) -> None:
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    kind, turn_id, _seq, _pts = HEADER.unpack_from(msg)
                    if kind == KIND_IDLE_VIDEO:
                        self.stats["idle_in"] += 1
                    else:
                        self.stats["frames_in"] += 1
                    if self.frame_sink is not None:
                        self.frame_sink(msg, kind, turn_id)
                    continue
                data = json.loads(msg)
                if data.get("type") == "first_frame":
                    fut = self._waiters.pop((data["turn_id"], data["seq"]), None)
                    if data["ms"] is not None:   # None: the segment had no frames to render
                        self.stats["first_frame_ms"] = (self.stats["first_frame_ms"] + [data["ms"]])[-50:]
                    if fut and not fut.done():
                        fut.set_result(data["ms"])
                elif data.get("type") == "gesture":
                    self.stats["gestures"][data["name"]] = self.stats["gestures"].get(data["name"], 0) + 1
                elif data.get("type") == "segment_done":
                    self.stats["skipped_frames"] += data.get("skipped", 0)   # incl. strided
                    self.stats["strided_frames"] += data.get("strided", 0)
                    if data.get("vram_reserved_mib") is not None:
                        self.stats["worker_vram_reserved_mib"] = data["vram_reserved_mib"]
                    if data.get("fps"):
                        self.stats["segment_fps"] = (self.stats["segment_fps"] + [data["fps"]])[-50:]
        except websockets.ConnectionClosed:
            pass
        finally:
            self.connected = False
            for fut in self._waiters.values():
                if not fut.done():
                    fut.set_result(None)
            self._waiters.clear()

    def send_json(self, obj: dict) -> None:
        if self.connected:
            asyncio.get_running_loop().create_task(self._safe_send(json.dumps(obj)))

    async def _safe_send(self, data) -> None:
        try:
            await self.ws.send(data)
        except (websockets.ConnectionClosed, AttributeError):
            self.connected = False

    def send_audio(self, turn_id: int, seq: int, pcm16_24k: np.ndarray, pts_ms: int) -> None:
        """Send a clause to lip-sync without waiting for its frames (all clauses after the first)."""
        if self.connected:
            asyncio.get_running_loop().create_task(
                self._safe_send(HEADER.pack(KIND_AUDIO, turn_id, seq, pts_ms) + pcm16_24k.tobytes()))

    async def speak(self, turn_id: int, seq: int, pcm16_24k: np.ndarray, pts_ms: int,
                    timeout_s: float = 1.5) -> float | None:
        """Send one clause; wait (bounded) until its first frame is rendered. Returns render ms."""
        if not self.connected:
            return None
        fut = asyncio.get_running_loop().create_future()
        self._waiters[(turn_id, seq)] = fut
        await self._safe_send(HEADER.pack(KIND_AUDIO, turn_id, seq, pts_ms) + pcm16_24k.tobytes())
        try:
            return await asyncio.wait_for(fut, timeout_s)
        except TimeoutError:
            self._waiters.pop((turn_id, seq), None)
            logger.warning(f"avatar first frame for turn {turn_id} seq {seq} not ready after {timeout_s}s")
            return None
