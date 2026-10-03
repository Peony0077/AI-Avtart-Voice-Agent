"""Avatar worker process (P3): streams avatar video frames to the backend.

Protocol (WebSocket, one client = the backend):
  backend → worker
    binary  16-byte header (kind=1, turn_id, seq, pts_ms) + PCM16 mono 24 kHz   one TTS clause
    json    {"type":"behavior","expression":...} | {"type":"state","state":...}
            {"type":"playback","turn_id":..,"event":"started"|"ended"}
            {"type":"cancel","turn_id":..}
  worker → backend
    binary  header(kind=3 idle | kind=2 speech, turn_id, seq, pts_ms) + JPEG
    json    {"type":"hello",...} | {"type":"first_frame","turn_id","seq"} | {"type":"segment_done",...}

Idle frames are paced at fps. Speech frames are rendered as fast as possible and
carry a pts on the turn's audio timeline; the browser shows them by audio clock.

Run: avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\worker.py --avatar dev_girl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import psutil
import torch
import websockets
from loguru import logger

from catchup import audible_ms, batch_frames, first_on_time, stride_for
from common import AVATARS
from engine import Loop, LoopDelta, MuseTalkEngine
from gestures import choose_gesture, cycle_windows, earliest_start, next_window
from motion import EXPRESSION_TO_LOOP

HEADER = struct.Struct("<B3xIII")
KIND_AUDIO, KIND_VIDEO, KIND_IDLE = 1, 2, 3
CONTEXT_FRAMES = 4   # 200 ms at 20 fps
PLAYBACK_SILENT_S = 5.0   # speaking turn without a playback report for this long = playback stopped
SMALL_BATCH = 4           # batch size while the renderer is only a little ahead of the audio
STATE_TO_LOOP = {"THINKING": "thinking", "LISTENING": "friendly", "IDLE": "friendly", "END_OF_TURN": "friendly"}  # warm idle


@dataclass
class Turn:
    turn_id: int
    base_idx: int
    loop: str
    rendered: int = 0
    playing: bool = False
    cancelled: bool = False
    first_frame_sent: set = field(default_factory=set)
    tail: np.ndarray | None = None     # last CONTEXT_FRAMES of audio, left context for the next segment
    next_pts: int = -1                 # where the next segment starts if the audio is contiguous
    played_ms: float = 0.0             # browser playback position on the turn's timeline ...
    played_at: float = 0.0             # ... reported at this perf_counter() time

    def audible_ms(self, at: float) -> float:
        """Playback position at time ``at`` (see catchup.audible_ms)."""
        return audible_ms(self.played_ms, self.played_at, time.perf_counter(), at)


class AvatarWorker:
    def __init__(self, avatar_id: str, batch: int, quality: int, lead_ms: int, gate_frames: int = 4,
                 warm_burst_ms: int = 0, late_ms: int = 100) -> None:
        folder = AVATARS / avatar_id
        meta = json.loads((folder / "avatar.json").read_text(encoding="utf-8"))
        self.fps = meta["fps"]
        self.loops = {name: Loop.load(folder / "loops" / name) for name in meta["loops"]}
        # head gestures: variants of the expression loops that differ only inside gesture windows
        n_forward = len(next(iter(self.loops.values()))) // 2
        self.cycle = 2 * n_forward
        self.variants: dict[tuple[str, str], Loop] = {}
        self.gesture_windows: dict[str, list[tuple[int, int]]] = {}
        for gesture, gm in meta.get("gestures", {}).items():
            for expression in gm["expressions"]:
                vdir = folder / "loops" / f"{expression}__{gesture}"
                if expression in self.loops and (vdir / "delta.pkl").exists():
                    self.variants[(expression, gesture)] = self.loops[expression].with_delta(LoopDelta.load(vdir))
            self.gesture_windows[gesture] = cycle_windows([tuple(w) for w in gm["windows"]], n_forward)
        self._gesture: tuple[str, int, int] | None = None      # (gesture, first frame, end frame), absolute
        self._reply_gesture: tuple[str, float] | None = None   # requested by the LLM for the next reply
        self.engine = MuseTalkEngine(fps=self.fps, batch_size=batch)
        self.warmup_s = self.engine.warmup(next(iter(self.loops.values())))
        self.quality = quality
        self.gate_frames = gate_frames
        self.warm_burst_ms = warm_burst_ms
        self.late_ms = late_ms
        self._frame_ms = 45.0                      # running estimates: GPU time per frame,
        self._compose_ms = 12.0                    # CPU compositing + JPEG per frame
        self._warming = False
        self.lead_frames = round(lead_ms * self.fps / 1000)
        self.idx = 0
        self.idle_loop = "friendly" if "friendly" in self.loops else next(iter(self.loops))
        self.speech_loop = self.idle_loop
        self._fade: tuple[str, int] | None = None   # (previous loop, frames left)
        self.turn: Turn | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.ws = None
        self.stats = {"segments": 0, "frames": 0, "render_ms": 0.0, "late_first_frames": 0, "skipped": 0, "strided": 0,
                      "gestures": {}}
        logger.info(f"avatar {avatar_id}: {len(self.loops)} loops x {len(self.idle_loop and self.loops[self.idle_loop])} frames, "
                    f"fps {self.fps}, warmup {self.warmup_s}s, {len(self.variants)} gesture variants")

    # ------------------------------------------------------------------ protocol
    async def handle(self, ws) -> None:
        if self.ws is not None:
            await ws.close(code=4001, reason="busy")
            return
        self.ws = ws
        await ws.send(json.dumps({"type": "hello", "fps": self.fps, "loops": list(self.loops),
                                  "frames": len(self.loops[self.idle_loop]), "size": self.loops[self.idle_loop].size,
                                  "load_s": self.engine.load_s, "warmup_s": self.warmup_s,
                                  "gestures": sorted({g for _, g in self.variants})}))
        tasks = [asyncio.create_task(self._idle()), asyncio.create_task(self._render())]
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    kind, turn_id, seq, pts = HEADER.unpack_from(msg)
                    if kind == KIND_AUDIO:
                        self._on_audio(turn_id, seq, pts, np.frombuffer(msg[HEADER.size:], np.int16))
                else:
                    self._on_json(json.loads(msg))
        except websockets.ConnectionClosed:
            pass
        finally:
            for t in tasks:
                t.cancel()
            self.ws, self.turn = None, None
            logger.info("backend disconnected")

    def _on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "behavior":
            loop = EXPRESSION_TO_LOOP.get(msg.get("expression", ""), None)
            if loop in self.loops:
                self.speech_loop = loop
                self._switch_idle(loop)
            gesture, intensity = msg.get("gesture"), float(msg.get("intensity") or 0.0)
            if str(msg.get("source", "")).startswith("state"):
                # e.g. a small nod when the avatar starts listening: at most one every ~5 s
                if self.turn is None or not self.turn.playing:
                    self._schedule_gesture(gesture, intensity, self.idx + 2, self.idle_loop, min_gap=5 * self.fps)
            else:
                self._reply_gesture = (gesture, intensity)          # played at the start of the reply
        elif kind == "state":
            if msg.get("state") == "THINKING":
                # the first clause arrives a few hundred ms later; a GPU that sat idle is measurably
                # slower for its first batches (bench_idle.py: ~650 → ~425 ms), so warm it now.
                # STT has already finished at THINKING, so this does not compete with it.
                if self.warm_burst_ms:
                    asyncio.get_running_loop().create_task(self._warm_burst(self.warm_burst_ms))
            loop = STATE_TO_LOOP.get(msg.get("state", ""))
            if loop in self.loops and (self.turn is None or not self.turn.playing):
                self._switch_idle(loop)
        elif kind == "playback" and self.turn and msg.get("turn_id") == self.turn.turn_id:
            if msg.get("event") in ("started", "progress"):
                self.turn.playing = True
                self.turn.played_ms = float(msg.get("played_ms") or 0)
                self.turn.played_at = time.perf_counter()
            elif msg.get("event") == "ended":
                self.turn = None
        elif (kind == "cancel" and self.turn and msg.get("turn_id") == self.turn.turn_id) or kind == "reset":
            # reset: the browser session ended (tab closed while the avatar spoke) - back to idle
            if self.turn is not None:
                self.turn.cancelled = True
            self.turn = None
            while not self.queue.empty():
                self.queue.get_nowait()
        elif kind == "stats":
            asyncio.create_task(self._send_json({"type": "stats", **self._stats()}))

    def _on_audio(self, turn_id: int, seq: int, pts: int, pcm: np.ndarray) -> None:
        if self.turn is None or self.turn.turn_id != turn_id:
            if self.turn is not None and turn_id < self.turn.turn_id:
                return  # stale
            self.turn = Turn(turn_id, base_idx=self.idx + self.lead_frames, loop=self.speech_loop)
            logger.info(f"turn {turn_id}: speaking with loop '{self.speech_loop}' from frame {self.turn.base_idx}")
            if self._reply_gesture is not None:
                gesture, intensity = self._reply_gesture
                self._reply_gesture = None
                self._schedule_gesture(gesture, intensity, self.turn.base_idx, self.turn.loop)
        self.queue.put_nowait((self.turn, seq, pts, pcm))

    async def _warm_burst(self, ms: float = 150.0) -> None:
        if self._warming:
            return
        self._warming = True
        try:
            loop = self.loops[self.speech_loop]
            chunks = torch.zeros(2, 50, 384)
            end = time.perf_counter() + ms / 1000
            while time.perf_counter() < end and self.queue.empty():
                await asyncio.to_thread(self.engine.faces, loop, chunks, 0)
        finally:
            self._warming = False

    def _schedule_gesture(self, name: str | None, intensity: float, after: int, loop_name: str,
                          min_gap: int = 0) -> None:
        """Play ``name`` in the next gesture window starting at or after frame ``after``."""
        available = {g for e, g in self.variants if e == loop_name}
        chosen = choose_gesture(name, intensity, available)
        if chosen is None or chosen not in self.gesture_windows:
            return
        after = earliest_start(self._gesture, self.idx, after, state_default=min_gap > 0, min_gap=min_gap)
        if after is None:
            return
        start, end = next_window(after, self.gesture_windows[chosen], self.cycle)
        self._gesture = (chosen, start, end)
        self.stats["gestures"][chosen] = self.stats["gestures"].get(chosen, 0) + 1
        logger.info(f"gesture {chosen} (requested {name}, intensity {intensity}) at frames {start}-{end}")
        asyncio.get_running_loop().create_task(self._send_json(
            {"type": "gesture", "name": chosen, "requested": name, "in_ms": round((start - self.idx) * 1000 / self.fps)}))

    def _source(self, loop_name: str, i: int) -> Loop:
        """The loop that provides frame ``i``: the gesture variant inside a scheduled gesture window."""
        g = self._gesture
        if g is not None and g[1] <= i < g[2]:
            variant = self.variants.get((loop_name, g[0]))
            if variant is not None:
                return variant
        return self.loops[loop_name]

    def _switch_idle(self, loop: str) -> None:
        if loop != self.idle_loop:
            self._fade = (self.idle_loop, 6)
            self.idle_loop = loop

    # ------------------------------------------------------------------ idle stream
    async def _idle(self) -> None:
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        while True:
            next_t += period
            await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
            if next_t < time.perf_counter() - 1.0:       # fell far behind (e.g. system stall): resync
                next_t = time.perf_counter()
            turn = self.turn
            if turn is not None and turn.playing and time.perf_counter() - turn.played_at > PLAYBACK_SILENT_S:
                # the browser reports playback every ~100 ms; silence means it stopped without saying
                # so (connection lost) - never leave the avatar frozen without idle frames
                logger.warning(f"turn {turn.turn_id}: no playback report for {PLAYBACK_SILENT_S:.0f} s, back to idle")
                turn.cancelled = True
                self.turn = None
            if self.turn is None or not self.turn.playing:
                loop = self.loops[self.idle_loop]
                if self._fade:
                    prev, left = self._fade
                    alpha = 1 - left / 7
                    blend = cv2.addWeighted(self.loops[prev].frame(self.idx), 1 - alpha, loop.frame(self.idx), alpha, 0)
                    jpg = cv2.imencode(".jpg", blend, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
                    self._fade = (prev, left - 1) if left > 1 else None
                else:
                    src = self._source(self.idle_loop, self.idx)
                    jpg = src.frames_jpg[self.idx % len(src)]
                await self._send(HEADER.pack(KIND_IDLE, 0, self.idx, 0) + jpg)
            self.idx += 1

    # ------------------------------------------------------------------ speech rendering
    async def _render(self) -> None:
        while True:
            turn, seq, pts, pcm = await self.queue.get()
            if turn.cancelled:
                continue
            try:
                await self._render_segment(turn, seq, pts, pcm)
            except Exception:  # noqa: BLE001 - one bad segment must not stop the render loop for good
                logger.exception(f"turn {turn.turn_id} seq {seq}: render failed")
                await self._release(turn, seq)

    async def _release(self, turn: Turn, seq: int) -> None:
        """Tell the backend not to wait for frames of this segment (none will come)."""
        if seq not in turn.first_frame_sent:
            turn.first_frame_sent.add(seq)
            await self._send_json({"type": "first_frame", "turn_id": turn.turn_id, "seq": seq, "ms": None})

    async def _render_segment(self, turn: Turn, seq: int, pts: int, pcm: np.ndarray) -> None:
        """Two-stage pipeline: the GPU stage of batch k+1 runs while the CPU composites batch k.

        Streaming TTS (clone voice) sends ~1 s chunks that split words. Features are computed per
        segment with zero padding at the edges, which would close the lips at every boundary, so a
        contiguous segment gets the previous segment's last 200 ms as left context (those frames
        are computed and dropped). Segments shorter than one frame produce no frames.
        """
        b = self.engine.batch_size
        t0 = time.perf_counter()
        spf = 24000 // self.fps                       # PCM samples per video frame
        contiguous = turn.tail is not None and abs(pts - turn.next_pts) <= 2
        ctx = CONTEXT_FRAMES if contiguous else 0
        audio = np.concatenate([turn.tail, pcm]) if contiguous else pcm
        turn.next_pts = pts + round(len(pcm) / 24)
        turn.tail = audio[-CONTEXT_FRAMES * spf:] if len(audio) >= CONTEXT_FRAMES * spf else None
        if len(audio) // spf - ctx <= 0:
            await self._release(turn, seq)
            return
        chunks = (await asyncio.to_thread(self.engine.audio_features, audio))[ctx:]
        loop = self.loops[turn.loop]
        n = chunks.shape[0]
        if n == 0:
            await self._release(turn, seq)
            return
        # First clause of a turn: release the audio only when gate_frames frames exist, and ramp the
        # batch size (gate frames, 4, then b) so rendering stays ahead of playback at turn start
        # (measured: a 2-frame gate followed by a batch of 8 let frames arrive up to 368 ms late).
        # The first batch is exactly the gate: one batch of 4 opens a 4-frame gate in 243 ms; the
        # earlier 2 + 4 ramp rendered 6 frames before the gate could open (337 ms).
        first_seg = turn.rendered == 0
        ramp = [self.gate_frames, 4] if first_seg else []
        gate = min(n, self.gate_frames if first_seg else 1)
        pending: asyncio.Task | None = None
        base = turn.base_idx + turn.rendered
        skipped = strided = 0
        s = 0
        while s < n:
            if turn.cancelled:
                break
            ramped = bool(ramp)
            size = ramp.pop(0) if ramp else b
            stride = 1
            if ramped and s >= gate:
                # the ramp batch after the gate: the audio starts now, so a GPU slower than real time
                # (LLM still writing the reply) would put these frames behind at once
                stride = stride_for(self._frame_ms, self.fps)
            elif size == b and turn.playing:
                # A batch of b delays its first frame by b GPU frames (~b x 45 ms). Since the later sentences
                # no longer wait for their frames (no gaps in the speech), the renderer is often only a
                # little ahead of the audio: then smaller batches keep the frames on time instead of
                # skipped (measured: 11 % of the frames of a 5-sentence reply skipped with batches of 8).
                lead = pts + s * 1000 / self.fps - turn.audible_ms(time.perf_counter())
                if lead < b * self._frame_ms + self._compose_ms + self.late_ms:
                    size = SMALL_BATCH
                    stride = stride_for(self._frame_ms, self.fps)   # GPU slower than real time: every 2nd/3rd frame
            if self.late_ms and turn.playing:
                # Catch up instead of drifting (catchup.py): skip frames that could only reach the
                # browser more than late_ms after their audio; the batch keeps its size.
                k = first_on_time(s, size, n, pts, self.fps, self.late_ms, turn.audible_ms(time.perf_counter()),
                                  self._frame_ms, self._compose_ms, stride)
                if k > s:
                    logger.debug(f"turn {turn.turn_id} seq {seq}: skip frames {s}-{k - 1} of {n} (pts {pts}, audible "
                                 f"{turn.audible_ms(time.perf_counter()):.0f}, gpu {self._frame_ms:.0f} ms/frame)")
                    skipped += k - s
                    s = k
                    if s >= n:
                        break
            frames = batch_frames(s, size, n, stride)
            t_gpu = time.perf_counter()
            srcs = [self._source(turn.loop, base + k) for k in frames]
            idx, faces = await asyncio.to_thread(self.engine.faces, loop, chunks[frames], base + s, srcs,
                                                 [base + k for k in frames])
            # running estimate: quick to rise (the LLM starts writing at turn start, 45 -> 100+ ms per frame,
            # and a slow estimate chose too small a stride for the first batches), slow to fall
            ms = (time.perf_counter() - t_gpu) * 1000 / len(frames)
            a = 0.6 if ms > self._frame_ms else 0.2
            self._frame_ms = (1 - a) * self._frame_ms + a * ms
            if pending is not None:
                await pending
            pending = asyncio.create_task(self._compose_and_send(turn, seq, pts, srcs, idx, faces, frames, t0, gate))
            e = min(n, frames[-1] + stride)
            skipped += (e - s) - len(frames)            # left out by the stride
            strided += (e - s) - len(frames)
            s = e
        if pending is not None:
            await pending
        if seq not in turn.first_frame_sent:
            await self._release(turn, seq)          # everything was skipped: do not hold the audio back
        turn.rendered += n
        self.stats["skipped"] += skipped
        self.stats["strided"] += strided
        ms = (time.perf_counter() - t0) * 1000
        self.stats["segments"] += 1
        self.stats["frames"] += n
        self.stats["render_ms"] += ms
        await self._send_json({"type": "segment_done", "turn_id": turn.turn_id, "seq": seq, "frames": n,
                               "skipped": skipped, "strided": strided, "render_ms": round(ms, 1), "audio_ms": round(len(pcm) / 24),
                               "vram_reserved_mib": round(torch.cuda.memory_reserved() / 2**20),
                               "fps": round(n / (ms / 1000), 1) if ms else None})

    async def _compose_and_send(self, turn: Turn, seq: int, pts: int, srcs: list[Loop], idx: list[int],
                                faces: np.ndarray, frames: list[int], t0: float, gate: int) -> None:
        """CPU stage. Each frame goes out as soon as it is composited: waiting for the whole batch of 8
        (~80 ms) made the first frames of every batch the ones most likely to arrive late.
        ``frames``: the frames' positions in the segment (every 2nd/3rd when the GPU is too slow)."""
        for k, (i, face) in enumerate(zip(idx, faces, strict=True)):
            jpg = await asyncio.to_thread(self._compose_one, srcs[k], i, face)
            if turn.cancelled:
                return
            frame_pts = pts + round(frames[k] * 1000 / self.fps)
            await self._send(HEADER.pack(KIND_VIDEO, turn.turn_id, seq, frame_pts) + jpg)
            if seq not in turn.first_frame_sent and frames[k] + 1 >= gate:
                turn.first_frame_sent.add(seq)
                await self._send_json({"type": "first_frame", "turn_id": turn.turn_id, "seq": seq,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    def _compose_one(self, loop: Loop, i: int, face: np.ndarray) -> bytes:
        t = time.perf_counter()
        frame = self.engine.compose(loop, i, face)
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
        self._compose_ms = 0.8 * self._compose_ms + 0.2 * (time.perf_counter() - t) * 1000
        return jpg

    # ------------------------------------------------------------------ io
    async def _send(self, data: bytes) -> None:
        if self.ws is not None:
            try:
                await self.ws.send(data)
            except websockets.ConnectionClosed:
                pass

    async def _send_json(self, obj: dict) -> None:
        await self._send_text(json.dumps(obj))

    async def _send_text(self, text: str) -> None:
        if self.ws is not None:
            try:
                await self.ws.send(text)
            except websockets.ConnectionClosed:
                pass

    def _stats(self) -> dict:
        s = dict(self.stats)
        s["render_fps"] = round(s["frames"] / (s["render_ms"] / 1000), 1) if s["render_ms"] else None
        s["vram_alloc_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        s["vram_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
        s["rss_mib"] = round(psutil.Process().memory_info().rss / 2**20)
        return s


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="dev_girl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--lead-ms", type=int, default=250, help="expected delay from first audio to playback start")
    ap.add_argument("--gate-frames", type=int, default=4, help="frames rendered before a turn's first audio is released")
    ap.add_argument("--warm-burst-ms", type=int, default=0, help="GPU warm-up when the turn enters THINKING (0 = off)")
    ap.add_argument("--late-ms", type=int, default=100, help="skip frames that would arrive this late (0 = never skip)")
    args = ap.parse_args()
    worker = AvatarWorker(args.avatar, args.batch, args.quality, args.lead_ms, args.gate_frames, args.warm_burst_ms,
                          args.late_ms)
    async with websockets.serve(worker.handle, args.host, args.port, max_size=None, ping_interval=20):
        logger.info(f"avatar worker listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
