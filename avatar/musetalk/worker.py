"""Avatar worker process (P3): streams avatar video frames to the backend.

Protocol (WebSocket, one client = the backend):
  backend → worker
    binary  16-byte header (kind=1, turn_id, seq, pts_ms) + PCM16 mono 24 kHz   one TTS clause
    json    {"type":"behavior","expression":...} | {"type":"state","state":...}
            {"type":"playback","turn_id":..,"event":"started"|"ended"}
            {"type":"speech_end","turn_id":..,"total_ms":..}   last clause sent: render the closing tail
            {"type":"cancel","turn_id":..}
  worker → backend
    binary  header(kind=3 idle | kind=2 speech, turn_id, frame, pts_ms) + JPEG
            frame = absolute loop frame number (idle and speech share it, so the browser can join
            the idle video to the end of a reply without a jump)
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
import threading
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
from h264 import H264Stream, ffmpeg_exe
from motion import EXPRESSION_TO_LOOP
from speech_body import breath_levels, frame_rms, lift_body, settle, to_lift

HEADER = struct.Struct("<B3xIII")
VIDEO_HEADER = struct.Struct("<BB2xIII")    # speech frames: byte 1 = flags (FLAG_TAIL: a closing-tail frame)
FLAG_TAIL = 1
FLAG_KEY = 2              # H.264: a keyframe (a decoder can start here)
KIND_AUDIO, KIND_VIDEO, KIND_IDLE = 1, 2, 3
KIND_VIDEO_H264, KIND_IDLE_H264 = 4, 5   # the same frames as H.264 access units (h264.py)
CONTEXT_FRAMES = 4   # 200 ms at 20 fps
PLAYBACK_SILENT_S = 5.0   # speaking turn without a playback report for this long = playback stopped
SMALL_BATCH = 4           # batch size while the renderer is only a little ahead of the audio
# After the last word (owner's test: "when she finishes her words her face moves back to the original
# face"): the mouth closes on silence, then the lip-synced lower face fades into the loop's own face
# on the same frames (same head pose), and the idle video continues from the next frame.
TAIL_CLOSE_FRAMES = 4     # 200 ms at 20 fps
TAIL_FADE_FRAMES = 8      # 400 ms
# Tail frames rendered with lip sync: every 2nd (the browser blends the frames in between); the last
# tail frame is the loop's own frame and needs no GPU. Rendering all 12 took ~800 ms, so after a short
# reply the tail came too late (the renderer only keeps pace with the audio) and the mouth snapped back.
TAIL_RENDERED = (0, 2, 4, 6, 8)
IDLE_LAG_FRAMES = 3       # the browser plays idle frames ~150 ms behind the worker (jitter buffer)
STATE_TO_LOOP = {"THINKING": "thinking", "LISTENING": "friendly", "IDLE": "friendly", "END_OF_TURN": "friendly"}  # warm idle
# Emotions that change and stay (owner, 2026-10-05: "only small & short changes in face emotion"):
# a later sentence of a reply can change the face (behavior source "clause"), blended over
# EXPRESSION_FADE_FRAMES; after a happy or concerned reply the face stays for LINGER_FRAMES before it
# relaxes into the default.
EXPRESSION_FADE_FRAMES = 8   # 400 ms
START_BIAS_MAX = 10          # frames the learned reply start may move (start_gap)
LINGER_FRAMES = 50           # 2.5 s
LINGERING = ("happy", "concerned")


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
    end_ms: float | None = None        # total audio of the reply (speech_end), then the tail follows
    tail_done: bool = False
    tail_tries: int = 0
    segment_loops: dict = field(default_factory=dict)   # seq -> expression loop of that sentence
    fade_from: str | None = None       # the previous sentence's loop while the face changes ...
    fade_left: int = 0                 # ... for this many more frames
    fade_total: int = EXPRESSION_FADE_FRAMES   # length of that blend
    breath: float = 0.0                # breath level at the end of the last sentence (speech_body.py)
    stream: H264Stream | None = None   # H.264: the reply's own video stream (None: JPEG frames)

    def audible_ms(self, at: float) -> float:
        """Playback position at time ``at`` (see catchup.audible_ms)."""
        return audible_ms(self.played_ms, self.played_at, time.perf_counter(), at)


class AvatarWorker:
    def __init__(self, avatar_id: str, batch: int, quality: int, lead_ms: int, gate_frames: int = 4,
                 warm_burst_ms: int = 0, late_ms: int = 100, decoder: str = "vae", h264_kbps: int = 1000,
                 h264_idle_kbps: int = 600, h264_encoder: str = "libx264") -> None:
        folder = AVATARS / avatar_id
        meta = json.loads((folder / "avatar.json").read_text(encoding="utf-8"))
        self.fps = meta["fps"]
        self.loops = {name: Loop.load(folder / "loops" / name) for name in meta["loops"]}
        # head gestures: variants of the expression loops that differ only inside gesture windows
        pingpong = meta.get("pingpong", True)
        n_loop = len(next(iter(self.loops.values())))
        n_forward = n_loop // 2 if pingpong else n_loop
        self.cycle = n_loop
        self.variants: dict[tuple[str, str], Loop] = {}
        self.gesture_windows: dict[str, list[tuple[int, int]]] = {}
        for gesture, gm in meta.get("gestures", {}).items():
            for expression in gm["expressions"]:
                vdir = folder / "loops" / f"{expression}__{gesture}"
                if expression in self.loops and (vdir / "delta.pkl").exists():
                    self.variants[(expression, gesture)] = self.loops[expression].with_delta(LoopDelta.load(vdir))
            self.gesture_windows[gesture] = cycle_windows([tuple(w) for w in gm["windows"]], n_forward, pingpong)
        self._gesture: tuple[str, int, int] | None = None      # (gesture, first frame, end frame), absolute
        self._reply_gesture: tuple[str, float] | None = None   # requested by the LLM for the next reply
        self.engine = MuseTalkEngine(fps=self.fps, batch_size=batch, decoder=decoder)
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
        self._pending_idle: tuple[str, int] | None = None   # (loop, from frame): relax after an emotional reply
        # Where a reply starts in the loop: the idle frame on screen when its audio starts. lead_ms and
        # IDLE_LAG_FRAMES are guesses; the page measures how far off they were (start_gap, in frames) and
        # this learns it (the v3 head moves quickly, so 5 frames off was a visible jump at every reply).
        self.start_bias = 0.0
        self.start_hold_frames = 0   # the page holds each reply's first audio this long (start_hold)
        # Video codec of the current browser (backend: {"type":"codec"}): JPEG frames, or H.264 streams -
        # one for the idle video and one per reply, taken from a spare started in advance (an ffmpeg
        # process takes ~70 ms to start; a reply must not wait for it).
        self.codec = "jpeg"
        self.h264_kbps, self.h264_idle_kbps, self.h264_encoder = h264_kbps, h264_idle_kbps, h264_encoder
        self._idle_stream: H264Stream | None = None
        self._spare: H264Stream | None = None
        self._au_queue: asyncio.Queue = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
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
        self._loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(self._idle()), asyncio.create_task(self._render()),
                 asyncio.create_task(self._send_access_units())]
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
            if self.turn is not None:
                self._end_stream(self.turn)
            self.ws, self.turn = None, None
            self._set_codec("jpeg")
            logger.info("backend disconnected")

    def _on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "behavior":
            loop = EXPRESSION_TO_LOOP.get(msg.get("expression", ""), None)
            source = str(msg.get("source", ""))
            if loop in self.loops:
                self.speech_loop = loop
                if source == "clause" and self.turn is not None:
                    pass                    # from the next sentence's audio on (_on_audio)
                elif source in ("state:LISTENING", "state:IDLE"):
                    self._relax_idle(loop)
                else:
                    self._switch_idle(loop)
            gesture, intensity = msg.get("gesture"), float(msg.get("intensity") or 0.0)
            if source == "clause":
                pass
            elif source.startswith("state"):
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
                if msg.get("state") in ("LISTENING", "IDLE"):
                    self._relax_idle(loop)
                else:
                    self._switch_idle(loop)
        elif kind == "playback" and self.turn and msg.get("turn_id") == self.turn.turn_id:
            if msg.get("event") in ("started", "progress"):
                self.turn.playing = True
                self.turn.played_ms = float(msg.get("played_ms") or 0)
                self.turn.played_at = time.perf_counter()
            elif msg.get("event") == "ended":
                # the idle video continues on the frame after the reply's last one (its closing tail)
                turn = self.turn
                self.turn = None
                self._end_stream(turn)                 # (normally already ended after its closing tail)
                if turn.rendered:
                    self.idx = turn.base_idx + turn.rendered
                    if turn.loop in self.loops and turn.loop != self.idle_loop:
                        # the reply may have changed its face sentence by sentence: idle goes on with
                        # the face it ended with (its last frame is that loop's own frame)
                        self.idle_loop, self._fade = turn.loop, None
        elif kind == "codec":
            self._set_codec(str(msg.get("codec") or "jpeg"))
            asyncio.create_task(self._send_json({"type": "codec", "codec": self.codec}))
        elif kind == "idle_keyframe" and self.codec == "h264":
            self._restart_idle_stream()              # the backend dropped idle frames: start again from a keyframe
        elif kind == "speech_end" and self.turn and msg.get("turn_id") == self.turn.turn_id:
            self.turn.end_ms = float(msg.get("total_ms") or 0)
            self.queue.put_nowait((self.turn, 0, 0, None))          # seq 0: the closing tail
        elif (kind == "cancel" and self.turn and msg.get("turn_id") == self.turn.turn_id) or kind == "reset":
            # reset: the browser session ended (tab closed while the avatar spoke) - back to idle
            if self.turn is not None:
                self.turn.cancelled = True
                self._end_stream(self.turn)
                if self.turn.playing:      # idle continues near the frame that was on screen
                    shown = self.turn.audible_ms(time.perf_counter()) * self.fps / 1000
                    self.idx = self.turn.base_idx + max(0, min(self.turn.rendered, round(shown)))
            self.turn = None
            if kind == "reset":
                self._set_codec("jpeg")            # the next browser says which codec it can show
                self.start_hold_frames = 0
            while not self.queue.empty():
                self.queue.get_nowait()
        elif kind == "start_hold":
            frames = max(0, min(20, round(float(msg.get("ms") or 0) * self.fps / 1000)))
            if frames != self.start_hold_frames:
                logger.info(f"the page holds replies {frames} frames (network jitter)")
            self.start_hold_frames = frames
        elif kind == "start_gap":
            gap = float(msg.get("gap") or 0.0)
            self.start_bias = max(-START_BIAS_MAX, min(START_BIAS_MAX, self.start_bias - 0.5 * gap))
            logger.debug(f"reply start {gap:+.1f} frames off the idle on screen: bias {self.start_bias:+.1f}")
        elif kind == "stats":
            asyncio.create_task(self._send_json({"type": "stats", **self._stats()}))

    def _on_audio(self, turn_id: int, seq: int, pts: int, pcm: np.ndarray) -> None:
        if self.turn is None or self.turn.turn_id != turn_id:
            if self.turn is not None and turn_id < self.turn.turn_id:
                return  # stale
            # the first speech frame should follow the idle frame on screen when the audio starts: the
            # audio starts ~lead_ms from now (plus the page's start hold), and the browser shows idle frames
            # IDLE_LAG_FRAMES late - so the start may be behind idx
            start = self.idx + max(-6, self.lead_frames + self.start_hold_frames - IDLE_LAG_FRAMES
                                   + round(self.start_bias))
            if self.turn is not None:
                self._end_stream(self.turn)        # a newer reply replaces it
            self.turn = Turn(turn_id, base_idx=start, loop=self.speech_loop, stream=self._take_stream())
            logger.info(f"turn {turn_id}: speaking with loop '{self.speech_loop}' from frame {self.turn.base_idx}")
            if self._fade is not None and self._fade[0] in self.loops and self._fade[0] != self.speech_loop:
                # the idle video is still fading from the thinking face into the reply's: the first
                # lip-synced frames continue that fade instead of cutting it short (with the v3 faces
                # the cut changed 3.8 % of the picture at once; e2e phase6_speech_end)
                prev, left = self._fade
                left -= self.turn.base_idx - self.idx
                if left > 0:
                    self.turn.fade_from, self.turn.fade_left, self.turn.fade_total = prev, left, 6   # idle: 1 - left / 7
            if self._reply_gesture is not None:
                gesture, intensity = self._reply_gesture
                self._reply_gesture = None
                self._schedule_gesture(gesture, intensity, self.turn.base_idx, self.turn.loop)
        self.turn.segment_loops[seq] = self.speech_loop
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

    def _keep_talking(self, turn: Turn, at: int) -> None:
        """While the avatar speaks, chain "talk" windows (motion.GESTURES) so her head moves as people's
        do when they talk; a gesture the reply asked for (nod, tilt, ...) is never replaced. Called
        before the frames from ``at`` on are rendered."""
        if (turn.loop, "talk") not in self.variants or "talk" not in self.gesture_windows:
            return
        g = self._gesture
        if g is not None and g[2] > at:                 # a gesture is scheduled or playing over these frames
            return
        start, end = next_window(at, self.gesture_windows["talk"], self.cycle)
        last = turn.base_idx + round((turn.end_ms or 0) * self.fps / 1000) + TAIL_CLOSE_FRAMES + TAIL_FADE_FRAMES
        if turn.end_ms is not None and end > last:
            return                                      # would outlast the reply and its closing tail
        self._gesture = ("talk", start, end)
        self.stats["talk_windows"] = self.stats.get("talk_windows", 0) + 1

    def _source(self, loop_name: str, i: int) -> Loop:
        """The loop that provides frame ``i``: the gesture variant inside a scheduled gesture window."""
        g = self._gesture
        if g is not None and g[1] <= i < g[2]:
            variant = self.variants.get((loop_name, g[0]))
            if variant is not None:
                return variant
        return self.loops[loop_name]

    def _switch_idle(self, loop: str) -> None:
        self._pending_idle = None
        if loop != self.idle_loop:
            self._fade = (self.idle_loop, 6)
            self.idle_loop = loop

    def _relax_idle(self, loop: str) -> None:
        """Back to a state's default face - after a happy or concerned reply only LINGER_FRAMES later,
        so the emotion does not vanish with the last word."""
        if self.idle_loop in LINGERING and loop != self.idle_loop:
            if self._pending_idle is None:
                self._pending_idle = (loop, self.idx + LINGER_FRAMES)
        else:
            self._switch_idle(loop)

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
                self._end_stream(turn)
                self.turn = None
            if self._pending_idle is not None and self.idx >= self._pending_idle[1]:
                self._switch_idle(self._pending_idle[0])
            if self.turn is None or not self.turn.playing:
                stream = self._idle_stream if self.codec == "h264" else None
                if self._fade:
                    prev, left = self._fade
                    alpha = 1 - left / 7
                    # both sides with any gesture in progress (base frames alone made the head snap)
                    blend = cv2.addWeighted(self._source(prev, self.idx).frame(self.idx), 1 - alpha,
                                            self._source(self.idle_loop, self.idx).frame(self.idx), alpha, 0)
                    self._fade = (prev, left - 1) if left > 1 else None
                    if stream is not None:
                        await asyncio.to_thread(stream.encode, blend, ("idle", self.idx, stream))
                    else:
                        jpg = cv2.imencode(".jpg", blend, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
                        await self._send(HEADER.pack(KIND_IDLE, 0, self.idx, 0) + jpg)
                else:
                    src = self._source(self.idle_loop, self.idx)
                    if stream is not None:
                        await asyncio.to_thread(self._encode_idle, stream, src, self.idx)
                    else:
                        await self._send(HEADER.pack(KIND_IDLE, 0, self.idx, 0) + src.frames_jpg[self.idx % len(src)])
            self.idx += 1

    # ------------------------------------------------------------------ speech rendering
    async def _render(self) -> None:
        while True:
            turn, seq, pts, pcm = await self.queue.get()
            if turn.cancelled:
                continue
            try:
                if pcm is None:
                    await self._render_tail(turn)
                    continue
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
        want = turn.segment_loops.pop(seq, turn.loop)
        if want != turn.loop and want in self.loops:      # this sentence has another face: blend into it
            logger.info(f"turn {turn.turn_id} seq {seq}: expression {turn.loop} -> {want}")
            turn.fade_from, turn.fade_left, turn.loop = turn.loop, EXPRESSION_FADE_FRAMES, want
            turn.fade_total = EXPRESSION_FADE_FRAMES
        loop = self.loops[turn.loop]
        n = chunks.shape[0]
        if n == 0:
            await self._release(turn, seq)
            return
        # breathing that follows the speech: in pauses the shoulders rise, while speaking they settle
        levels = breath_levels(frame_rms(pcm, spf, n), self.fps, turn.breath)
        lifts = to_lift(levels)
        turn.breath = float(levels[-1])
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
            self._keep_talking(turn, base + frames[0])
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
            pending = asyncio.create_task(self._compose_and_send(turn, seq, pts, srcs, idx, faces, frames, t0, gate,
                                                                 lifts))
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

    async def _render_tail(self, turn: Turn) -> None:
        """After the reply's last word: TAIL_CLOSE_FRAMES of lip sync on silence (the mouth closes
        naturally from its last sound), then TAIL_FADE_FRAMES in which the lip-synced lower face fades
        into the loop's own face on the same frames. The last tail frame is the loop frame itself, so
        the idle video that continues on the next frame joins it without a jump. Tail frames carry pts
        after the end of the audio; the browser keeps drawing them by the audio clock."""
        if turn.cancelled or turn.tail_done or turn.rendered == 0:
            return
        if turn.end_ms is not None and turn.next_pts < turn.end_ms - 30 and turn.tail_tries < 40:
            turn.tail_tries += 1                      # the last clause is not rendered yet: after it
            await asyncio.sleep(0.05)
            self.queue.put_nowait((turn, 0, 0, None))
            return
        turn.tail_done = True
        t0 = time.perf_counter()
        spf = 24000 // self.fps
        total = TAIL_CLOSE_FRAMES + TAIL_FADE_FRAMES
        ctx = CONTEXT_FRAMES if turn.tail is not None else 0
        silence = np.zeros(total * spf, np.int16)
        audio = np.concatenate([turn.tail, silence]) if ctx else silence
        chunks = (await asyncio.to_thread(self.engine.audio_features, audio))[ctx:ctx + total]
        n = chunks.shape[0]
        if n == 0:
            return
        base = turn.base_idx + turn.rendered
        pts = turn.next_pts if turn.next_pts >= 0 else round(turn.end_ms or 0)
        last = n - 1
        render = [k for k in TAIL_RENDERED if k < last]
        tail_lifts = to_lift(settle(turn.breath, n))           # the breath settles: 0 on the loop's own last frame
        turn.breath = 0.0
        srcs = [self._source(turn.loop, base + k) for k in render]
        idx, faces = await asyncio.to_thread(self.engine.faces, self.loops[turn.loop], chunks[render], base, srcs,
                                             [base + k for k in render])
        for j, k in enumerate(render):
            f = k - TAIL_CLOSE_FRAMES + 1                  # 1..TAIL_FADE_FRAMES inside the fade
            x = min(1.0, max(0.0, f / TAIL_FADE_FRAMES))
            alpha = 1.0 - x * x * (3 - 2 * x)               # smoothstep: eases in and out
            frame = await asyncio.to_thread(self._compose_one, srcs[j], idx[j], faces[j], alpha, float(tail_lifts[k]))
            if turn.cancelled:
                return
            await self._send_speech(turn, frame, base + k, pts + round(k * 1000 / self.fps), FLAG_TAIL)
        src = self._source(turn.loop, base + last)          # the loop's own frame: the idle video continues from it
        if turn.stream is not None:
            await self._send_speech(turn, src.frame(base + last), base + last, pts + round(last * 1000 / self.fps), FLAG_TAIL)
        else:
            await self._send(VIDEO_HEADER.pack(KIND_VIDEO, FLAG_TAIL, turn.turn_id, base + last,
                                               pts + round(last * 1000 / self.fps)) + src.frames_jpg[(base + last) % len(src)])
        self._end_stream(turn)                                # the reply's video is complete
        turn.rendered += n
        self.stats["tails"] = self.stats.get("tails", 0) + 1
        logger.debug(f"turn {turn.turn_id}: closing tail of {n} frames (frames {base}-{base + n - 1}, pts {pts}) "
                     f"in {(time.perf_counter() - t0) * 1000:.0f} ms")

    async def _compose_and_send(self, turn: Turn, seq: int, pts: int, srcs: list[Loop], idx: list[int],
                                faces: np.ndarray, frames: list[int], t0: float, gate: int,
                                lifts: np.ndarray | None = None) -> None:
        """CPU stage. Each frame goes out as soon as it is composited: waiting for the whole batch of 8
        (~80 ms) made the first frames of every batch the ones most likely to arrive late.
        ``frames``: the frames' positions in the segment (every 2nd/3rd when the GPU is too slow)."""
        for k, (i, face) in enumerate(zip(idx, faces, strict=True)):
            lift = float(lifts[frames[k]]) if lifts is not None else 0.0
            if turn.fade_left > 0 and turn.fade_from in self.loops:
                weight = 1.0 - turn.fade_left / (turn.fade_total + 1)
                turn.fade_left -= 1
                old = self._source(turn.fade_from, i)
                frame = await asyncio.to_thread(self._compose_blend, old, srcs[k], i, face, weight, lift)
            else:
                frame = await asyncio.to_thread(self._compose_one, srcs[k], i, face, 1.0, lift)
            if turn.cancelled:
                return
            frame_pts = pts + round(frames[k] * 1000 / self.fps)
            gated = seq not in turn.first_frame_sent and frames[k] + 1 >= gate
            await self._send_speech(turn, frame, turn.base_idx + turn.rendered + frames[k], frame_pts, 0,
                                    (seq, t0) if gated else None)

    def _compose_one(self, loop: Loop, i: int, face: np.ndarray, alpha: float = 1.0, lift: float = 0.0) -> np.ndarray:
        """Composite one frame; ``alpha`` < 1 fades the lip-synced face into the loop's own face; ``lift``:
        the body below the face moved up by this many px (speech_body.py)."""
        t = time.perf_counter()
        frame = self.engine.compose(loop, i, face)
        if alpha < 1.0:
            frame = cv2.addWeighted(frame, alpha, loop.frame(i), 1.0 - alpha, 0)
        if abs(lift) >= 0.05:
            frame = lift_body(frame, int(loop.coords[i][3]), lift)
        self._compose_ms = 0.8 * self._compose_ms + 0.2 * (time.perf_counter() - t) * 1000
        return frame

    def _compose_blend(self, old: Loop, new: Loop, i: int, face: np.ndarray, weight: float, lift: float = 0.0) -> np.ndarray:
        """The two expressions' frames (same head pose) blended, and the lip-synced face pasted once into
        the blend: the face changes smoothly from one sentence's expression to the next. (Pasting into
        both and blending would show the mouth twice wherever their face boxes differ.)"""
        t = time.perf_counter()
        background = cv2.addWeighted(new.frame(i), weight, old.frame(i), 1.0 - weight, 0)
        frame = self.engine.compose(new, i, face, background=background)
        if abs(lift) >= 0.05:
            frame = lift_body(frame, int(new.coords[i][3]), lift)
        self._compose_ms = 0.8 * self._compose_ms + 0.2 * (time.perf_counter() - t) * 1000
        return frame

    # ------------------------------------------------------------------ video out (JPEG or H.264)
    async def _send_speech(self, turn: Turn, frame: np.ndarray, idx: int, pts: int, flags: int,
                           gate: tuple[int, float] | None = None) -> None:
        """One lip-synced frame of ``turn``. ``gate`` = (seq, t0): the segment's gate frame - after it the
        backend may release the segment's audio (first_frame)."""
        if turn.stream is not None:
            if gate is not None:
                turn.first_frame_sent.add(gate[0])         # sent with this frame's access unit
            if await asyncio.to_thread(turn.stream.encode, frame, ("speech", turn, idx, pts, flags, gate)):
                return
            logger.warning(f"turn {turn.turn_id}: video stream ended, JPEG from here")
            turn.stream = None
            if gate is not None:
                turn.first_frame_sent.discard(gate[0])
        t = time.perf_counter()
        jpg = await asyncio.to_thread(lambda: cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes())
        self._compose_ms = 0.8 * self._compose_ms + 0.2 * (self._compose_ms + (time.perf_counter() - t) * 1000)
        await self._send(VIDEO_HEADER.pack(KIND_VIDEO, flags, turn.turn_id, idx, pts) + jpg)
        if gate is not None and gate[0] not in turn.first_frame_sent:
            turn.first_frame_sent.add(gate[0])
            await self._send_json({"type": "first_frame", "turn_id": turn.turn_id, "seq": gate[0],
                                   "ms": round((time.perf_counter() - gate[1]) * 1000, 1)})

    async def _send_access_units(self) -> None:
        """Encoded frames (from the encoders' reader threads) to the backend, in order."""
        while True:
            meta, au, key = await self._au_queue.get()
            if not meta:
                continue
            flags = FLAG_KEY if key else 0
            if meta[0] == "idle":
                if meta[2] is not self._idle_stream:
                    continue                               # the last frames of a replaced idle stream
                self.stats["h264_bytes"] = self.stats.get("h264_bytes", 0) + len(au)
                await self._send(VIDEO_HEADER.pack(KIND_IDLE_H264, flags, 0, meta[1], 0) + au)
                continue
            _, turn, idx, pts, fflags, gate = meta
            if turn.cancelled:
                continue                                   # its whole stream is dropped (never decoded)
            self.stats["h264_bytes"] = self.stats.get("h264_bytes", 0) + len(au)
            await self._send(VIDEO_HEADER.pack(KIND_VIDEO_H264, fflags | flags, turn.turn_id, idx, pts) + au)
            if gate is not None:
                await self._send_json({"type": "first_frame", "turn_id": turn.turn_id, "seq": gate[0],
                                       "ms": round((time.perf_counter() - gate[1]) * 1000, 1)})

    @staticmethod
    def _encode_idle(stream: H264Stream, src: Loop, idx: int) -> None:
        stream.encode(src.frame(idx), ("idle", idx, stream))

    def _on_access_unit(self, meta, au: bytes, key: bool) -> None:
        """Called from an encoder's reader thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._au_queue.put_nowait, (meta, au, key))

    def _new_stream(self, kbps: int) -> H264Stream | None:
        try:
            return H264Stream(self._on_access_unit, size=tuple(self.loops[self.idle_loop].size), fps=self.fps,
                              kbps=kbps, gop=2 * self.fps, encoder=self.h264_encoder)
        except Exception as exc:  # noqa: BLE001 - no ffmpeg: JPEG frames
            logger.warning(f"H.264 stream could not start ({exc!r}): JPEG frames")
            return None

    def _set_codec(self, codec: str) -> None:
        if codec == "h264" and ffmpeg_exe() is None:
            logger.warning("the browser can show H.264 but ffmpeg is missing: JPEG frames")
            codec = "jpeg"
        if codec == self.codec and (codec == "jpeg" or self._idle_stream is not None):
            return
        self.codec = codec
        for stream in (self._idle_stream, self._spare):
            self._close_later(stream)
        self._idle_stream = self._spare = None
        if codec == "h264":
            self._idle_stream = self._new_stream(self.h264_idle_kbps)
            self._spare = self._new_stream(self.h264_kbps)
            if self._idle_stream is None:
                self.codec = "jpeg"
        logger.info(f"video: {self.codec}")

    def _restart_idle_stream(self) -> None:
        if self.codec != "h264":
            return
        self._close_later(self._idle_stream)
        self._idle_stream = self._new_stream(self.h264_idle_kbps)

    def _take_stream(self) -> H264Stream | None:
        """The reply's video stream (H.264): the spare, and a new spare started in the background."""
        if self.codec != "h264":
            return None
        stream, self._spare = self._spare, None
        if stream is None or stream.closed:
            stream = self._new_stream(self.h264_kbps)
        if self._loop is not None:
            self._loop.run_in_executor(None, self._refill_spare)
        return stream

    def _refill_spare(self) -> None:
        if self.codec == "h264" and self._spare is None:
            spare = self._new_stream(self.h264_kbps)
            if spare is not None and (self.codec != "h264" or self._spare is not None):
                spare.close()                     # the codec changed meanwhile, or another spare is there
            else:
                self._spare = spare

    def _end_stream(self, turn: Turn) -> None:
        """No more frames for this reply: the encoder flushes its last frames and exits."""
        stream, turn.stream = turn.stream, None
        self._close_later(stream)

    @staticmethod
    def _close_later(stream: H264Stream | None) -> None:
        """Close without waiting here: ffmpeg takes a moment to exit, and the idle video must not stop."""
        if stream is not None:
            threading.Thread(target=stream.close, daemon=True).start()

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
    ap.add_argument("--decoder", choices=("vae", "taesd"), default="vae", help="latent decoder (taesd: ~2.5x faster lip sync)")
    ap.add_argument("--h264-kbps", type=int, default=1000, help="H.264 bitrate of the lip-synced video")
    ap.add_argument("--h264-idle-kbps", type=int, default=600, help="H.264 bitrate of the idle video")
    ap.add_argument("--h264-encoder", choices=("libx264", "nvenc"), default="libx264")
    args = ap.parse_args()
    worker = AvatarWorker(args.avatar, args.batch, args.quality, args.lead_ms, args.gate_frames, args.warm_burst_ms,
                          args.late_ms, args.decoder, args.h264_kbps, args.h264_idle_kbps, args.h264_encoder)
    async with websockets.serve(worker.handle, args.host, args.port, max_size=None, ping_interval=20):
        logger.info(f"avatar worker listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
