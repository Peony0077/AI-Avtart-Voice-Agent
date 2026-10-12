"""Per-turn speech output: clauses → TTS → audio packets (+ avatar, when enabled).

Keeps a timeline of what was synthesised so the session can work out which
words the user actually heard when a turn is interrupted.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from app.behavior.speech_text import normalize_for_speech
from app.core.gpu import locked_thread
from app.transport.protocol import KIND_AUDIO, KIND_AUDIO_OPUS, pack
from app.tts.base import pause_after, trim_silence
from app.tts.opus import opus_payload

AHEAD_MS = 2500.0   # with the avatar: synthesise the next sentence when less audio than this is still queued


@dataclass
class Segment:
    seq: int
    text: str
    start_ms: float
    duration_ms: float
    words: list[tuple[str, float, float]] = field(default_factory=list)


class SpeechOutput:
    def __init__(self, turn, tts, outbox, gpu_lock: asyncio.Lock,
                 on_first_audio: Callable[[], None],
                 avatar_sink: Callable[[int, int, np.ndarray, int], Awaitable[None]] | None = None,
                 audio_codec: str = "pcm") -> None:
        self.turn = turn
        self.audio_codec = audio_codec             # what the page decodes: "opus" or "pcm" (opus.py)
        self.tts = tts
        self.outbox = outbox
        self.gpu_lock = gpu_lock
        self.on_first_audio = on_first_audio
        self.avatar_sink = avatar_sink
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.pending_chars = 0                     # text queued, not yet synthesised
        self.segments: list[Segment] = []
        self.total_ms = 0.0
        self.first_timing: dict | None = None
        self.finished = asyncio.Event()
        self.error: Exception | None = None        # set when speech output failed (finished is set too)
        self.task = asyncio.create_task(self._run(), name=f"speech-{turn.turn_id}")
        turn.tasks.append(self.task)

    def say(self, clause: str) -> None:
        if clause.strip():
            self.turn.mark("first_clause")
            self.pending_chars += len(clause)
            self.queue.put_nowait(clause)

    def backlog_ms(self) -> float:
        """Speech still to come: synthesised but not yet heard, plus queued text (~65 ms a character)."""
        return max(0.0, self.total_ms - self.turn.played_ms) + 65.0 * self.pending_chars

    def close(self) -> None:
        self.queue.put_nowait(None)

    async def _run(self) -> None:
        # ``finished`` is always set: a TTS error (e.g. CUDA out of memory, a crashed worker) used to leave the
        # turn waiting for it forever, in THINKING/SPEAKING, with no message to the page
        try:
            await self._speak()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported by the session after finished
            self.error = exc
            logger.exception(f"turn {self.turn.turn_id}: speech output failed")
        finally:
            self.finished.set()

    async def _speak(self) -> None:
        tid = self.turn.turn_id
        seq = 0
        while True:
            clause = await self.queue.get()
            if clause is None or self.turn.is_cancelled:
                break
            self.pending_chars = max(0, self.pending_chars - len(clause))
            if seq and self.avatar_sink is not None:
                await self._pace()
                if self.turn.is_cancelled:
                    break
            if hasattr(self.tts, "stream"):
                seq = await self._stream_clause(clause, seq)
                continue
            spoken = normalize_for_speech(clause)
            if not any(ch.isalnum() for ch in spoken):
                continue                                  # only a list number ("2."): nothing to say
            t_req = time.perf_counter()
            if self.turn.is_cancelled:
                break
            result, t_lock = await locked_thread(self.gpu_lock, self._timed_synth, spoken)
            if result is None:
                break
            # the engine's ~0.3 s + ~0.5 s padding made ~0.8 s gaps between sentences: natural pause instead
            result = trim_silence(result, pause_after(clause))
            if seq == 0:
                self.first_timing = {"lock_wait_ms": round((t_lock - t_req) * 1000, 1),
                                     "thread_ms": round((time.perf_counter() - t_lock) * 1000, 1),
                                     "synth_ms": result.synth_ms, "chars": len(clause)}
            if self.turn.is_cancelled:
                break
            if not len(result.audio):
                continue
            seq += 1
            self.turn.mark("tts_first_audio")
            segment = Segment(seq, clause, self.total_ms, result.duration_ms, result.words)
            self.segments.append(segment)
            pcm = (np.clip(result.audio, -1, 1) * 32767).astype(np.int16)
            encoded = self._encode(pcm)              # while the avatar renders the clause's first frame
            if self.avatar_sink is not None:
                await self.avatar_sink(tid, seq, pcm, int(self.total_ms))
            self.outbox.json({"type": "tts_segment", "turn_id": tid, "seq": seq, "text": clause,
                              "start_ms": round(segment.start_ms), "duration_ms": round(segment.duration_ms),
                              "synth_ms": result.synth_ms,
                              "words": [[w, round(s * 1000), round(e * 1000)] for w, s, e in result.words]}, tid)
            await self._send_audio(tid, seq, pcm, encoded)
            if seq == 1:
                self.turn.mark("audio_sent")
                self.on_first_audio()
            self.total_ms += result.duration_ms
        if not self.turn.is_cancelled:
            self.outbox.json({"type": "tts_done", "turn_id": tid, "total_ms": round(self.total_ms),
                              "segments": seq}, tid)

    def _timed_synth(self, text: str):
        """(result or None when the turn was cancelled while waiting for the GPU, the time the lock was
        taken) - runs in the worker thread, under the GPU lock."""
        t_lock = time.perf_counter()
        return (None if self.turn.is_cancelled else self.tts.synthesize(text)), t_lock

    async def _pace(self) -> None:
        """With the avatar: synthesise the next sentence just in time (AHEAD_MS of audio queued ahead
        of what the listener hears). Synthesising all sentences at once competed with the lip-sync
        renderer for the GPU at the start of a reply (MuseTalk 45 -> 106-177 ms per frame, ~10 % of the
        frames skipped); the earlier per-sentence avatar wait avoided that but made gaps in the speech."""
        last, since = self.turn.played_ms, time.perf_counter()
        while not self.turn.is_cancelled and self.total_ms - self.turn.played_ms > AHEAD_MS:
            if self.turn.played_ms != last:
                last, since = self.turn.played_ms, time.perf_counter()
            elif time.perf_counter() - since > 3.0:      # no playback reports: do not hold the speech back
                return
            await asyncio.sleep(0.05)

    def _encode(self, pcm: np.ndarray) -> asyncio.Future | None:
        """Opus for the page, started at once (in a thread: ~4 ms per second of speech)."""
        if self.audio_codec != "opus":
            return None
        return asyncio.ensure_future(asyncio.to_thread(opus_payload, pcm))

    async def _send_audio(self, tid: int, seq: int, pcm: np.ndarray, encoded: asyncio.Future | None) -> None:
        pts = int(self.total_ms)
        if encoded is not None:
            try:
                self.outbox.binary(pack(KIND_AUDIO_OPUS, tid, seq, pts, await encoded), tid)
                return
            except Exception as exc:  # noqa: BLE001 - e.g. libsndfile without Opus: PCM from here on
                logger.warning(f"Opus encoding failed ({exc!r}): PCM audio")
                self.audio_codec = "pcm"
        self.outbox.binary(pack(KIND_AUDIO, tid, seq, pts, pcm.tobytes()), tid)

    async def _emit(self, seq: int, pcm: np.ndarray) -> None:
        tid = self.turn.turn_id
        encoded = self._encode(pcm)
        if self.avatar_sink is not None:
            await self.avatar_sink(tid, seq, pcm, int(self.total_ms))
        await self._send_audio(tid, seq, pcm, encoded)
        if seq == 1:
            self.turn.mark("audio_sent")
            self.on_first_audio()

    async def _stream_clause(self, clause: str, seq: int) -> int:
        """Streaming engines (clone voice): send each chunk as soon as it exists.

        With the avatar, sending a chunk waits for its first video frame; chunks that arrive in
        the meantime are merged into one segment (fewer, longer segments lip-sync better).
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in self.tts.stream(normalize_for_speech(clause)):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        t0 = time.perf_counter()
        async with self.gpu_lock:
            producer = loop.run_in_executor(None, produce)
            segment = Segment(seq + 1, clause, self.total_ms, 0.0, [])
            first, done = True, False
            while not done and (chunk := await queue.get()) is not None:
                if self.avatar_sink is not None and not queue.empty():
                    parts = [chunk]
                    while not queue.empty():
                        if (nxt := queue.get_nowait()) is None:
                            done = True
                            break
                        parts.append(nxt)
                    chunk = np.concatenate(parts)
                if self.turn.is_cancelled:
                    continue          # drain the worker's stream, send nothing
                seq += 1
                if first:
                    self.turn.mark("tts_first_audio")
                    self.segments.append(segment)
                    if self.first_timing is None:
                        self.first_timing = {"first_chunk_ms": round((time.perf_counter() - t0) * 1000, 1),
                                             "chars": len(clause), "streaming": True}
                    self.outbox.json({"type": "tts_segment", "turn_id": self.turn.turn_id, "seq": seq,
                                      "text": clause, "start_ms": round(self.total_ms), "streaming": True},
                                     self.turn.turn_id)
                    first = False
                pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
                await self._emit(seq, pcm)
                dur = len(pcm) / self.tts.sample_rate * 1000
                segment.duration_ms += dur
                self.total_ms += dur
            await producer
        return seq

    def heard_text(self, played_ms: float) -> str:
        """Words whose audio finished before ``played_ms`` on this turn's timeline."""
        out: list[str] = []
        for seg in self.segments:
            end = seg.start_ms + seg.duration_ms
            if played_ms >= end - 30:
                out.append(seg.text)
                continue
            if played_ms <= seg.start_ms:
                break
            local = played_ms - seg.start_ms
            if seg.words:
                spoken = [w for w, _s, e in seg.words if e * 1000 <= local + 30]
                if spoken:
                    out.append(" ".join(spoken))
            else:  # proportional fallback (clone engines without timestamps)
                n = int(len(seg.text) * local / max(1.0, seg.duration_ms))
                cut = seg.text.rfind(" ", 0, n)
                if cut > 0:
                    out.append(seg.text[:cut])
            break
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(out)).strip()
