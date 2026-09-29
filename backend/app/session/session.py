"""One conversation session = one browser connection.

Owns the state machine, turn registry, conversation memory and outbox.
Inputs: typed text (``user_text``) or microphone PCM (binary frames → VAD → STT).
Outputs: streamed text, behaviour events, TTS audio packets (+ avatar frames in Phase 4).
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid

from fastapi import WebSocket
from loguru import logger

from app.audio.vad import StreamingVAD, VadEvent
from app.behavior.manager import for_state, from_meta
from app.behavior.meta import MetaStreamParser
from app.behavior.speech_text import clean_delta
from app.core.logging import write_metrics
from app.llm.client import LLMStats
from app.llm.prompt import ConversationMemory
from app.services import Services
from app.session.outbox import Outbox
from app.session.state_machine import Event, State, StateMachine
from app.session.turn import TurnContext, TurnRegistry
from app.transport.protocol import KIND_IDLE_VIDEO
from app.tts.chunker import ClauseChunker
from app.tts.speech_output import SpeechOutput

ACTIVE = (State.THINKING, State.SPEAKING)


class Session:
    def __init__(self, ws: WebSocket, services: Services) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.ws = ws
        self.svc = services
        self.cfg = services.settings
        self.sm = StateMachine()
        self.turns = TurnRegistry()
        self.memory = ConversationMemory(persona=self.cfg.persona_name, max_turns=self.cfg.llm.history_turns,
                                         knowledge_enabled=self.cfg.rag.enabled)
        self.outbox = Outbox(ws, self.turns.is_current)
        self.voice_out = False            # client plays assistant audio
        self.vad: StreamingVAD | None = None
        self._stt_task: asyncio.Task | None = None
        self._carry_text = ""             # transcript of an utterance the user continued
        self._spec: tuple[float, asyncio.Task] | None = None  # (speech_end_t, speculative STT task)
        self._summary_task: asyncio.Task | None = None
        self.dropped_idle_frames = 0
        self.sm.subscribe(self._on_state_change)
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.outbox.start()
        self.outbox.json({"type": "session", "session_id": self.id, "persona": self.cfg.persona_name,
                          "profile": self.cfg.profile, "voice_available": self.svc.voice_ready,
                          "avatar_available": self.svc.avatar_ready,
                          "avatar": self.svc.avatar.hello if self.svc.avatar_ready else None,
                          "server_time": time.time()}, force=True)
        self.sm.fire(Event.START)

    async def close(self, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        if self.turns.current:
            self.turns.current.cancel()
        for task in (self._stt_task, self._summary_task):
            if task and not task.done():
                task.cancel()
        self.sm.try_fire(Event.STOP)
        await self.outbox.stop()
        logger.info(f"session {self.id} closed ({reason})")

    # ------------------------------------------------------------------ inbound
    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.voice_out = bool(msg.get("voice_out", False)) and self.svc.voice_ready
        elif kind == "voice":
            await self._set_voice(bool(msg.get("enabled")))
        elif kind == "user_text":
            text = str(msg.get("text", "")).strip()
            if text:
                await self.start_turn("text", text)
        elif kind == "interrupt":
            await self.interrupt(msg.get("reason", "manual"))
        elif kind == "playback":
            self._on_playback(msg)
        elif kind == "ping":
            self.outbox.json({"type": "pong", "t": msg.get("t"), "server_time": time.time()}, force=True)
        else:
            logger.debug(f"unknown message type {kind!r}")

    async def on_binary(self, data: bytes) -> None:
        if self.vad is None:
            return
        active = self.sm.state in ACTIVE
        for ev in self.vad.process(data, assistant_active=active):
            if ev.kind == "start":
                await self._on_speech_start(ev)
            elif ev.kind == "pause":
                self._on_speech_pause(ev)
            else:
                await self._on_speech_end(ev)

    async def _set_voice(self, enabled: bool) -> None:
        if enabled and not self.svc.voice_ready:
            self.outbox.json({"type": "error", "message": "voice models are not loaded"}, force=True)
            return
        self.voice_out = enabled
        if enabled and self.vad is None:
            self.vad = await asyncio.to_thread(StreamingVAD, self.cfg.vad)
        elif not enabled:
            self.vad = None
        self.outbox.json({"type": "voice", "enabled": enabled}, force=True)

    # ------------------------------------------------------------------ voice input
    async def _on_speech_start(self, ev: VadEvent) -> None:
        self.outbox.json({"type": "vad", "event": "start", "bargein": ev.bargein}, force=True)
        if self.sm.state in ACTIVE:
            await self.interrupt("barge_in")
        elif self.sm.state is State.END_OF_TURN:
            # The user kept talking before STT finished: keep that text and wait for the rest.
            self.sm.fire(Event.SPEECH_RESUMED)

    def _on_speech_pause(self, ev: VadEvent) -> None:
        """Start STT speculatively; reused at the endpoint if the user stayed silent."""
        if self.sm.state is not State.LISTENING or not len(ev.audio):
            return
        if self._spec and not self._spec[1].done():
            self._spec[1].cancel()
        self._spec = (ev.speech_end_t, asyncio.create_task(self._stt(ev.audio), name="stt-spec"))

    async def _stt(self, audio) -> tuple[str, float]:
        t_req = time.perf_counter()
        async with self.svc.gpu_lock:
            t_lock = time.perf_counter()
            text = await asyncio.to_thread(self.svc.stt.transcribe, audio)
        t_end = time.perf_counter()
        self._stt_timing = {"lock_wait_ms": round((t_lock - t_req) * 1000, 1),
                            "thread_ms": round((t_end - t_lock) * 1000, 1), **self.svc.stt.last_timing}
        return text, time.time()

    async def _on_speech_end(self, ev: VadEvent) -> None:
        self.outbox.json({"type": "vad", "event": "end", **ev.stats}, force=True)
        if self.sm.state is not State.LISTENING:
            return
        self.sm.fire(Event.SPEECH_END)
        self._stt_task = asyncio.create_task(self._transcribe(ev), name="stt")

    async def _transcribe(self, ev: VadEvent) -> None:
        t_end = ev.speech_end_t or ev.t
        t_endpoint = time.time()
        spec, self._spec = self._spec, None
        speculative = spec is not None and spec[0] == ev.speech_end_t   # no new speech since the pause
        if speculative:
            text, t_stt = await spec[1]
        else:
            if spec and not spec[1].done():
                spec[1].cancel()
            text, t_stt = await self._stt(ev.audio)
        if self.sm.state is not State.END_OF_TURN:      # user resumed speaking meanwhile
            self._carry_text = f"{self._carry_text} {text}".strip()
            return
        text = f"{self._carry_text} {text}".strip()
        self._carry_text = ""
        if text and self._looks_like_echo(text):
            logger.warning(f"ignored probable echo of the assistant: {text!r}")
            self.outbox.json({"type": "echo_ignored", "text": text}, force=True)
            text = ""
        if not text:
            self.sm.fire(Event.TRANSCRIPT_EMPTY)
            self.outbox.json({"type": "transcript", "text": "", "final": True}, force=True)
            return
        turn = await self.start_turn("voice", text, t0=t_end)
        turn.mark("vad_endpoint", t_endpoint)
        turn.mark("stt_done", t_stt)
        turn.utterance_ms = ev.stats.get("duration_ms")
        turn.stt_speculative = speculative
        turn.stt_timing = getattr(self, "_stt_timing", None)

    def _looks_like_echo(self, text: str) -> bool:
        """True if the transcript mostly repeats the assistant's last words (speaker → mic leak)."""
        last = self.memory.turns[-1]["content"] if self.memory.turns and self.memory.turns[-1]["role"] == "assistant" else ""
        words = [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2]
        said = set(re.findall(r"[a-z']+", last.lower()))
        if len(words) < 3 or not said:
            return False
        return sum(w in said for w in words) / len(words) >= 0.8

    # ------------------------------------------------------------------ turns
    async def start_turn(self, source: str, text: str, t0: float | None = None) -> TurnContext:
        if self.sm.state in ACTIVE:
            await self.interrupt("new_input")
        turn = self.turns.new_turn(source, text)
        turn.mark("t0", t0)
        self.sm.fire(Event.TEXT_INPUT if source == "text" else Event.TRANSCRIPT_READY)
        self.outbox.json({"type": "user_message", "turn_id": turn.turn_id, "text": text, "source": source},
                         turn.turn_id, force=True)
        task = asyncio.create_task(self._run_turn(turn), name=f"turn-{turn.turn_id}")
        turn.tasks.append(task)
        return turn

    async def _run_turn(self, turn: TurnContext) -> None:
        try:
            await self._respond(turn)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"turn {turn.turn_id} failed")
            self.outbox.json({"type": "error", "turn_id": turn.turn_id, "message": str(exc)}, force=True)
            if self.turns.is_current(turn.turn_id):
                turn.cancel()
                self.sm.fire(Event.FAILURE)
                self.sm.fire(Event.RECOVER)

    async def _respond(self, turn: TurnContext) -> None:
        tid = turn.turn_id
        knowledge, citations = await self._retrieve(turn)
        messages = self.memory.build(turn.user_text, knowledge)
        parser, spoken, stats = MetaStreamParser(), [], LLMStats()
        speech = chunker = None
        if self.voice_out:
            speech = SpeechOutput(turn, self.svc.tts, self.outbox, self.svc.gpu_lock,
                                  on_first_audio=lambda: self._first_output(turn),
                                  avatar_sink=self.svc.avatar_sink_for(self, turn) if self.svc.avatar_ready else None)
            turn.speech = speech
            chunker = ClauseChunker()
        behavior_sent = False
        turn.mark("llm_request")
        async for delta in self.svc.llm.stream_chat(messages, stats):
            if turn.is_cancelled:
                return
            turn.mark("llm_first_token")
            _meta, text = parser.feed(delta)
            text = clean_delta(text)
            if not behavior_sent and (parser.meta is not None or text):
                self._send_behavior(turn, from_meta(parser.meta))
                behavior_sent = True
            if text:
                self._emit_text(turn, text, spoken)
                if chunker:
                    for clause in chunker.feed(text):
                        speech.say(clause)
        tail = clean_delta(parser.flush())
        if tail:
            self._emit_text(turn, tail, spoken)
            if chunker:
                for clause in chunker.feed(tail):
                    speech.say(clause)
        if chunker:
            for clause in chunker.flush():
                speech.say(clause)
            speech.close()
        turn.mark("llm_done")
        reply = "".join(spoken).strip()
        turn.header = parser.raw_header or ""
        if citations:
            self.outbox.json({"type": "citations", "turn_id": tid, "sources": citations}, tid)

        if speech is not None:
            await speech.finished.wait()
            if turn.is_cancelled:
                return
            if speech.total_ms > 0:
                # wait for the browser to finish playing (bounded)
                timeout = speech.total_ms / 1000 + 8.0
                try:
                    await asyncio.wait_for(turn.playback_ended.wait(), timeout=timeout)
                except TimeoutError:
                    logger.warning(f"turn {tid}: no playback-ended ack after {timeout:.0f}s")
        if not self.turns.is_current(tid):
            return
        self.memory.add(turn.user_text, turn.header + reply)
        if self.sm.state in ACTIVE:
            self.sm.fire(Event.RESPONSE_DONE)
        self.outbox.json({"type": "assistant_done", "turn_id": tid, "text": reply, "interrupted": False},
                         tid, force=True)
        self._emit_metrics(turn, stats, interrupted=False)
        self._maybe_summarize()

    async def _retrieve(self, turn: TurnContext) -> tuple[str | None, list[dict]]:
        if self.svc.rag is None:
            return None, []
        result = await asyncio.to_thread(self.svc.rag.retrieve, turn.user_text)
        turn.mark("rag_done")
        turn.rag_stats = result.stats
        return result.context, result.citations

    def _emit_text(self, turn: TurnContext, text: str, spoken: list[str]) -> None:
        spoken.append(text)
        turn.spoken_text = "".join(spoken)
        turn.mark("llm_first_text")
        self.outbox.json({"type": "llm_delta", "turn_id": turn.turn_id, "text": text}, turn.turn_id)
        if not self.voice_out:
            self._first_output(turn)

    def _first_output(self, turn: TurnContext) -> None:
        if self.turns.is_current(turn.turn_id) and self.sm.state is State.THINKING:
            turn.mark("first_output")
            self.sm.fire(Event.FIRST_OUTPUT)

    def _on_playback(self, msg: dict) -> None:
        turn = self.turns.current
        if turn is None or msg.get("turn_id") != turn.turn_id:
            return
        event = msg.get("event")
        turn.played_ms = float(msg.get("played_ms", turn.played_ms))
        if event == "started":
            turn.mark("client_first_audio")
            self.svc.avatar_playback(turn.turn_id, "started")
        elif event == "ended":
            turn.mark("client_playback_ended")
            turn.playback_ended.set()
            self.svc.avatar_playback(turn.turn_id, "ended")

    def on_avatar_frame(self, packet: bytes, kind: int, turn_id: int) -> None:
        """Avatar worker frame → browser. Idle frames are dropped if the link is backed up."""
        if kind == KIND_IDLE_VIDEO:
            if self.outbox.backlog > 6:
                self.dropped_idle_frames += 1
                return
            self.outbox.binary(packet, 0, force=True)
        else:
            self.outbox.binary(packet, turn_id)

    # ------------------------------------------------------------------ interruption
    async def interrupt(self, reason: str) -> None:
        turn = self.turns.current
        if turn is None or self.sm.state not in ACTIVE:
            return
        t_start = time.perf_counter()
        turn.mark("interrupted")
        self.sm.fire(Event.BARGE_IN)
        turn.cancel()
        self.outbox.json({"type": "interrupt", "turn_id": turn.turn_id, "reason": reason}, force=True)
        if self.svc.avatar_ready:
            await self.svc.avatar_cancel(turn.turn_id)
        heard = self._heard_text(turn)
        body = f"{heard} [interrupted by the user]" if heard else "[interrupted before replying]"
        self.memory.add(turn.user_text, (turn.header or "") + body)
        self.sm.fire(Event.CANCEL_DONE)
        cancel_ms = (time.perf_counter() - t_start) * 1000
        self.outbox.json({"type": "assistant_done", "turn_id": turn.turn_id, "text": heard, "interrupted": True,
                          "reason": reason}, force=True)
        logger.info(f"turn {turn.turn_id} interrupted ({reason}) in {cancel_ms:.1f} ms; heard={heard[:60]!r}")
        turn.cancel_ms = round(cancel_ms, 2)
        self._emit_metrics(turn, None, interrupted=True)

    def _heard_text(self, turn: TurnContext) -> str:
        """What the user actually received: displayed text (text mode) or played audio (voice)."""
        if turn.speech is not None:
            return turn.speech.heard_text(turn.played_ms)
        return turn.spoken_text.strip()

    # ------------------------------------------------------------------ memory
    def _maybe_summarize(self) -> None:
        if not self.memory.needs_summary() or (self._summary_task and not self._summary_task.done()):
            return

        async def run() -> None:
            await asyncio.sleep(0.5)
            if self.sm.state is not State.LISTENING:
                return
            try:
                text = await self.svc.llm.complete(self.memory.summary_request(), max_tokens=120)
                self.memory.apply_summary(text)
                logger.info(f"rolling summary updated ({len(text)} chars)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"summary failed: {exc!r}")

        self._summary_task = asyncio.create_task(run(), name="summary")

    # ------------------------------------------------------------------ events
    def _on_state_change(self, old: State, new: State, event: Event) -> None:
        self.outbox.json({"type": "state", "state": new.value, "prev": old.value, "event": event.value,
                          "turn_id": self.turns.current_id}, force=True)
        if new in (State.LISTENING, State.THINKING):
            self._send_behavior(None, for_state(new.value))
        if self.svc.avatar_ready:
            self.svc.avatar_state(new.value)

    def _send_behavior(self, turn: TurnContext | None, behavior) -> None:
        self.outbox.json({"type": "behavior", "turn_id": turn.turn_id if turn else 0, **behavior.to_dict()},
                         force=turn is None)
        if self.svc.avatar_ready:
            self.svc.avatar_behavior(behavior.to_dict())

    def _emit_metrics(self, turn: TurnContext, stats: LLMStats | None, interrupted: bool) -> None:
        t0 = turn.timings.get("t0", turn.started_at)
        rel = {k: round((v - t0) * 1000, 1) for k, v in turn.timings.items()}
        record = {"type": "metrics", "session": self.id, "turn_id": turn.turn_id, "source": turn.source,
                  "interrupted": interrupted, "ms_from_t0": rel, "llm": stats.__dict__ if stats else None,
                  "utterance_ms": turn.utterance_ms, "cancel_ms": turn.cancel_ms,
                  "stt_speculative": turn.stt_speculative, "stt_timing": turn.stt_timing,
                  "speech_ms": round(turn.speech.total_ms) if turn.speech else None,
                  "tts_first_timing": turn.speech.first_timing if turn.speech else None,
                  "rag": turn.rag_stats, "avatar_first_ms": turn.avatar_first_ms, "ts": time.time()}
        self.outbox.json(record, force=True)
        write_metrics(record)
