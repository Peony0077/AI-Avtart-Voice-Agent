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

import numpy as np
import soundfile as sf
from fastapi import WebSocket
from loguru import logger

from app.audio.mic_stats import MicStats
from app.audio.vad import StreamingVAD, VadEvent
from app.behavior.manager import (
    behavior_from_text,
    for_state,
    from_meta,
    refine_behavior,
    smile_for_happy_user,
    smile_for_user_text,
)
from app.behavior.meta import MetaStreamParser
from app.behavior.speech_text import clean_delta
from app.core.logging import write_metrics
from app.llm.client import LLMStats
from app.llm.prompt import ConversationMemory
from app.services import Services
from app.session.endpointing import looks_unfinished
from app.session.outbox import Outbox
from app.session.prefetch import AnswerPrefetch
from app.session.state_machine import Event, State, StateMachine
from app.session.turn import TurnContext, TurnRegistry
from app.transport.protocol import HEADER, KIND_IDLE_VIDEO
from app.tts.chunker import ClauseChunker
from app.tts.speech_output import SpeechOutput

ACTIVE = (State.THINKING, State.SPEAKING)


def _lateness_stats(values: list[float], backlog: int) -> dict | None:
    """Speech frames forwarded after the browser was already playing: how late were they here?"""
    if not values:
        return None
    v = sorted(values)
    return {"frames": len(v), "late_100ms": sum(x > 100 for x in v), "p50_ms": round(v[len(v) // 2]),
            "p90_ms": round(v[int(len(v) * 0.9)]), "max_ms": round(v[-1]), "max_outbox_backlog": backlog}


GREETING_TEXT = "(The user started the conversation.)"


async def _scripted(text: str):
    """A fixed reply as LLM-like deltas (word by word), so it goes through the same speech path."""
    for piece in re.findall(r"\S+\s*", text):
        yield piece


class Session:
    def __init__(self, ws: WebSocket, services: Services) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.ws = ws
        self.svc = services
        self.cfg = services.settings
        self.sm = StateMachine()
        self.turns = TurnRegistry()
        persona = services.persona
        self.memory = ConversationMemory(persona=persona.name, max_turns=self.cfg.llm.history_turns,
                                         knowledge_enabled=self.cfg.rag.enabled, behavior_tag=self.cfg.llm.behavior_tag,
                                         personality=persona.personality,
                                         long_term=services.memory_store.load() if persona.memory else None)
        self._memory_gen = services.memory_store.generation   # erased later (any tab): nothing from before is kept
        self._greeted = False
        self.outbox = Outbox(ws, self.turns.is_current)
        self.voice_out = False            # replies are spoken (client plays assistant audio)
        self.speaker_on = False           # client asked for spoken replies (also for typed messages)
        self.mic_on = False               # client streams the microphone
        self.vad: StreamingVAD | None = None
        self.mic_stats = MicStats()
        self._stt_task: asyncio.Task | None = None
        self._carry_text = ""             # transcript of an utterance the user continued
        self._spec: tuple[float, asyncio.Task] | None = None  # (speech_end_t, speculative STT task)
        self._prefetch: AnswerPrefetch | None = None           # answer started at the same pause
        self._summary_task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        self.dropped_idle_frames = 0
        self.sm.subscribe(self._on_state_change)
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.outbox.start()
        self.outbox.json({"type": "session", "session_id": self.id, "persona": self.memory.persona,
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
        self.svc.avatar_reset()
        self._drop_prefetch()
        for task in (self._stt_task, self._summary_task, self._warm_task):
            if task and not task.done():
                task.cancel()
        self.sm.try_fire(Event.STOP)
        await self.outbox.stop()
        if self.svc.persona.memory and (said := self.memory.user_said()):
            self.svc.remember(said, self._memory_gen)       # what she remembers next time (persona.py)
        if (mic := self.mic_stats.summary()) is not None:
            logger.info(f"session {self.id} microphone audio: {mic}")
        logger.info(f"session {self.id} closed ({reason})")

    # ------------------------------------------------------------------ inbound
    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.speaker_on = bool(msg.get("voice_out", False))
            self._update_voice_out()
        elif kind == "speaker":
            self.speaker_on = bool(msg.get("enabled"))
            self._update_voice_out()
            self.outbox.json({"type": "speaker", "enabled": self.voice_out}, force=True)
        elif kind == "voice":
            if msg.get("mic"):
                mic = msg["mic"]
                logger.info(f"session {self.id} microphone: {mic}")
                if mic.get("contextRate") not in (None, 16000):
                    logger.warning(f"browser runs the microphone at {mic['contextRate']} Hz: resampled to 16 kHz in the page")
                if mic.get("echoCancellation") is False:
                    logger.warning("browser echo cancellation is OFF: the avatar may hear itself (use headphones)")
            await self._set_voice(bool(msg.get("enabled")))
        elif kind == "user_text":
            text = str(msg.get("text", "")).strip()
            if text:
                await self.start_turn("text", text)
        elif kind == "interrupt":
            await self.interrupt(msg.get("reason", "manual"))
        elif kind == "playback":
            self._on_playback(msg)
        elif kind == "greet":
            await self.greet()
        elif kind == "get_persona":
            self._send_persona()
        elif kind == "set_persona":
            persona = self.svc.set_persona(msg.get("persona") or {})
            self.memory.persona, self.memory.personality = persona.name, persona.personality
            self.memory.long_term = self.svc.memory_store.load() if persona.memory else None
            self._send_persona()
        elif kind == "forget_memory":
            self.svc.memory_store.clear()
            self.memory.long_term = None
            self.memory.forget_user()
            self._memory_gen = self.svc.memory_store.generation
            logger.info("memory of the user erased (Forget me)")
            self._send_persona()
        elif kind == "client_error":
            # a problem the page showed the user (the microphone did not start, ...)
            logger.warning(f"session {self.id} browser problem ({str(msg.get('where'))[:40]}): "
                           f"{str(msg.get('name'))[:60]}: {str(msg.get('message'))[:300]} | {str(msg.get('userAgent'))[:300]}")
        elif kind == "ping":
            self.outbox.json({"type": "pong", "t": msg.get("t"), "server_time": time.time()}, force=True)
        else:
            logger.debug(f"unknown message type {kind!r}")

    async def on_binary(self, data: bytes) -> None:
        if (warning := self.mic_stats.add(data, voice_on=self.vad is not None)) is not None:
            logger.warning(f"session {self.id} {warning}")
        if self.vad is None:
            return
        active = self.sm.state in ACTIVE
        for ev in self.vad.process(data, assistant_active=active):
            if ev.kind == "start":
                self.mic_stats.speech += 1
                await self._on_speech_start(ev)
            elif ev.kind == "pause":
                self._on_speech_pause(ev)
            elif ev.kind == "resume":
                self._drop_prefetch()          # the user kept talking: that answer is stale
            else:
                await self._on_speech_end(ev)

    async def _set_voice(self, enabled: bool) -> None:
        if enabled and not self.svc.voice_ready:
            self.outbox.json({"type": "error", "message": "voice models are not loaded"}, force=True)
            return
        self.mic_on = enabled
        self._update_voice_out()
        if enabled and self.vad is None:
            self.vad = await asyncio.to_thread(StreamingVAD, self.cfg.vad)
        elif not enabled:
            self.vad = None
        self.outbox.json({"type": "voice", "enabled": enabled}, force=True)

    def _update_voice_out(self) -> None:
        # Typed messages are spoken too when the client has a speaker on: the avatar's lip sync is
        # driven by the speech audio, so a text-only reply would leave the avatar silent.
        self.voice_out = (self.speaker_on or self.mic_on) and self.svc.voice_ready

    def _send_persona(self) -> None:
        p = self.svc.persona
        self.outbox.json({"type": "persona", **p.to_dict(), "voices": self.svc.voices(),
                          "remembered": self.svc.memory_store.load() if p.memory else None}, force=True)

    async def greet(self) -> None:
        """Say hello when the user starts talking to her (the page's Start voice), once per session and
        only before anything else was said: the configured greeting, or - when she remembers the user
        from earlier sessions - a short welcome back from the LLM."""
        if self._greeted or self.turns.current is not None or not self.voice_out:
            return
        self._greeted = True
        if self.memory.long_term:
            messages = [{"role": "system", "content": self.memory.build("")[0]["content"]},
                        {"role": "user", "content": "(The user has just come back and opened the conversation. "
                                                    "Greet them warmly in one short sentence, by name if you know "
                                                    "it, and ask how you can help.)"}]
            await self.start_turn("greeting", GREETING_TEXT, messages=messages)
        else:
            await self.start_turn("greeting", GREETING_TEXT, script=self.svc.persona.greeting_text())

    # ------------------------------------------------------------------ voice input
    async def _on_speech_start(self, ev: VadEvent) -> None:
        self.outbox.json({"type": "vad", "event": "start", "bargein": ev.bargein}, force=True)
        if self.sm.state in ACTIVE:
            await self.interrupt("barge_in")
        elif self.sm.state is State.END_OF_TURN:
            # The user kept talking before STT finished: keep that text and wait for the rest.
            self.sm.fire(Event.SPEECH_RESUMED)
        if self.cfg.stt.keep_warm_ms and (self._warm_task is None or self._warm_task.done()):
            self._warm_task = asyncio.create_task(self._keep_gpu_warm(), name="gpu-warm")

    async def _keep_gpu_warm(self) -> None:
        """While the user talks, nothing uses the GPU and its clocks drop, which made the STT at the
        end of the utterance ~2x slower. An encoder pass every keep_warm_ms keeps it clocked up. Runs
        only while speech is active (never during the silence before the speculative STT) and never
        while the assistant or another GPU job is running."""
        period = self.cfg.stt.keep_warm_ms / 1000
        while self.vad is not None and self.vad.in_speech:
            await asyncio.sleep(period)
            vad = self.vad
            if (vad is None or not vad.in_speech or vad.silence_ms > 0 or self.sm.state is not State.LISTENING
                    or self.svc.gpu_lock.locked() or self.svc.stt is None):
                continue
            async with self.svc.gpu_lock:
                await asyncio.to_thread(self.svc.stt.keep_warm)

    def _on_speech_pause(self, ev: VadEvent) -> None:
        """Start STT (and the answer) speculatively; reused at the endpoint if the user stayed silent."""
        if self.sm.state is not State.LISTENING or not len(ev.audio):
            return
        if self._spec and not self._spec[1].done():
            self._spec[1].cancel()
        self._drop_prefetch()
        stt = asyncio.create_task(self._stt(ev.audio), name="stt-spec")
        self._spec = (ev.speech_end_t, stt)
        if self.cfg.llm.prefetch_on_pause:
            self._prefetch = AnswerPrefetch(ev.speech_end_t, stt, self._prefetch_text, self._retrieve_text,
                                            self.memory.build, self.svc.llm, pause_after=self._pause_after())

    def _prefetch_text(self, stt_text: str) -> str | None:
        """The user text the answer would be for (same rules as at the endpoint), or None."""
        text = f"{self._carry_text} {stt_text}".strip()
        return text if text and not self._looks_like_echo(text) else None

    def _drop_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.cancel()
            self._prefetch = None

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
        prefetch, self._prefetch = self._prefetch, None
        speculative = spec is not None and spec[0] == ev.speech_end_t   # no new speech since the pause
        adopted = False
        try:
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
            if text and self.cfg.vad.continue_ms and looks_unfinished(text):
                # probably a pause in the middle of the sentence: wait a little longer for the rest
                deadline = t_end + (self.cfg.vad.endpoint_ms + self.cfg.vad.continue_ms) / 1000
                while time.time() < deadline and self.sm.state is State.END_OF_TURN:
                    await asyncio.sleep(0.03)
                if self.sm.state is not State.END_OF_TURN:  # the user went on: this part starts the turn
                    logger.info(f"utterance continues after a pause: {text!r}")
                    self._carry_text = text
                    return
            if text and self._looks_like_echo(text):
                logger.warning(f"ignored probable echo of the assistant: {text!r}")
                self.outbox.json({"type": "echo_ignored", "text": text}, force=True)
                text = ""
            if not text:
                self._log_utterance(ev, "", 0)            # nothing usable heard (noise, or filtered)
                self.sm.fire(Event.TRANSCRIPT_EMPTY)
                self.outbox.json({"type": "transcript", "text": "", "final": True,
                                  "duration_ms": ev.stats.get("duration_ms")}, force=True)
                return
            if prefetch is not None and speculative:
                adopted = await prefetch.matches(ev.speech_end_t, text)
                if self.sm.state is not State.END_OF_TURN:  # resumed while checking (bounded wait)
                    adopted = False
                    self._carry_text = text
                    return
            turn = await self.start_turn("voice", text, t0=t_end, prefetch=prefetch if adopted else None)
        finally:
            if prefetch is not None and not adopted:
                prefetch.cancel()
        turn.mark("vad_endpoint", t_endpoint)
        turn.mark("stt_done", t_stt)
        turn.utterance_ms = ev.stats.get("duration_ms")
        turn.stt_speculative = speculative
        turn.stt_timing = getattr(self, "_stt_timing", None)
        self._log_utterance(ev, text, turn.turn_id)

    def _log_utterance(self, ev: VadEvent, text: str, turn_id: int) -> None:
        """What was heard, how loud and how sure: the first thing to look at when recognition is wrong."""
        audio = ev.audio if ev.audio is not None else np.zeros(1, np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        info = dict(self.svc.stt.last_info) if self.svc.stt else {}
        info.update(duration_ms=round(len(audio) / 16), rms_dbfs=round(float(20 * np.log10(max(rms, 1e-6))), 1),
                    peak=round(float(np.abs(audio).max()), 3) if len(audio) else 0.0)
        logger.info(f"turn {turn_id} heard {text!r} {info}")
        if self.cfg.stt.debug_audio:
            folder = self.cfg.log_dir / "stt_debug"
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.id}_turn{turn_id}"
            sf.write(stem.with_suffix(".wav"), audio, 16000)
            stem.with_suffix(".txt").write_text(f"{text}\n{info}\n", encoding="utf-8")

    def _looks_like_echo(self, text: str) -> bool:
        """True if the transcript mostly repeats the assistant's last words (speaker → mic leak)."""
        last = self.memory.turns[-1]["content"] if self.memory.turns and self.memory.turns[-1]["role"] == "assistant" else ""
        words = [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2]
        said = set(re.findall(r"[a-z']+", last.lower()))
        if len(words) < 3 or not said:
            return False
        return sum(w in said for w in words) / len(words) >= 0.8

    # ------------------------------------------------------------------ turns
    async def start_turn(self, source: str, text: str, t0: float | None = None,
                         prefetch: AnswerPrefetch | None = None, messages: list[dict] | None = None,
                         script: str | None = None) -> TurnContext:
        """A turn answering ``text``: from the prefetched answer, the LLM (with ``messages`` instead of
        the conversation, e.g. the greeting) or a fixed ``script`` (no LLM)."""
        if self.sm.state in ACTIVE:
            await self.interrupt("new_input")
        self._drop_prefetch()                  # e.g. typed text while a spoken answer was being prefetched
        turn = self.turns.new_turn(source, text)
        turn.mark("t0", t0)
        if prefetch is not None:
            turn.prefetched = True
            turn.tasks.append(prefetch.task)   # a barge-in cancels it like any other turn task
            prefetch.resume_fn = (lambda: self._resume_llm(turn)) if self.voice_out else None
            prefetch.adopted.set()
        self.sm.fire(Event.TEXT_INPUT if source in ("text", "greeting") else Event.TRANSCRIPT_READY)
        self.outbox.json({"type": "user_message", "turn_id": turn.turn_id, "text": text, "source": source},
                         turn.turn_id, force=True)
        task = asyncio.create_task(self._run_turn(turn, prefetch, messages, script), name=f"turn-{turn.turn_id}")
        turn.tasks.append(task)
        return turn

    async def _run_turn(self, turn: TurnContext, prefetch: AnswerPrefetch | None = None,
                        messages: list[dict] | None = None, script: str | None = None) -> None:
        try:
            await self._respond(turn, prefetch, messages, script)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"turn {turn.turn_id} failed")
            self.outbox.json({"type": "error", "turn_id": turn.turn_id, "message": str(exc)}, force=True)
            if self.turns.is_current(turn.turn_id):
                turn.cancel()
                self.sm.fire(Event.FAILURE)
                self.sm.fire(Event.RECOVER)

    async def _respond(self, turn: TurnContext, prefetch: AnswerPrefetch | None = None,
                       messages: list[dict] | None = None, script: str | None = None) -> None:
        tid = turn.turn_id
        citations: list[dict] = []
        if prefetch is not None:               # answer already streaming since the VAD pause
            stats = prefetch.stats
            deltas = prefetch.deltas()
        elif script is not None:               # a fixed reply (the greeting): no LLM
            stats = LLMStats()
            turn.mark("llm_request")
            deltas = _scripted(script)
        else:
            knowledge, citations = await self._retrieve(turn) if messages is None else (None, [])
            stats = LLMStats()
            turn.mark("llm_request")
            deltas = self.svc.llm.stream_chat(messages or self.memory.build(turn.user_text, knowledge), stats,
                                              pause_after=self._pause_after() if self.voice_out else None,
                                              resume=lambda: self._resume_llm(turn))
        parser, spoken = MetaStreamParser(), []
        speech = chunker = None
        if self.voice_out:
            speech = SpeechOutput(turn, self.svc.tts, self.outbox, self.svc.gpu_lock,
                                  on_first_audio=lambda: self._first_output(turn),
                                  avatar_sink=self.svc.avatar_sink_for(self, turn) if self.svc.avatar_ready else None)
            turn.speech = speech
            chunker = ClauseChunker()
        behavior_sent = False
        behavior = None
        first_clause = True
        tagged = self.cfg.llm.behavior_tag

        def say(clause: str) -> None:
            nonlocal first_clause, behavior, behavior_sent
            if first_clause and not tagged:
                # no tag: the behaviour comes from the user's words and the reply's first clause, and
                # reaches the avatar before this clause's audio does
                behavior = smile_for_happy_user(behavior_from_text(turn.user_text, clause), turn.user_text)
                self._send_behavior(turn, behavior)
                behavior_sent = True
            elif first_clause and behavior is not None:
                # the first words show the reply's stance ("Yes, ..." -> nod, "No, ..." -> shake) better
                # than the LLM's tag; refined before this clause's audio reaches the avatar
                refined = refine_behavior(behavior, clause)
                if refined is not None:
                    self._send_behavior(turn, refined)
            first_clause = False
            speech.say(clause)

        async for delta in deltas:
            if turn.is_cancelled:
                return
            if prefetch is not None:
                if self.svc.rag is not None:
                    turn.mark("rag_done", prefetch.t_rag)
                turn.mark("llm_request", prefetch.t_request)
            turn.mark("llm_first_token", prefetch.t_first_token if prefetch else None)
            _meta, text = parser.feed(delta)
            text = clean_delta(text)
            if text.startswith(" ") and spoken and spoken[-1].endswith(" "):
                text = text.lstrip(" ")        # "normal —" + " a" -> "normal, " + "a" (not "normal,  a")
            if not behavior_sent and (parser.meta is not None or text) and (tagged or parser.meta is not None):
                behavior = smile_for_happy_user(from_meta(parser.meta), turn.user_text)
                self._send_behavior(turn, behavior)
                behavior_sent = True
            elif not behavior_sent and text and chunker is None:      # no speech (muted): from the first words
                behavior = smile_for_happy_user(behavior_from_text(turn.user_text, text), turn.user_text)
                self._send_behavior(turn, behavior)
                behavior_sent = True
            if text:
                self._emit_text(turn, text, spoken)
                if chunker:
                    for clause in chunker.feed(text):
                        say(clause)
        tail = clean_delta(parser.flush())
        if tail:
            self._emit_text(turn, tail, spoken)
            if chunker:
                for clause in chunker.feed(tail):
                    say(clause)
        if chunker:
            for clause in chunker.flush():
                say(clause)
            speech.close()
        turn.mark("llm_done")
        if prefetch is not None:
            citations, turn.rag_stats = prefetch.citations, prefetch.rag_stats
        reply = "".join(spoken).strip()
        turn.header = parser.raw_header or ""
        if citations:
            self.outbox.json({"type": "citations", "turn_id": tid, "sources": citations}, tid)

        if speech is not None:
            await speech.finished.wait()
            if turn.is_cancelled:
                return
            if speech.avatar_sink is not None and speech.total_ms > 0:
                self.svc.avatar_speech_end(tid, speech.total_ms)     # the avatar closes its mouth after the last word
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
        knowledge, citations, turn.rag_stats = await self._retrieve_text(turn.user_text)
        if self.svc.rag is not None:
            turn.mark("rag_done")
        return knowledge, citations

    async def _retrieve_text(self, text: str) -> tuple[str | None, list[dict], dict | None]:
        if self.svc.rag is None:
            return None, [], None
        result = await asyncio.to_thread(self.svc.rag.retrieve, text)
        return result.context, result.citations, result.stats

    def _emit_text(self, turn: TurnContext, text: str, spoken: list[str]) -> None:
        spoken.append(text)
        turn.spoken_text = "".join(spoken)
        turn.mark("llm_first_text")
        self.outbox.json({"type": "llm_delta", "turn_id": turn.turn_id, "text": text}, turn.turn_id)
        if not self.voice_out:
            self._first_output(turn)

    def _pause_after(self):
        """LLM pause points for spoken replies: after every new speakable clause (as the speech chunker
        cuts them). Just-in-time writing: generating the rest of a reply at once shared the GPU with the
        first clause's synthesis and the avatar's first frames (slower start), or - resumed when the first
        audio was out - with the lip sync of the first sentences (12 % of the avatar's frames skipped).
        None with the behaviour tag (its header is not speech)."""
        if self.cfg.llm.behavior_tag:
            return None
        seen = 0

        def pause_after(text: str) -> bool:
            nonlocal seen
            n = len(ClauseChunker().feed(clean_delta(text)))
            if n > seen:
                seen = n
                return True
            return False
        return pause_after

    async def _resume_llm(self, turn: TurnContext) -> None:
        """When the paused LLM continues: the first time once the first audio is out, later when less
        than llm.speech_ahead_ms of speech is still to come. Bounded, so a reply never stalls."""
        try:
            if not turn.first_audio_sent.is_set():
                await asyncio.wait_for(turn.first_audio_sent.wait(), 8.0)
                return
            deadline = time.perf_counter() + 15.0
            while (not turn.is_cancelled and turn.speech is not None and time.perf_counter() < deadline
                   and turn.speech.backlog_ms() > self.cfg.llm.speech_ahead_ms):
                await asyncio.sleep(0.05)
        except TimeoutError:
            pass

    def _first_output(self, turn: TurnContext) -> None:
        turn.first_audio_sent.set()                     # the paused LLM continues (_resume_llm)
        if self.turns.is_current(turn.turn_id) and self.sm.state is State.THINKING:
            turn.mark("first_output")
            self.sm.fire(Event.FIRST_OUTPUT)

    def _on_playback(self, msg: dict) -> None:
        turn = self.turns.current
        if turn is None or msg.get("turn_id") != turn.turn_id:
            return
        event = msg.get("event")
        turn.played_ms = float(msg.get("played_ms", turn.played_ms))
        turn.playback_gap_ms = float(msg.get("gap_ms") or turn.playback_gap_ms)
        turn.played_at = time.time()
        if event == "started":
            turn.mark("client_first_audio")
            self.svc.avatar_playback(turn.turn_id, "started")
        elif event == "progress":
            self.svc.avatar_playback(turn.turn_id, "progress", turn.played_ms)   # lets the avatar catch up
        elif event == "ended":
            turn.mark("client_playback_ended")
            if turn.playback_gap_ms > 300:
                logger.warning(f"turn {turn.turn_id}: {turn.playback_gap_ms:.0f} ms of gaps in the spoken reply")
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
            turn = self.turns.current
            if turn is not None and turn.turn_id == turn_id and turn.played_at:
                # how late is this frame already here? (browser position + time since its last report)
                pts = HEADER.unpack_from(packet)[3]
                audible = turn.played_ms + min((time.time() - turn.played_at) * 1000, 250.0)
                turn.av_lateness.append(audible - pts)
                turn.av_max_backlog = max(turn.av_max_backlog, self.outbox.backlog)
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
        # The user went on talking right after a pause, before hearing much of the reply: that was the
        # same question, not an interruption. The next utterance completes it (seen in user testing:
        # one question became three turns, each answered by its last fragment).
        endpoint = turn.timings.get("vad_endpoint", turn.started_at)
        continued = (reason == "barge_in" and turn.source == "voice" and turn.played_ms < 1500
                     and time.time() - endpoint < self.cfg.vad.continuation_s)
        if continued:
            self._carry_text = turn.user_text
        elif heard:
            self.memory.add(turn.user_text, f"{turn.header or ''}{heard} [interrupted by the user]", interrupted=True)
        else:
            # stopped before a word of it was heard: the request is dropped from her history - kept, the
            # 4B model often answered the next message and then the stopped one ("OK. Let me tell you a
            # story about a dragon ..." after Stop on a story) - and the next message still gets the hint
            self.memory.after_interruption = True
        self.sm.fire(Event.CANCEL_DONE)
        cancel_ms = (time.perf_counter() - t_start) * 1000
        self.outbox.json({"type": "assistant_done", "turn_id": turn.turn_id, "text": heard, "interrupted": True,
                          "reason": reason, "continued": continued}, force=True)
        logger.info(f"turn {turn.turn_id} interrupted ({reason}) in {cancel_ms:.1f} ms; heard={heard[:60]!r}"
                    + (" - the user continued the question" if continued else ""))
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
            behavior = for_state(new.value)
            if new is State.THINKING and self.turns.current is not None:
                # a greeting, thanks or good news gets a smile right away instead of the thinking look
                behavior = smile_for_user_text(self.turns.current.user_text) or behavior
            self._send_behavior(None, behavior)
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
                  "llm_prefetched": turn.prefetched, "playback_gap_ms": round(turn.playback_gap_ms),
                  "avatar_forward": _lateness_stats(turn.av_lateness, turn.av_max_backlog),
                  "speech_ms": round(turn.speech.total_ms) if turn.speech else None,
                  "tts_first_timing": turn.speech.first_timing if turn.speech else None,
                  "rag": turn.rag_stats, "avatar_first_ms": turn.avatar_first_ms, "ts": time.time()}
        self.outbox.json(record, force=True)
        write_metrics(record)
