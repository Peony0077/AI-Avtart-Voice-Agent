"""One conversation session = one browser connection.

Owns the state machine, turn registry, conversation memory and outbox.
Inputs: typed text (``user_text``) or microphone PCM (binary frames → VAD → STT).
Outputs: streamed text, behaviour events, TTS audio packets and the avatar's video frames.
"""
from __future__ import annotations

import asyncio
import base64
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
    clause_behavior,
    for_state,
    for_user_pause,
    for_user_speaking,
    from_meta,
    refine_behavior,
    smile_for_happy_user,
    smile_for_user_text,
)
from app.behavior.meta import MetaStreamParser
from app.behavior.speech_text import SpeechTextStream
from app.core.gpu import locked_thread
from app.core.logging import write_metrics
from app.hpn import enquiry as hpn_enquiry
from app.hpn import match as hpn_match
from app.hpn.engine import Engine as HelperEngine
from app.hpn.engine import mask_private
from app.hpn.voice import spoken as helper_spoken
from app.leads import worth_reading
from app.llm.client import LLMStats
from app.llm.prompt import ConversationMemory, lead_rules
from app.services import Services
from app.session.endpointing import looks_unfinished
from app.session.outbox import VIDEO_WINDOW, Outbox
from app.session.prefetch import AnswerPrefetch
from app.session.state_machine import Event, State, StateMachine
from app.session.turn import TurnContext, TurnRegistry
from app.transport.protocol import FLAG_KEY, HEADER, KIND_IDLE_H264, KIND_IDLE_VIDEO, KIND_VIDEO_H264
from app.tts.chunker import ClauseChunker
from app.tts.speech_output import SpeechOutput

ACTIVE = (State.THINKING, State.SPEAKING)
# Inbound limits: a microphone frame is ~20-100 ms of PCM16 (a few KiB); a typed message is a question.
MAX_MIC_FRAME_BYTES = 64 * 1024
MAX_USER_TEXT_CHARS = 2000
# H.264 bitrate (kbit/s) of the reply's and of the idle video at full quality (worker defaults), and the
# lowest share of it on a slow connection (adapt_video)
VIDEO_KBPS = (700, 350)          # (speech, idle) kbit/s for a 512x512 avatar at 20 fps (video_kbps scales it)


def video_kbps(avatar: dict | None) -> tuple[int, int]:
    """Full-quality bitrates for this avatar: VIDEO_KBPS scaled by pixels^0.75 and frame rate. A 1024x1024,
    25 fps avatar (owner, 2026-10-10: "the resolution is not good") gets ~2.5/1.2 Mbit/s; the adaptation
    (next_video_scale) still lowers it on a slow connection, down to VIDEO_SCALE_MIN."""
    w, h = (avatar or {}).get("size") or (512, 512)
    fps = (avatar or {}).get("fps") or 20
    factor = max(1.0, (w * h / (512 * 512)) ** 0.75 * fps / 20)
    return round(VIDEO_KBPS[0] * factor), round(VIDEO_KBPS[1] * factor)
VIDEO_SCALE_MIN = 0.35


def next_video_scale(scale: float, ceiling: float, congested: bool, calm_s: float = 0.0,
                     floor: float = VIDEO_SCALE_MIN, fit: float | None = None) -> tuple[float, float]:
    """(share of VIDEO_KBPS, ceiling) after a reply: frames dropped -> 60 %, and the ceiling 75 % of the
    share that was too much; no drops -> 25 % more, up to the ceiling, which rises 5 % per such reply and
    is forgotten after a minute without drops (calm_s: seconds since the last drop) - a short slowdown
    does not keep the picture soft. A ceiling 80 % rising 10 % dropped frames every third reply on a link
    carrying half the video; this one about every seventh, and a probe a minute."""
    # floor: the lowest share (an absolute ~245 kbit/s whatever the avatar's size); fit: the share that the
    # measured delivery rate carries - one congested reply is then enough also from an HD avatar's 2.5 Mbit/s
    # (60 % steps took three replies to get under an 800 kbit/s link, and 35 % of 2.5 Mbit/s never did)
    if congested:
        new = scale * 0.6 if fit is None else min(scale * 0.6, fit)
        return max(floor, new), max(floor, min(scale * 0.75, new * 1.25))
    ceiling = 1.0 if calm_s > 60 else min(1.0, ceiling * 1.05)
    return min(ceiling, scale * 1.25), ceiling
# A first clause shorter than this ("Sure!", "Absolutely!", "Happy to help!": under ~1.5 s of speech)
# is over before the next one could be written after the first audio: the LLM writes on (clause_pauses)
SHORT_OPENER_CHARS = 24


def clause_pauses():
    """LLM pause points for spoken replies: after every new speakable clause (as the speech chunker
    cuts them), except after a short opener. Measured (2026-10-05): "Sure!" is 0.8 s of speech; the next
    sentence, written only once the first audio was out and with lip sync sharing the GPU (29 ms a token
    instead of 12), arrived 120 ms before it was due, and in one run 448 ms too late - a gap."""
    seen = 0

    def pause_after(text: str) -> bool:
        nonlocal seen
        # the same text the speech gets (_respond): a clause seen here is one the speech can start with
        clauses = ClauseChunker().feed(SpeechTextStream().feed(text))
        if len(clauses) <= seen:
            return False
        seen = len(clauses)
        return not (seen == 1 and len(clauses[0]) < SHORT_OPENER_CHARS)
    return pause_after


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


# Only the owner (owner link) changes the persona, reads the leads or previews voices; customers
# (customer link) talk to her and can make her forget them.
OWNER_ONLY = ("set_persona", "get_leads", "preview_voice")


class Session:
    def __init__(self, ws: WebSocket, services: Services, role: str = "owner", visitor_id: str | None = None) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.ws = ws
        self.svc = services
        self.cfg = services.settings
        self.role = role                  # "owner" | "customer"
        self.visitor_id = visitor_id      # the customer's browser (random id it keeps)
        self.last_activity = time.time()  # the customer spoke, typed or listened (SessionManager's idle hand-over)
        self.started = False              # waits in line until SessionManager starts it
        self.ready = False                # started, and what the page sent while it waited applied (api/ws.py)
        self.pending: dict[str, dict] = {}   # the page's hello, visible, voice, ... while it waited (by type)
        # this conversation's avatar stream (one per conversation slot, set by api/ws.py SessionManager when it
        # starts); the services' only one by default
        self.avatar_link = getattr(services, "avatar", None)
        self.sm = StateMachine()
        self.turns = TurnRegistry()
        persona = services.persona
        # Houston Party Network mode: every answer is the Event helper's (app/hpn); no memory, no leads
        self.hpn = HelperEngine(services.hpn_kb, persona.name) if services.hpn_kb is not None else None
        self._hpn_last: str | None = None          # the previous answer's intent ("and tomorrow?")
        # a customer's details, confirmed, sent to HPN as an enquiry (app/hpn/enquiry.py)
        self.enquiry = (hpn_enquiry.EnquiryFlow(self.hpn.business(), mode=services.settings.hpn.enquiries)
                        if self.hpn is not None and services.settings.hpn.enquiries != "off" else None)
        self._hpn_last_title: str | None = None
        # each customer has their own memory; the owner's is data/memory.json
        self.memory_store = (services.memory_store if role == "owner"
                             else services.visitor_memories.store(visitor_id or ""))
        self.memory = ConversationMemory(persona=persona.name, max_turns=self.cfg.llm.history_turns,
                                         knowledge_enabled=self.cfg.rag.enabled, behavior_tag=self.cfg.llm.behavior_tag,
                                         personality=persona.personality, long_term=self._long_term(),
                                         context_tokens=self.cfg.llm.context, reply_tokens=self.cfg.llm.max_tokens)
        self._apply_business()
        self._memory_gen = self.memory_store.generation if self.memory_store else 0   # erased later: nothing from before is kept
        self._greeted = False
        self.outbox = Outbox(ws, self.turns.is_current)
        self.voice_out = False            # replies are spoken (client plays assistant audio)
        self.speaker_on = False           # client asked for spoken replies (also for typed messages)
        self.speaker_set = False          # the page said so explicitly ("speaker"): then Mute wins over the mic
        self.mic_on = False               # client streams the microphone
        self.vad: StreamingVAD | None = None
        self.mic_stats = MicStats()
        self._stt_tasks: set[asyncio.Task] = set()   # transcriptions of finished utterances (_on_speech_end)
        self._utterance = 0               # +1 at every utterance end: an older transcription only carries its text
        self._carry_text = ""             # transcript of an utterance the user continued
        self._bad_frames = 0              # microphone frames refused (too long, odd length)
        self._spec: tuple[float, asyncio.Task] | None = None  # (speech_end_t, speculative STT task)
        self._prefetch: AnswerPrefetch | None = None           # answer started at the same pause
        self._summary_task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        self.dropped_idle_frames = 0
        self.video_codec = "jpeg"         # what the browser decodes (hello): "h264" or "jpeg"
        self._idle_wait_key = False       # H.264 idle frames were dropped: send again from a keyframe
        self._speech_wait_key = 0         # the reply (turn id) whose H.264 frames were dropped: from a keyframe on
        self._keyframe_asked = 0.0        # when the worker was last asked for a keyframe
        self.audio_codec = "pcm"          # the voice for the page (hello): "opus" or "pcm" (tts/opus.py)
        self.page_visible = True          # hidden page: no video (nobody sees it; it only filled the link)
        self.dropped_speech_frames = 0
        self.video_scale = 1.0            # share of VIDEO_KBPS for this connection (_adapt_video)
        self._video_ceiling = 1.0         # below the share at which frames were last dropped
        self._congested_at = 0.0
        self._adapted_at = 0.0
        self._drops_mark = 0              # dropped frames when the current reply started
        self.sm.subscribe(self._on_state_change)
        self._closed = False

    @property
    def avatar_ready(self) -> bool:
        link = getattr(self, "avatar_link", None)
        return link is not None and link.connected

    def _long_term(self) -> str | None:
        if getattr(self, "hpn", None) is not None:
            return None                   # HPN mode keeps no memory of customers
        return self.memory_store.load() if self.svc.persona.memory and self.memory_store is not None else None

    def _apply_business(self) -> None:
        """The business and lead-generation rules for the system prompt (persona.py)."""
        p = self.svc.persona
        self.memory.company = p.company
        self.memory.lead = lead_rules(p.company, p.offer, p.next_step) if p.lead_gen else ""

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.started = True
        self.last_activity = time.time()   # waiting in line is not inactivity (SessionManager's idle hand-over)
        self.outbox.start()
        self.outbox.json({"type": "session", "session_id": self.id, "persona": self.memory.persona,
                          "profile": self.cfg.profile, "voice_available": self.svc.voice_ready,
                          "avatar_available": self.avatar_ready,
                          "avatar": self.avatar_link.hello if self.avatar_ready else None, "role": self.role,
                          "hpn": self._hpn_session(), "server_time": time.time()}, force=True)
        self.sm.fire(Event.START)

    def _hpn_session(self) -> dict | None:
        if self.hpn is None:
            return None
        from app.hpn.engine import STARTER_CHIPS
        from app.hpn.knowledge import SITE

        b = self.hpn.business()
        return {"site": SITE, "phone_display": b["phone_display"], "phone_e164": b["phone_e164"],
                "email": b["email"], "chips": STARTER_CHIPS}

    async def close(self, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        if self.turns.current:
            self.turns.current.cancel()
        self.svc.avatar_reset(link=self.avatar_link)
        self._drop_prefetch()
        for task in (*self._stt_tasks, self._spec[1] if self._spec else None, self._summary_task, self._warm_task):
            if task and not task.done():
                task.cancel()
        self.sm.try_fire(Event.STOP)
        await self.outbox.stop()
        if (self.hpn is None and self.svc.persona.memory and self.memory_store is not None
                and (said := self.memory.user_said())):
            self.svc.remember(said, self._memory_gen, self.memory_store)   # what she remembers next time
        if self.hpn is None and self.svc.persona.lead_gen and worth_reading(self.memory.said):
            self.svc.capture_lead(self.memory.lead_transcript(), self.id, self.visitor_id)   # leads.py
        if (mic := self.mic_stats.summary()) is not None:
            logger.info(f"session {self.id} microphone audio: {mic}")
        logger.info(f"session {self.id} closed ({reason})")

    # ------------------------------------------------------------------ inbound
    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind in OWNER_ONLY and self.role != "owner":
            logger.warning(f"session {self.id}: '{kind}' refused (customer link)")
            return
        if kind in ("user_text", "playback", "greet") or kind == "voice" and msg.get("enabled"):
            self.last_activity = time.time()
        if kind == "hello":
            self.speaker_on = bool(msg.get("voice_out", False))
            self._update_voice_out()
            # H.264 when the browser decodes it (WebCodecs, secure pages): ~1 Mbit/s instead of 6.7 (h264.py)
            self.video_codec = "h264" if "h264" in (msg.get("video") or []) else "jpeg"
            self.svc.avatar_codec(self.video_codec, link=self.avatar_link)      # remembered: sent again if the worker restarts
            self.audio_codec = "opus" if "opus" in (msg.get("audio") or []) else "pcm"
            if msg.get("ack"):                           # the page acknowledges what it received (outbox.py)
                self.outbox.window = VIDEO_WINDOW
            self.video_scale = self._video_ceiling = 1.0  # a new connection starts at full quality
            self._apply_video_rate(*video_kbps(self.avatar_link.hello if self.avatar_ready else None))
            logger.info(f"session {self.id}: video {self.video_codec}, voice {self.audio_codec}, "
                        f"window {'on' if self.outbox.window else 'off'}")
            try:      # the customer's time zone (minutes east of UTC), for questions about the time
                offset = int(msg["utc_offset_min"])
                self.memory.utc_offset_min = offset if -14 * 60 <= offset <= 14 * 60 else None
            except (KeyError, TypeError, ValueError):
                pass
        elif kind == "ack":
            try:
                self.outbox.ack(int(msg.get("bytes") or 0))
            except (TypeError, ValueError):
                pass
        elif kind == "visible":
            self._set_visible(bool(msg.get("visible", True)))
        elif kind == "speaker":
            self.speaker_on = bool(msg.get("enabled"))
            self.speaker_set = True
            self._update_voice_out()
            self.outbox.json({"type": "speaker", "enabled": self.voice_out}, force=True)
        elif kind == "voice":
            if msg.get("mic"):
                mic = msg["mic"] if isinstance(msg["mic"], dict) else {}
                # cut: a page (a customer link too) could send megabytes here, and all of it went to the log
                logger.info(f"session {self.id} microphone: {str(mic)[:600]}")
                if mic.get("contextRate") not in (None, 16000):
                    logger.warning(f"browser runs the microphone at {mic['contextRate']} Hz: resampled to 16 kHz in the page")
                if mic.get("echoCancellation") is False:
                    logger.warning("browser echo cancellation is OFF: the avatar may hear itself (use headphones)")
            await self._set_voice(bool(msg.get("enabled")))
        elif kind == "user_text":
            text = str(msg.get("text", "")).strip()[:MAX_USER_TEXT_CHARS]
            if text:
                await self.start_turn("text", text)
        elif kind == "interrupt":
            await self.interrupt(str(msg.get("reason", "manual"))[:40])   # (logged and echoed back)
        elif kind == "playback":
            self._on_playback(msg)
        elif kind == "greet":
            await self.greet()
        elif kind == "get_persona":
            self._send_persona()
        elif kind == "set_persona":
            persona = await self.svc.set_persona(msg.get("persona") or {})
            self.memory.persona, self.memory.personality = persona.name, persona.personality
            self.memory.long_term = self._long_term()
            self._apply_business()
            self._send_persona()
        elif kind == "forget_memory":
            if self.memory_store is not None:
                self.memory_store.clear()
                self._memory_gen = self.memory_store.generation
            self.memory.long_term = None
            self.memory.forget_user()
            logger.info(f"session {self.id}: memory of the {'user' if self.role == 'owner' else 'customer'} erased (Forget me)")
            self._send_persona()
        elif kind == "get_leads":
            rows = list(reversed(self.svc.lead_store.load()))          # newest first
            self.outbox.json({"type": "leads", "leads": rows, "csv": self.svc.lead_store.to_csv(rows)}, force=True)
        elif kind == "preview_voice":
            voice = str(msg.get("voice", ""))
            wav = await self.svc.voice_preview(voice)
            self.outbox.json({"type": "voice_preview", "voice": voice,
                              "wav_b64": base64.b64encode(wav).decode() if wav else None}, force=True)
        elif kind == "avatar_start":
            try:
                gap = float(msg.get("gap"))
            except (TypeError, ValueError):
                gap = None
            if gap is not None and abs(gap) <= 40 and self.avatar_ready:
                self.svc.avatar_start_gap(gap, link=self.avatar_link)
        elif kind == "start_hold":
            # the page holds a reply's first audio this long (its jitter buffer, AvatarStage.jsx)
            try:
                self.svc.avatar_start_hold(max(0, min(1000, int(msg.get("ms") or 0))), link=self.avatar_link)
            except (TypeError, ValueError):
                pass
        elif kind == "client_error":
            # a problem the page showed the user (the microphone did not start, ...)
            logger.warning(f"session {self.id} browser problem ({str(msg.get('where'))[:40]}): "
                           f"{str(msg.get('name'))[:60]}: {str(msg.get('message'))[:300]} | {str(msg.get('userAgent'))[:300]}")
        elif kind == "ping":
            self.outbox.json({"type": "pong", "t": msg.get("t"), "server_time": time.time()}, force=True)
        else:
            logger.debug(f"unknown message type {kind!r}")

    async def on_binary(self, data: bytes) -> None:
        if len(data) > MAX_MIC_FRAME_BYTES or len(data) % 2:
            # not microphone PCM16 (an odd length made the VAD raise on every frame after it)
            self._bad_frames += 1
            if self._bad_frames in (1, 100, 10000):
                logger.warning(f"session {self.id}: refused {self._bad_frames} binary frame(s) ({len(data)} bytes)")
            return
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
            self.outbox.json({"type": "error", "code": "voice_unavailable", "message": "voice models are not loaded"},
                             force=True)
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
        # (with the microphone on, Mute used to be ignored: the replies stayed spoken)
        wanted = self.speaker_on if self.speaker_set else (self.speaker_on or self.mic_on)
        self.voice_out = wanted and self.svc.voice_ready

    def _send_persona(self) -> None:
        p = self.svc.persona
        if self.role != "owner":        # a customer sees her name and what she remembers about them
            self.outbox.json({"type": "persona", "name": p.name, "memory": p.memory,
                              "remembered": self._long_term()}, force=True)
            return
        self.outbox.json({"type": "persona", **p.to_dict(), "voices": self.svc.voices(),
                          "remembered": self._long_term()}, force=True)

    async def greet(self) -> None:
        """Say hello when the user starts talking to her (the page's Start voice), once per session and
        only before anything else was said: the configured greeting, or - when she remembers the user
        from earlier sessions - a short welcome back from the LLM."""
        if self._greeted or self.turns.current is not None or not self.voice_out:
            return
        self._greeted = True
        if self.hpn is not None:
            self.hpn.name = self.svc.persona.name
            reply = self.hpn.start().as_dict()
            await self.start_turn("greeting", GREETING_TEXT, script=helper_spoken(reply),
                                  helper={"reply": reply, "matched": None})
            return
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
        self.last_activity = time.time()
        self._cancel_summary()             # a turn is coming: the LLM is needed for it
        if self.sm.state is State.LISTENING:
            self._send_behavior(None, for_user_speaking())      # active listening: an attentive face
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
            await locked_thread(self.svc.gpu_lock, self.svc.stt.keep_warm)

    def _on_speech_pause(self, ev: VadEvent) -> None:
        """Start STT (and the answer) speculatively; reused at the endpoint if the user stayed silent."""
        if self.sm.state is not State.LISTENING or not len(ev.audio):
            return
        self._send_behavior(None, for_user_pause())             # "mm-hm": a small nod
        if self._spec and not self._spec[1].done():
            self._spec[1].cancel()
        self._drop_prefetch()
        stt = asyncio.create_task(self._stt(ev.audio), name="stt-spec")
        self._spec = (ev.speech_end_t, stt)
        if self.cfg.llm.prefetch_on_pause and self.hpn is None:     # HPN answers take a millisecond
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
        def transcribe():
            return time.perf_counter(), self.svc.stt.transcribe(audio)

        t_lock, text = await locked_thread(self.svc.gpu_lock, transcribe)
        t_end = time.perf_counter()
        self._stt_timing = {"lock_wait_ms": round((t_lock - t_req) * 1000, 1),
                            "thread_ms": round((t_end - t_lock) * 1000, 1), **self.svc.stt.last_timing}
        return text, time.time()

    async def _on_speech_end(self, ev: VadEvent) -> None:
        self.outbox.json({"type": "vad", "event": "end", **ev.stats}, force=True)
        if self.sm.state is not State.LISTENING:
            return
        self.sm.fire(Event.SPEECH_END)
        # The user may have paused, gone on and stopped again while the first part was still being
        # transcribed: that transcription only carries its text over (_transcribe), this one waits for it
        # and answers the whole utterance (the older one used to start a turn with the first half).
        previous = {t for t in self._stt_tasks if not t.done()}
        self._utterance += 1
        task = asyncio.create_task(self._transcribe_safe(ev, self._utterance, previous), name="stt")
        self._stt_tasks.add(task)
        task.add_done_callback(self._stt_tasks.discard)

    def _cancel_transcription(self) -> None:
        self._utterance += 1                      # a transcription still running starts no turn
        for task in list(self._stt_tasks):
            if not task.done() and task is not asyncio.current_task():
                task.cancel()
        if self._spec and not self._spec[1].done():
            self._spec[1].cancel()
        self._spec = None
        self._carry_text = ""

    async def _transcribe_safe(self, ev: VadEvent, utterance: int = 0, previous: set | None = None) -> None:
        try:
            await self._transcribe(ev, utterance or self._utterance, previous)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. CUDA out of memory in Whisper
            logger.exception(f"session {self.id}: speech recognition failed")
            self.outbox.json({"type": "error", "code": "stt_failed", "message": f"speech recognition failed: {exc}"},
                             force=True)
            # only for the newest utterance: an older one failing must not cancel the one being transcribed now
            if self.sm.state is State.END_OF_TURN and (utterance or self._utterance) == self._utterance:
                self.sm.fire(Event.TRANSCRIPT_EMPTY)

    async def _transcribe(self, ev: VadEvent, utterance: int, previous: set | None = None) -> None:
        t_end = ev.speech_end_t or ev.t
        t_endpoint = time.time()

        def superseded() -> bool:                   # the user went on (or typed) after this utterance
            return self.sm.state is not State.END_OF_TURN or utterance != self._utterance

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
            if previous:                                    # the earlier part's text first (its carry)
                await asyncio.wait(previous)
            if superseded():                                # user resumed speaking meanwhile
                self._carry_text = f"{self._carry_text} {text}".strip()
                return
            text = f"{self._carry_text} {text}".strip()
            self._carry_text = ""
            if text and self.cfg.vad.continue_ms and looks_unfinished(text):
                # probably a pause in the middle of the sentence: wait a little longer for the rest
                deadline = t_end + (self.cfg.vad.endpoint_ms + self.cfg.vad.continue_ms) / 1000
                while time.time() < deadline and not superseded():
                    await asyncio.sleep(0.03)
                if superseded():                            # the user went on: this part starts the turn
                    logger.info(f"utterance continues after a pause: {self._loggable(text)!r}")
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
                if superseded():                            # resumed while checking (bounded wait)
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
        logger.info(f"turn {turn_id} heard {self._loggable(text)!r} {info}")
        if self.cfg.stt.debug_audio and self.hpn is None:    # HPN mode keeps no voice (avatar_overrides.json)
            folder = self.cfg.log_dir / "stt_debug"
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.id}_turn{turn_id}"
            sf.write(stem.with_suffix(".wav"), audio, 16000)
            stem.with_suffix(".txt").write_text(f"{text}\n{info}\n", encoding="utf-8")

    def _loggable(self, text: str) -> str:
        """What the log may keep of what the user said: in HPN mode card numbers, phone numbers and email
        addresses are masked (HPN's own rule, and what the avatar tells customers)."""
        return mask_private(text) if self.hpn is not None else text

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
                         script: str | None = None, helper: dict | None = None) -> TurnContext:
        """A turn answering ``text``: from the prefetched answer, the LLM (with ``messages`` instead of
        the conversation, e.g. the greeting) or a fixed ``script`` (no LLM)."""
        if self.sm.state in ACTIVE:
            await self.interrupt("new_input")
        if self.sm.state is State.END_OF_TURN and source in ("text", "greeting"):
            self._cancel_transcription()       # typed while the spoken words were transcribed: the typed text wins
        self._drop_prefetch()                  # e.g. typed text while a spoken answer was being prefetched
        self._cancel_summary()                 # the reply needs the LLM (the summary is tried after it again)
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
        task = asyncio.create_task(self._run_turn(turn, prefetch, messages, script, helper),
                                   name=f"turn-{turn.turn_id}")
        turn.tasks.append(task)
        return turn

    async def _run_turn(self, turn: TurnContext, prefetch: AnswerPrefetch | None = None,
                        messages: list[dict] | None = None, script: str | None = None,
                        helper: dict | None = None) -> None:
        try:
            await self._respond(turn, prefetch, messages, script, helper)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"turn {turn.turn_id} failed")
            self.outbox.json({"type": "error", "code": "turn_failed", "turn_id": turn.turn_id, "message": str(exc)},
                             force=True)
            if self.turns.is_current(turn.turn_id):
                turn.cancel()
                self.sm.fire(Event.FAILURE)
                self.sm.fire(Event.RECOVER)

    async def _respond(self, turn: TurnContext, prefetch: AnswerPrefetch | None = None,
                       messages: list[dict] | None = None, script: str | None = None,
                       helper: dict | None = None) -> None:
        tid = turn.turn_id
        citations: list[dict] = []
        if self.hpn is not None and prefetch is None and messages is None and script is None:
            # HPN mode: the Event helper's answer (app/hpn), said as written - no LLM writes it
            helper = await self._hpn_answer(turn)
            script = helper_spoken(helper["reply"])
        if helper is not None:                 # the page shows the formatted answer, cards and buttons
            self.outbox.json({"type": "helper_reply", "turn_id": tid, **helper}, tid, force=True)
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
        parser, spoken, cleaner = MetaStreamParser(), [], SpeechTextStream()
        speech = chunker = None
        if self.voice_out:
            speech = SpeechOutput(turn, self.svc.tts, self.outbox, self.svc.gpu_lock,
                                  on_first_audio=lambda: self._first_output(turn),
                                  avatar_sink=self.svc.avatar_sink_for(self, turn) if self.avatar_ready else None,
                                  audio_codec=self.audio_codec)
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
                    behavior = refined
                    self._send_behavior(turn, refined)
            elif behavior is not None:
                # a later sentence: its own face if it is happy or sad (the avatar blends into it)
                changed = clause_behavior(behavior, clause)
                if changed is not None:
                    behavior = changed
                    self._send_behavior(turn, changed)
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
            text = cleaner.feed(text)              # whole words (a URL or an email is cleaned in one piece)
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
        tail = cleaner.feed(parser.flush()) + cleaner.flush()
        if tail.startswith(" ") and spoken and spoken[-1].endswith(" "):
            tail = tail.lstrip(" ")
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
            if speech.error is not None:
                self.outbox.json({"type": "error", "code": "voice_failed", "turn_id": tid,
                                  "message": f"the voice failed for this reply: {speech.error}"}, force=True)
            if speech.avatar_sink is not None and speech.total_ms > 0:
                self.svc.avatar_speech_end(tid, speech.total_ms, link=self.avatar_link)     # the avatar closes its mouth after the last word
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

    async def _hpn_answer(self, turn: TurnContext) -> dict:
        """The Event helper's reply to the turn; an unclear question re-matched by the LLM (app/hpn/match.py)."""
        engine = self.hpn
        engine.name = self.svc.persona.name
        text = turn.user_text
        flow, dropped = self.enquiry, False
        reply = None
        if flow is not None and flow.active:
            reply = flow.handle(text)
            if reply is None:                 # they asked something else: nothing is sent, and they are told
                flow.draft.step, dropped = "cancelled", True
        if reply is None and flow is not None and flow.wants_to_start(text):
            reply = flow.start(text)
        if reply is not None and reply.intent == "enquiry.sending":
            status, body = await hpn_enquiry.post(flow.pending, self.id)
            reply = flow.finish(status, body)
        if reply is not None:
            flow.remember(text, reply.intent)
            self._hpn_last = reply.intent
            logger.info(f"turn {turn.turn_id}: HPN enquiry step {reply.intent}")
            return {"reply": reply.as_dict(), "matched": None}
        reply = engine.ask(text, last=self._hpn_last)
        if dropped:
            reply.text = f"I haven't sent your details.\n\n{reply.text}"
        matched = None
        if reply.kind == "fallback" and self.cfg.hpn.llm_match:
            cands = hpn_match.candidates(engine)
            try:
                turn.mark("llm_request")
                answer = await asyncio.wait_for(
                    self.svc.llm.complete(hpn_match.prompt(cands, text, self._hpn_last_title), max_tokens=6,
                                          temperature=0.0), self.cfg.hpn.llm_match_timeout_s)
                pick = hpn_match.parse(answer, len(cands))
            except Exception as exc:  # noqa: BLE001 - the helper's own "not sure" answer stands
                logger.warning(f"turn {turn.turn_id}: HPN re-match failed: {exc!r}")
                pick = 0
            if pick:
                again = engine.ask(cands[pick - 1].question, last=self._hpn_last)
                if again.confident and again.kind in ("answer", "live"):
                    reply, matched = again, cands[pick - 1].label
            logger.info(f"turn {turn.turn_id}: HPN re-match -> {matched or 'none'}")
        self._hpn_last = reply.intent
        self._hpn_last_title = matched or (reply.intent if reply.kind in ("answer", "live") else None)
        if flow is not None:
            flow.remember(text, reply.intent)
        logger.info(f"turn {turn.turn_id}: HPN answer {reply.intent} (confident {reply.confident})")
        return {"reply": reply.as_dict(), "matched": matched}

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
        """LLM pause points for spoken replies (clause_pauses). Just-in-time writing: generating the rest
        of a reply at once shared the GPU with the first clause's synthesis and the avatar's first frames
        (slower start), or - resumed when the first audio was out - with the lip sync of the first
        sentences (12 % of the avatar's frames skipped). None with the behaviour tag (its header is not
        speech)."""
        if self.cfg.llm.behavior_tag:
            return None
        return clause_pauses()

    async def _resume_llm(self, turn: TurnContext) -> None:
        """When the paused LLM continues: the first time once the first audio is out, later when less
        than llm.speech_ahead_ms of speech is still to come. Bounded, so a reply never stalls."""
        try:
            if not turn.first_audio_sent.is_set():
                await asyncio.wait_for(turn.first_audio_sent.wait(), 8.0)
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
            self._drops_mark = self.dropped_speech_frames + self.dropped_idle_frames
            turn.mark("client_first_audio")
            self.svc.avatar_playback(turn.turn_id, "started", link=self.avatar_link)
        elif event == "progress":
            self.svc.avatar_playback(turn.turn_id, "progress", turn.played_ms, link=self.avatar_link)   # lets the avatar catch up
        elif event == "ended":
            if self.video_codec == "h264" and self.page_visible:
                self._adapt_video(self.dropped_speech_frames + self.dropped_idle_frames > self._drops_mark)
            turn.mark("client_playback_ended")
            if turn.playback_gap_ms > 300:
                logger.warning(f"turn {turn.turn_id}: {turn.playback_gap_ms:.0f} ms of gaps in the spoken reply")
            turn.playback_ended.set()
            self.svc.avatar_playback(turn.turn_id, "ended", link=self.avatar_link)

    def _ask_keyframe(self, speech: bool, idle: bool = True) -> None:
        """Frames were dropped: the worker continues from a keyframe (at most every 2 s; the streams have
        one every 2 s anyway)."""
        if time.time() - self._keyframe_asked < 2.0:
            return
        self._keyframe_asked = time.time()
        if idle:
            self.svc.avatar_idle_keyframe(link=self.avatar_link)
        if speech:
            self.svc.avatar_speech_keyframe(link=self.avatar_link)

    def _apply_video_rate(self, speech: int, idle: int) -> None:
        """The worker's bitrates. (The outbox window stays VIDEO_WINDOW: scaled with an HD bitrate it queued
        ~1.5 s of video ahead of the voice on an 800 kbit/s link.)"""
        self.svc.avatar_bitrate(speech, idle, link=self.avatar_link)

    def _adapt_video(self, congested: bool) -> bool:
        """The video's bitrate follows the connection: when frames had to be dropped (the link is slower
        than the video) the next video gets 60 % of the bitrate, after a reply without drops 25 % more,
        up to full quality (as real-time video does, multiplicative decrease and gradual increase). At
        500 kbit/s a 700 kbit/s video could only freeze; at ~250 it is a little softer and keeps moving.
        It climbs back only below the share at which frames were last dropped (next_video_scale): going
        straight back up dropped frames every other reply (500 kbit/s: 44 %, 5 %, 21 % of the lips late)."""
        now = time.time()
        if congested and now - self._adapted_at < 2.0:
            return False
        full = video_kbps(self.avatar_link.hello if self.avatar_ready else None)
        floor = VIDEO_SCALE_MIN * VIDEO_KBPS[0] / full[0]
        fit = None
        delivered = self.outbox.delivery_kbps() if congested else None
        if delivered:                    # 70 % of what the link carried, less the voice (Opus ~56 kbit/s + overhead)
            fit = max(0.0, 0.7 * delivered - 64) / full[0]
        scale, self._video_ceiling = next_video_scale(self.video_scale, self._video_ceiling, congested,
                                                      now - self._congested_at, floor, fit)
        if congested:
            self._congested_at = now
        if abs(scale - self.video_scale) < 0.02:
            return False
        self.video_scale, self._adapted_at = scale, now
        speech, idle = round(full[0] * scale), round(full[1] * scale)
        self._apply_video_rate(speech, idle)
        logger.info(f"session {self.id}: video at {speech}/{idle} kbit/s "
                    f"({'frames were dropped' if congested else 'the connection keeps up'}"
                    f"{f', the link carried {delivered:.0f} kbit/s' if delivered else ''})")
        return True                                       # (the worker restarts the idle video at the new rate)

    def _set_visible(self, visible: bool) -> None:
        """The page was hidden (another tab, minimised) or shown again. Hidden, the browser hardly runs the
        page: the video it was sent piled up, the server's keepalive pings could not get through, and the
        connection was closed every 40 s (owner's tab, 2026-10-05). Now no video while hidden (the voice
        goes on); shown again, the video continues from keyframes (the page restarts its decoders)."""
        if visible == self.page_visible:
            return
        self.page_visible = visible
        logger.info(f"session {self.id}: page {'visible' if visible else 'hidden'}")
        self.outbox.clear_lane("idle")
        if visible and self.video_codec == "h264":
            self._idle_wait_key = True
            self._speech_wait_key = self.turns.current_id
            self._keyframe_asked = 0.0
            self._ask_keyframe(speech=True)

    def avatar_restarted(self, hello: dict) -> None:
        """The avatar worker (re)connected during this session: its size, fps and frame numbers to the page."""
        if self.started:
            self.outbox.json({"type": "avatar", "avatar": hello}, force=True)

    def on_avatar_frame(self, packet: bytes, kind: int, turn_id: int) -> None:
        """Avatar worker frame → browser, behind the voice (outbox lanes). Idle frames give way when the
        link is backed up: a JPEG one is simply dropped; H.264 frames depend on the ones before, so the
        queued idle video is dropped as a whole and starts again from the next keyframe."""
        if not self.page_visible:
            self.dropped_idle_frames += kind in (KIND_IDLE_VIDEO, KIND_IDLE_H264)
            self.dropped_speech_frames += kind not in (KIND_IDLE_VIDEO, KIND_IDLE_H264)
            return
        if kind == KIND_IDLE_VIDEO:
            if self.outbox.backlog > 6:
                self.dropped_idle_frames += 1
                return
            self.outbox.binary(packet, 0, force=True, lane="idle")
        elif kind == KIND_IDLE_H264:
            if self._idle_wait_key:
                if not packet[1] & FLAG_KEY:
                    self.dropped_idle_frames += 1
                    return
                self._idle_wait_key = False
            if self.outbox.lane_backlog("idle") > 20:          # 1 s of idle video waiting: the link is slow
                self.dropped_idle_frames += self.outbox.clear_lane("idle") + 1
                self._idle_wait_key = True
                adapted = self.turns.current is None or self.turns.current.played_at == 0   # between replies
                adapted = adapted and self._adapt_video(congested=True)
                self._ask_keyframe(speech=False, idle=not adapted)
                return
            self.outbox.binary(packet, 0, force=True, lane="idle")
        elif kind == KIND_VIDEO_H264:
            # The reply's video on a link slower than the video: frames already later than the voice are
            # useless, and they would delay the next ones. Over 0.5 s waiting: dropped, the reply's video
            # continues from a keyframe (the lips catch up instead of lagging further and further).
            if self._speech_wait_key == turn_id:
                if not packet[1] & FLAG_KEY:
                    self.dropped_speech_frames += 1
                    return
                self._speech_wait_key = 0
            if self.outbox.lane_backlog("speech") > 10:
                self.dropped_speech_frames += self.outbox.clear_lane("speech") + 1
                self._speech_wait_key = turn_id
                adapted = self._adapt_video(congested=True)     # first: the reply continues on a lighter stream
                self._ask_keyframe(speech=True, idle=not adapted)
                return
            self._forward_speech(packet, turn_id)
        else:
            self._forward_speech(packet, turn_id)

    def _forward_speech(self, packet: bytes, turn_id: int) -> None:
        turn = self.turns.current
        if turn is not None and turn.turn_id == turn_id and turn.played_at:
            # how late is this frame already here? (browser position + time since its last report)
            pts = HEADER.unpack_from(packet)[3]
            audible = turn.played_ms + min((time.time() - turn.played_at) * 1000, 250.0)
            turn.av_lateness.append(audible - pts)
            turn.av_max_backlog = max(turn.av_max_backlog, self.outbox.backlog)
        self.outbox.binary(packet, turn_id, lane="speech")

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
        if self.avatar_ready:
            await self.svc.avatar_cancel(turn.turn_id, link=self.avatar_link)
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
        if self.hpn is not None:
            return                        # nothing to summarize for: HPN answers do not use the conversation
        if not self.memory.needs_summary() or (self._summary_task and not self._summary_task.done()):
            return

        async def run() -> None:
            await asyncio.sleep(0.5)
            if self.sm.state is not State.LISTENING:
                return
            try:
                # only the turns in this request leave the archive: those archived meanwhile stay for the next
                count = self.memory.summary_count()
                text = await self.svc.llm.complete(self.memory.summary_request(count), max_tokens=120)
                self.memory.apply_summary(text, count)
                logger.info(f"rolling summary updated ({len(text)} chars)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"summary failed: {exc!r}")

        self._summary_task = asyncio.create_task(run(), name="summary")

    def _cancel_summary(self) -> None:
        """The user speaks or a turn starts: the summary request would compete with the reply for the LLM.
        Cancelled (the request is closed, llama-server stops it); it is tried again after the next reply."""
        task = self._summary_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

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
        if self.avatar_ready:
            self.svc.avatar_state(new.value, link=self.avatar_link)

    def _send_behavior(self, turn: TurnContext | None, behavior) -> None:
        self.outbox.json({"type": "behavior", "turn_id": turn.turn_id if turn else 0, **behavior.to_dict()},
                         force=turn is None)
        if self.avatar_ready:
            self.svc.avatar_behavior(behavior.to_dict(), link=self.avatar_link)

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
