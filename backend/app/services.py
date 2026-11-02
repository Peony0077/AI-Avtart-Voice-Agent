"""Process-wide services (models, clients, subprocess supervisors).

Created once in the FastAPI lifespan and shared by all sessions. Heavy models
are loaded here (never per turn) and warmed up before the first session.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace

from loguru import logger

from app.core.config import Settings
from app.core.gpu import gpu_snapshot, locked_thread, vram_free_gib
from app.leads import LeadStore, is_lead, lead_request, parse_lead
from app.llm.client import LLMClient
from app.llm.prompt import lead_rules, system_prompt
from app.llm.server import LlamaServerSupervisor
from app.persona import MemoryStore, Persona, PersonaStore, VisitorMemories, memory_request

# Measured on the RTX 3060 (docs/ARCHITECTURE.md); used to refuse loads that would make Windows
# spill VRAM into system RAM (which silently makes everything ~10x slower).
VRAM_NEED_MIB = {
    "stt:openai/whisper-large-v3-turbo": 2000, "stt:openai/whisper-small.en": 800,
    "tts:kokoro": 700, "tts:clone": 2600, "tts:clone_light": 900,
    # measured 2026-10-05 (dev_girl_v3, TAESD): 2610 MiB loaded and warm, ~3000 MiB peak while rendering (torch 2484
    # reserved + CUDA context). 4000 refused the avatar with 4.35 GiB free while desktop apps held 2.5 GiB.
    "avatar": 3300,
}
# Customers are in the US and Canada: American voices only (the British ones need another phonemizer
# and accent). Kokoro averages comma-separated voices: Heart's warmth with Bella's brighter energy.
VOICE_BLENDS = ("af_heart,af_bella", "af_heart,af_nicole")
# Memory and lead extraction after a conversation use the same LLM as the conversations. They used
# to start at once and the next conversation's reply waited behind them (a 890 ms gap in the e2e run;
# with a waiting line the next customer starts the moment the previous one leaves). Now they run only
# when no conversation has needed the LLM for BACKGROUND_QUIET_S, and stop (to retry later) as soon as
# one does. A job is never dropped: after BACKGROUND_MAX_WAIT_S (a steady line of customers kept the LLM
# busy and leads were given up after 6 tries) it runs to the end and the next reply waits for it. At
# shutdown the jobs still queued run (at most BACKGROUND_DRAIN_S) before the LLM server stops.
BACKGROUND_QUIET_S = 1.5
BACKGROUND_MAX_WAIT_S = 180.0
BACKGROUND_DRAIN_S = 60.0
PREVIEW_TEXT = ("Hi there! I'm so glad you stopped by. Tell me a little about what you're looking for, "
                "and I'll be happy to help.")


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        os.environ.setdefault("HF_HOME", str(settings.model_home / "hf"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are pre-downloaded; no network at runtime
        self.llm_server = LlamaServerSupervisor(settings)
        self.llm = LLMClient(settings.llm, wait_ready=self.llm_server.wait_until_ready
                             if settings.llm.manage_server else None)
        self.stt = None
        self.tts = None
        self.rag = None
        self.avatar = None              # the first avatar connection (the one that runs the worker process)
        self.avatars: list = []         # one per conversation slot (settings.conversations)
        self.hpn_kb = None              # Houston Party Network mode: the Event helper's knowledge (app/hpn)
        self.load_report: dict[str, float] = {}
        self.warnings: list[str] = []
        self.gpu_lock = asyncio.Lock()  # serialises our own GPU bursts (STT / TTS)
        # persona and memory of the user (persona.py), set from the page; stored under data/
        self.persona_store = PersonaStore(settings.repo_path("data/persona.json"),
                                          Persona(name=settings.persona_name, voice=settings.tts.voice))
        self.memory_store = MemoryStore(settings.repo_path("data/memory.json"))     # the owner's
        self.visitor_memories = VisitorMemories(settings.repo_path("data/visitors"))  # each customer's
        self.lead_store = LeadStore(settings.repo_path("data/leads.jsonl"))
        self.persona = self.persona_store.load()
        self._background: set[asyncio.Task] = set()
        self._llm_jobs: list[tuple[str, object, float]] = []   # (name, job, queued at: time.monotonic())
        self._llm_runner: asyncio.Task | None = None
        self._llm_job: asyncio.Task | None = None   # the job running now
        self._draining = False                      # shutting down: run what is queued, no more waiting
        self.conversation_busy = lambda: False      # set once by main.py: a conversation is using the LLM

    def voices(self) -> list[str]:
        if self.tts is None or not hasattr(self.tts, "voices"):
            return []
        stock = [v for v in self.tts.voices() if v.startswith("a")]
        blends = [b for b in VOICE_BLENDS if all(v in stock for v in b.split(","))]
        # the persona's voice if it is another one that exists here (e.g. a British one); not when its
        # files are missing: then it is not offered, and set_voice would fail
        voice = self.persona.voice
        current = [voice] if voice and voice not in stock + blends and self._voice_available(voice) else []
        return blends + stock + current

    def _voice_available(self, voice: str) -> bool:
        """``voice`` (or each voice of a blend) is in the local model cache."""
        if self.tts is None or not hasattr(self.tts, "voices"):
            return False
        have = set(self.tts.voices())
        return bool(voice) and all(v in have for v in voice.split(","))

    async def voice_preview(self, voice: str) -> bytes | None:
        """A short sample of ``voice`` as WAV (Settings: "Listen"), without changing the persona."""
        if voice not in self.voices() or not hasattr(self.tts, "synthesize_with"):
            return None
        import io

        import numpy as np
        import soundfile as sf

        async with self.gpu_lock:
            result = await asyncio.to_thread(self.tts.synthesize_with, PREVIEW_TEXT, voice)
        buf = io.BytesIO()
        sf.write(buf, (np.clip(result.audio, -1, 1) * 32767).astype(np.int16), result.sample_rate, format="WAV")
        return buf.getvalue()

    async def set_persona(self, data: dict) -> Persona:
        """Apply and store the persona settings from the page (valid fields only)."""
        persona = PersonaStore.merge(self.persona, data, self.voices())
        if persona.voice != self.persona.voice and hasattr(self.tts, "set_voice"):
            # under the GPU lock and off the event loop: the cloned voice (converted_tts) measures the new
            # base voice's tone colour, a few short syntheses
            try:
                await locked_thread(self.gpu_lock, self.tts.set_voice, persona.voice)
            except Exception as exc:  # noqa: BLE001 - the voice stays as it was
                logger.error(f"voice {persona.voice!r} could not be loaded: {exc!r}")
                persona = replace(persona, voice=self.persona.voice)
        self.persona = persona
        self.persona_store.save(persona)
        logger.info(f"persona updated: {persona.name}, voice {persona.voice}, memory {'on' if persona.memory else 'off'}")
        return persona

    def remember(self, transcript: str, generation: int | None = None, store: MemoryStore | None = None) -> None:
        """After a session: merge what the user said into the memory of earlier sessions (background).
        ``generation``: the memory's when the session began (or the user last said "Forget me" in it); if
        the memory was erased since, nothing is saved. ``store``: a customer's own memory (default: the
        owner's)."""
        store = store or self.memory_store
        generation = store.generation if generation is None else generation

        async def run() -> None:
            try:
                text = await self.llm.complete(memory_request(store.load(), transcript, self._transcript_room(160)),
                                               max_tokens=160)
                if not text.strip():
                    return
                if store.save(text, generation):
                    logger.info(f"memory of the user updated ({len(text.split())} words)")
                else:
                    logger.info("memory update dropped: the user asked to be forgotten meanwhile")
            except Exception as exc:  # noqa: BLE001 - the next session simply starts without it
                logger.warning(f"memory update failed: {exc!r}")

        self._queue_llm_job("memory update", run)

    def _transcript_room(self, reply_tokens: int) -> int | None:
        """Characters a background job's transcript may use: the context minus the reply and ~500 tokens
        of instructions/memory, at 3 characters a token (below prompt.CHARS_PER_TOKEN: digits, emails and
        names tokenize worse than plain English)."""
        settings = getattr(self, "settings", None)      # (None: a stripped-down Services in the tests)
        return None if settings is None else max(2000, (settings.llm.context - reply_tokens - 500) * 3)

    def capture_lead(self, transcript: str, session_id: str, visitor_id: str | None = None) -> None:
        """After a conversation: the lead in it, if any (leads.py), saved for the owner (background).
        The log says only whether there was one: names and contact details stay out of the logs."""
        async def run() -> None:
            try:
                text = await self.llm.complete(lead_request(transcript, self.persona.company,
                                                            self._transcript_room(320)), max_tokens=320)
                lead = parse_lead(text)
                if is_lead(lead):
                    self.lead_store.save(lead, session_id, visitor_id)
                    logger.info(f"session {session_id}: lead saved (interest {lead['interest'] or 'unknown'}, "
                                f"contact {'yes' if lead['email'] or lead['phone'] else 'no'})")
                else:
                    logger.info(f"session {session_id}: no lead in this conversation")
            except Exception as exc:  # noqa: BLE001 - the conversation is over either way
                logger.warning(f"lead capture failed: {exc!r}")

        self._queue_llm_job(f"lead capture ({session_id})", run)

    def pending_background(self) -> int:
        """Memory and lead jobs not finished yet (the e2e data guard waits for 0)."""
        running = self._llm_runner is not None and not self._llm_runner.done()
        return len(self._llm_jobs) + (1 if running else 0)

    def _queue_llm_job(self, name: str, job) -> None:
        self._llm_jobs.append((name, job, time.monotonic()))
        self._ensure_llm_runner()

    def _ensure_llm_runner(self) -> asyncio.Task:
        if self._llm_runner is None or self._llm_runner.done():
            self._llm_runner = asyncio.get_running_loop().create_task(self._run_llm_jobs(), name="llm-background")
            self._background.add(self._llm_runner)
            self._llm_runner.add_done_callback(self._background.discard)
        return self._llm_runner

    def _may_wait(self, queued: float) -> bool:
        """The job may still give way to a conversation (not shutting down, not waiting too long already)."""
        return not self._draining and time.monotonic() - queued < BACKGROUND_MAX_WAIT_S

    async def _run_llm_jobs(self) -> None:
        """Background LLM jobs, one at a time (oldest first), while no conversation needs the LLM."""
        step = min(0.25, BACKGROUND_QUIET_S / 2)
        while self._llm_jobs:
            name, job, queued = self._llm_jobs[0]
            quiet = 0.0
            while quiet < BACKGROUND_QUIET_S and self._may_wait(queued):
                await asyncio.sleep(step)
                quiet = 0.0 if self.conversation_busy() else quiet + step
            if not self._may_wait(queued) and not self._draining:
                logger.warning(f"{name}: waited {BACKGROUND_MAX_WAIT_S:.0f} s for a quiet moment; runs now")
            self._llm_jobs.pop(0)
            task = self._llm_job = asyncio.get_running_loop().create_task(job(), name=name)
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if not task.done() and self.conversation_busy() and self._may_wait(queued):
                    task.cancel()                   # closes its request: the LLM stops at once
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self._llm_jobs.insert(0, (name, job, queued))   # first again when it is quiet
                    logger.info(f"{name}: paused, a conversation needs the LLM (retried later)")
                    break

    async def drain_background(self, timeout_s: float = BACKGROUND_DRAIN_S) -> None:
        """Shutdown: the memory and lead jobs still queued (close() of the last conversations queued them)
        run now, without waiting for a quiet moment, within ``timeout_s``; they were lost before."""
        if not self.pending_background():
            return
        self._draining = True
        runner = self._ensure_llm_runner()
        logger.info(f"shutdown: running {self.pending_background()} background job(s) first")
        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout_s)
        except TimeoutError:
            logger.warning(f"shutdown: {self.pending_background()} memory/lead job(s) not done in {timeout_s:.0f} s")
            for task in (runner, self._llm_job):
                if task is not None and not task.done():
                    task.cancel()

    @property
    def voice_ready(self) -> bool:
        return self.stt is not None and self.tts is not None

    @property
    def avatar_ready(self) -> bool:
        return self.avatar is not None and self.avatar.connected

    async def start(self) -> None:
        t0 = time.perf_counter()
        if self.settings.llm.manage_server:
            await asyncio.to_thread(self.llm_server.start)
        self.load_report["llm_server_s"] = round(time.perf_counter() - t0, 2)
        if self.settings.hpn.enabled:
            self._load_hpn()           # before the LLM warm-up: it warms the HPN matching prompt
        await self._warm_llm()
        if self.settings.voice_enabled:
            await asyncio.to_thread(self._load_voice)
        if self.settings.rag.enabled and not self.settings.hpn.enabled:   # HPN answers come from its knowledge
            await asyncio.to_thread(self._load_rag)
        if self.settings.avatar.enabled and not self.vram_ok("avatar"):
            self.warnings.append("avatar disabled for lack of VRAM")
        elif self.settings.avatar.enabled:
            from app.avatar.client import AvatarClient

            t0 = time.perf_counter()
            self.avatar = AvatarClient(self.settings)
            self.avatars = [self.avatar]
            try:
                await self.avatar.start()
                # one more connection per further conversation slot (the worker serves --streams of them)
                for slot in range(1, max(1, self.settings.conversations)):
                    link = AvatarClient(self.settings, slot=slot)
                    await link.start(60.0)
                    self.avatars.append(link)
                self.load_report["avatar_s"] = round(time.perf_counter() - t0, 2)
            except Exception as exc:  # noqa: BLE001 - voice still works without the avatar
                logger.error(f"avatar worker unavailable: {exc!r}")
                # in /health and the page too: only the log said why there was no avatar (e.g. the
                # profile's avatar_id not built yet, logs/avatar-worker.log)
                self.warnings.append(f"avatar unavailable ({type(exc).__name__}); see logs/avatar-worker.log")
        self.load_report["vram_used_mib_after_load"] = gpu_snapshot().get("vram_used_mib", -1)
        logger.info(f"services ready: {self.load_report}")

    def vram_ok(self, what: str) -> bool:
        """True if loading ``what`` still leaves the configured VRAM reserve free."""
        free = vram_free_gib()
        if free is None:
            return True
        need = VRAM_NEED_MIB.get(what, 1000) / 1024
        ok = free - need >= self.settings.vram_min_free_gib
        self.load_report[f"vram_free_gib_before_{what}"] = round(free, 2)
        if not ok:
            msg = (f"not enough free VRAM for {what}: {free:.2f} GiB free, needs ~{need:.2f} GiB "
                   f"+ {self.settings.vram_min_free_gib} GiB reserve")
            logger.error(msg)
            self.warnings.append(msg)
        return ok

    def _load_voice(self) -> None:
        from app.stt.whisper import WhisperSTT
        from app.tts.factory import create_tts

        if not self.vram_ok(f"stt:{self.settings.stt.model}"):
            self.warnings.append("voice disabled (text chat only) — close GPU-heavy apps or use profile 'lite'")
            return
        t0 = time.perf_counter()
        self.stt = WhisperSTT(self.settings.stt)
        self.load_report["stt_load_s"] = round(time.perf_counter() - t0, 2)
        self.load_report["stt_warmup_s"] = self.stt.warmup()
        engine = self.settings.tts.engine
        if engine != "kokoro" and not self.vram_ok(f"tts:{engine}"):
            self.warnings.append("clone voice skipped for lack of VRAM — using the stock voice")
            self.settings.tts.engine = "kokoro"
        t0 = time.perf_counter()
        try:
            self.tts = create_tts(self.settings)
        except Exception as exc:  # noqa: BLE001 - e.g. clone worker failed: fall back to the stock voice
            if self.settings.tts.engine == "kokoro":
                raise
            logger.error(f"clone TTS failed ({exc!r}); falling back to Kokoro")
            self.warnings.append(f"clone voice failed: {exc}")
            self.settings.tts.engine = "kokoro"
            self.tts = create_tts(self.settings)
        self.load_report["tts_load_s"] = round(time.perf_counter() - t0, 2)
        self._apply_persona_voice()
        self.load_report["tts_warmup_s"] = self.tts.warmup()

    def _apply_persona_voice(self) -> None:
        """The voice chosen in the page (persona.json) at start-up. One that is not here any more (e.g. not
        in this PC's model cache) made set_voice raise and the backend fail to start: the configured
        voice is used instead, with a warning."""
        voice = self.persona.voice
        if not hasattr(self.tts, "set_voice") or voice in ("", self.tts.voice):
            return
        try:
            if not self._voice_available(voice):
                raise FileNotFoundError("not in the local model cache")
            self.tts.set_voice(voice)
        except Exception as exc:  # noqa: BLE001
            msg = f"persona voice {voice!r} unavailable ({exc}): using the configured voice {self.tts.voice!r}"
            logger.warning(msg)
            self.warnings.append(msg)
            self.persona = replace(self.persona, voice=self.tts.voice)   # (persona.json is left as it is)

    def _load_hpn(self) -> None:
        """HPN mode without its knowledge would leave the LLM to make answers up: refused at start-up."""
        from app.hpn.knowledge import Knowledge

        t0 = time.perf_counter()
        folder = self.settings.repo_path(self.settings.hpn.knowledge_dir)
        try:
            self.hpn_kb = Knowledge.load(folder)
        except Exception as exc:
            raise RuntimeError(f"HPN mode: the knowledge in {folder} could not be read ({exc!r}); run "
                               "scripts/hpn_refresh.py") from exc
        kb = self.hpn_kb
        self.load_report["hpn_s"] = round(time.perf_counter() - t0, 3)
        logger.info(f"HPN knowledge: {len(kb.topics)} topics, {len(kb.offerings)} packages and add-ons, "
                    f"{len(kb.slots)} nights at {len(kb.venues)} venues; {kb.sources}")

    def hpn_status(self) -> dict | None:
        kb = self.hpn_kb
        if kb is None:
            return None
        return {"topics": len(kb.topics), "offerings": len(kb.offerings), "nights": len(kb.slots),
                "venues": len(kb.venues), "excluded_categories": list(kb.excluded_categories), **kb.sources}

    def _load_rag(self) -> None:
        index = self.settings.repo_path(self.settings.rag.index_dir)
        if not index.exists():
            logger.warning(f"RAG enabled but no index at {index}; run scripts/ingest.py")
            return
        from app.rag.retriever import Retriever

        t0 = time.perf_counter()
        try:
            self.rag = Retriever(self.settings.rag, index)
            self.rag.retrieve("warm up")
        except Exception as exc:  # noqa: BLE001 - conversation works without RAG
            logger.error(f"RAG unavailable: {exc!r}")
            self.rag = None
        self.load_report["rag_s"] = round(time.perf_counter() - t0, 2)

    async def _warm_llm(self) -> None:
        """Run one tiny request with the real system prompt so its KV prefix is cached: the persona's name,
        personality and business as a conversation's (session.py; a customer without a memory of earlier
        conversations) - with the configured name alone the cached prefix ended after a few words."""
        t0 = time.perf_counter()
        if self.hpn_kb is not None:
            # HPN mode: the LLM only re-matches unclear questions (app/hpn/match.py): its long list is the
            # prompt to cache
            from app.hpn import match
            from app.hpn.engine import Engine

            try:
                engine = Engine(self.hpn_kb, self.persona.name)
                await self.llm.complete(match.prompt(match.candidates(engine), "hello"), max_tokens=4, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LLM warm-up failed: {exc!r}")
            self.load_report["llm_warmup_s"] = round(time.perf_counter() - t0, 2)
            return
        p = self.persona
        prompt = system_prompt(p.name, knowledge_enabled=self.settings.rag.enabled,
                               behavior_tag=self.settings.llm.behavior_tag, personality=p.personality,
                               company=p.company, lead=lead_rules(p.company, p.offer, p.next_step) if p.lead_gen else "")
        try:
            await self.llm.complete([{"role": "system", "content": prompt}, {"role": "user", "content": "Hi"}], max_tokens=4)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM warm-up failed: {exc!r}")
        self.load_report["llm_warmup_s"] = round(time.perf_counter() - t0, 2)

    # ---- avatar hooks; no-ops when no avatar worker is connected ----
    # Each takes the conversation's own avatar connection (``link``: Session.avatar_link, one per conversation
    # slot); without one, the first (one conversation at a time, and the tests).
    def _link(self, link):
        return link if link is not None else self.avatar

    def _live(self, link):
        link = self._link(link)
        return link if link is not None and link.connected else None

    def avatar_sink_for(self, session, turn):
        link = self._link(getattr(session, "avatar_link", None))

        async def sink(turn_id: int, seq: int, pcm, pts_ms: int) -> None:
            if seq != 1:
                # Later clauses play after the audio already queued in the browser; holding them until
                # their first frame existed made gaps of up to 1.5 s between sentences (seen in user
                # testing). Their frames carry pts and are drawn by the audio clock (late ones skipped).
                link.send_audio(turn_id, seq, pcm, pts_ms)
                return
            ms = await link.speak(turn_id, seq, pcm, pts_ms)
            turn.mark("avatar_first_frame")
            turn.avatar_first_ms = ms
        return sink

    async def avatar_cancel(self, turn_id: int, link=None) -> None:
        if (live := self._live(link)) is not None:
            live.send_json({"type": "cancel", "turn_id": turn_id})

    def avatar_reset(self, link=None) -> None:
        """Session closed (e.g. the tab was closed while the avatar spoke): back to the idle stream."""
        link = self._link(link)
        if link is not None:
            link.codec = "jpeg"                   # the worker goes back to JPEG on reset
            link.start_hold_ms = 0
            link.kbps = None
        if (live := self._live(link)) is not None:
            live.send_json({"type": "reset"})

    def avatar_state(self, state: str, link=None) -> None:
        if (live := self._live(link)) is not None:
            live.send_json({"type": "state", "state": state})

    def avatar_behavior(self, behavior: dict, link=None) -> None:
        if (live := self._live(link)) is not None:
            live.send_json({"type": "behavior", "expression": behavior.get("expression"),
                            "emotion": behavior.get("emotion"), "gesture": behavior.get("gesture"),
                            "intensity": behavior.get("intensity"), "source": behavior.get("source")})

    def avatar_speech_end(self, turn_id: int, total_ms: float, link=None) -> None:
        """All of the reply's audio has gone to the worker: it renders the closing tail after it."""
        if (live := self._live(link)) is not None:
            live.send_json({"type": "speech_end", "turn_id": turn_id, "total_ms": round(total_ms)})

    def avatar_codec(self, codec: str, link=None) -> None:
        """The browser's video codec: "h264" (WebCodecs) or "jpeg" (avatar/musetalk/h264.py)."""
        link = self._link(link)
        if link is not None:
            link.codec = codec
        if (live := self._live(link)) is not None:
            live.send_json({"type": "codec", "codec": codec})

    def avatar_idle_keyframe(self, link=None) -> None:
        """Idle H.264 frames were dropped (slow link): the worker starts the idle stream again from a keyframe."""
        if (live := self._live(link)) is not None:
            live.send_json({"type": "idle_keyframe"})

    def avatar_bitrate(self, speech_kbps: int, idle_kbps: int, link=None) -> None:
        """H.264 bitrates for the current connection (session._adapt_video); remembered for a worker restart."""
        link = self._link(link)
        if link is not None:
            link.kbps = (speech_kbps, idle_kbps)
        if (live := self._live(link)) is not None:
            live.send_json({"type": "bitrate", "speech": speech_kbps, "idle": idle_kbps})

    def avatar_speech_keyframe(self, link=None) -> None:
        """The reply's video was dropped (slow link, hidden page): the worker continues it from a keyframe."""
        if (live := self._live(link)) is not None:
            live.send_json({"type": "speech_keyframe"})

    def avatar_start_hold(self, ms: int, link=None) -> None:
        """The page holds each reply's first audio ``ms`` longer (lip-synced frames arrived late over the
        network): the worker starts the reply's frames as much later in the loop."""
        link = self._link(link)
        if link is not None:
            link.start_hold_ms = ms
        if (live := self._live(link)) is not None:
            live.send_json({"type": "start_hold", "ms": ms})

    def avatar_start_gap(self, gap: float, link=None) -> None:
        """How many loop frames the first lip-synced frame was ahead of the idle frame on screen (the
        page measures it): the worker starts the next reply that much closer (worker.start_bias)."""
        if (live := self._live(link)) is not None:
            live.send_json({"type": "start_gap", "gap": gap})

    def avatar_playback(self, turn_id: int, event: str, played_ms: float = 0.0, link=None) -> None:
        if (live := self._live(link)) is not None:
            live.send_json({"type": "playback", "turn_id": turn_id, "event": event, "played_ms": round(played_ms)})

    async def stop(self) -> None:
        await self.drain_background()               # (needs the LLM server: before it stops)
        for link in reversed(self.avatars or ([self.avatar] if self.avatar is not None else [])):
            await link.stop()                       # the first (it runs the worker process) last
        if self.tts is not None and hasattr(self.tts, "close"):
            self.tts.close()
        await self.llm.aclose()
        if self.settings.llm.manage_server:
            await asyncio.to_thread(self.llm_server.stop)
