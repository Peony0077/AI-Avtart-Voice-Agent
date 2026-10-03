"""Process-wide services (models, clients, subprocess supervisors).

Created once in the FastAPI lifespan and shared by all sessions. Heavy models
are loaded here (never per turn) and warmed up before the first session.
"""
from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from app.core.config import Settings
from app.core.gpu import gpu_snapshot, vram_free_gib
from app.llm.client import LLMClient
from app.llm.prompt import system_prompt
from app.llm.server import LlamaServerSupervisor

# Measured on the RTX 3060 (docs/ARCHITECTURE.md); used to refuse loads that would make Windows
# spill VRAM into system RAM (which silently makes everything ~10x slower).
VRAM_NEED_MIB = {
    "stt:openai/whisper-large-v3-turbo": 2000, "stt:openai/whisper-small.en": 800,
    "tts:kokoro": 700, "tts:clone": 2600, "tts:clone_light": 900, "avatar": 4000,
}


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
        self.avatar = None
        self.load_report: dict[str, float] = {}
        self.warnings: list[str] = []
        self.gpu_lock = asyncio.Lock()  # serialises our own GPU bursts (STT / TTS)

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
        await self._warm_llm()
        if self.settings.voice_enabled:
            await asyncio.to_thread(self._load_voice)
        if self.settings.rag.enabled:
            await asyncio.to_thread(self._load_rag)
        if self.settings.avatar.enabled and not self.vram_ok("avatar"):
            self.warnings.append("avatar disabled for lack of VRAM")
        elif self.settings.avatar.enabled:
            from app.avatar.client import AvatarClient

            t0 = time.perf_counter()
            self.avatar = AvatarClient(self.settings)
            try:
                await self.avatar.start()
                self.load_report["avatar_s"] = round(time.perf_counter() - t0, 2)
            except Exception as exc:  # noqa: BLE001 - voice still works without the avatar
                logger.error(f"avatar worker unavailable: {exc!r}")
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
        self.load_report["tts_warmup_s"] = self.tts.warmup()

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
        """Run one tiny request with the real system prompt so its KV prefix is cached."""
        t0 = time.perf_counter()
        prompt = system_prompt(self.settings.persona_name)
        try:
            await self.llm.complete([{"role": "system", "content": prompt}, {"role": "user", "content": "Hi"}], max_tokens=4)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM warm-up failed: {exc!r}")
        self.load_report["llm_warmup_s"] = round(time.perf_counter() - t0, 2)

    # ---- avatar hooks; no-ops when no avatar worker is connected ----
    def avatar_sink_for(self, session, turn):
        async def sink(turn_id: int, seq: int, pcm, pts_ms: int) -> None:
            if seq != 1:
                # Later clauses play after the audio already queued in the browser; holding them until
                # their first frame existed made gaps of up to 1.5 s between sentences (seen in user
                # testing). Their frames carry pts and are drawn by the audio clock (late ones skipped).
                self.avatar.send_audio(turn_id, seq, pcm, pts_ms)
                return
            ms = await self.avatar.speak(turn_id, seq, pcm, pts_ms)
            turn.mark("avatar_first_frame")
            turn.avatar_first_ms = ms
        return sink

    async def avatar_cancel(self, turn_id: int) -> None:
        if self.avatar_ready:
            self.avatar.send_json({"type": "cancel", "turn_id": turn_id})

    def avatar_reset(self) -> None:
        """Session closed (e.g. the tab was closed while the avatar spoke): back to the idle stream."""
        if self.avatar_ready:
            self.avatar.send_json({"type": "reset"})

    def avatar_state(self, state: str) -> None:
        if self.avatar_ready:
            self.avatar.send_json({"type": "state", "state": state})

    def avatar_behavior(self, behavior: dict) -> None:
        if self.avatar_ready:
            self.avatar.send_json({"type": "behavior", "expression": behavior.get("expression"),
                                   "emotion": behavior.get("emotion"), "gesture": behavior.get("gesture"),
                                   "intensity": behavior.get("intensity"), "source": behavior.get("source")})

    def avatar_playback(self, turn_id: int, event: str, played_ms: float = 0.0) -> None:
        if self.avatar_ready:
            self.avatar.send_json({"type": "playback", "turn_id": turn_id, "event": event,
                                   "played_ms": round(played_ms)})

    async def stop(self) -> None:
        if self.avatar is not None:
            await self.avatar.stop()
        if self.tts is not None and hasattr(self.tts, "close"):
            self.tts.close()
        await self.llm.aclose()
        if self.settings.llm.manage_server:
            await asyncio.to_thread(self.llm_server.stop)
