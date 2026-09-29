"""Process-wide services (models, clients, subprocess supervisors).

Created once in the FastAPI lifespan and shared by all sessions. Heavy models
are loaded here (never per turn) and warmed up before the first session.
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger

from app.core.config import Settings
from app.llm.client import LLMClient
from app.llm.prompt import system_prompt
from app.llm.server import LlamaServerSupervisor


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm_server = LlamaServerSupervisor(settings)
        self.llm = LLMClient(settings.llm)
        self.stt = None      # Phase 2
        self.vad_factory = None
        self.tts = None
        self.rag = None      # Phase 5
        self.avatar = None   # Phase 4
        self.load_report: dict[str, float] = {}
        self.gpu_lock = asyncio.Lock()  # serialises our own GPU bursts (STT/TTS)

    async def start(self) -> None:
        t0 = time.perf_counter()
        if self.settings.llm.manage_server:
            await asyncio.to_thread(self.llm_server.start)
        self.load_report["llm_server_s"] = round(time.perf_counter() - t0, 2)
        await self._warm_llm()

    async def _warm_llm(self) -> None:
        """Run one tiny request with the real system prompt so its KV prefix is cached."""
        t0 = time.perf_counter()
        prompt = system_prompt(self.settings.persona_name, knowledge_enabled=self.settings.rag.enabled)
        try:
            await self.llm.complete([{"role": "system", "content": prompt}, {"role": "user", "content": "Hi"}], max_tokens=4)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM warm-up failed: {exc!r}")
        self.load_report["llm_warmup_s"] = round(time.perf_counter() - t0, 2)
        logger.info(f"services ready: {self.load_report}")

    async def stop(self) -> None:
        await self.llm.aclose()
        if self.settings.llm.manage_server:
            await asyncio.to_thread(self.llm_server.stop)
