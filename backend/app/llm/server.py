"""Supervisor for the llama.cpp ``llama-server`` process (P2).

Runs the prebuilt Windows binary as a child process, waits until it reports
healthy, restarts it if it crashes, and stops it on shutdown. Uses a plain
thread + ``subprocess.Popen`` (works with any asyncio event loop on Windows).
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from loguru import logger

from app.core.config import Settings
from app.core.procjob import JOB, free_port

READY_TIMEOUT_S = 180.0
RESTART_BACKOFF_S = (2.0, 60.0)     # first and longest wait between restarts of a crashing llama-server
HUNG_AFTER_S = 600.0                # a restarted llama-server that never answers /health is restarted again


def port_of(base_url: str) -> int:
    url = urlsplit(base_url)
    return url.port or (443 if url.scheme == "https" else 80)


class LlamaServerSupervisor:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.llm
        self.exe = settings.repo_path(self.cfg.server_exe)
        self.model = settings.model_file(self.cfg.model_path)
        self.log_path: Path = settings.log_dir / "llama-server.log"
        # only for a server of our own: an external one (manage_server: false) may be at any URL
        self.port = port_of(self.cfg.base_url) if self.cfg.manage_server else None
        self._proc: subprocess.Popen | None = None
        self._spawned_at = 0.0
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self.restarts = 0
        self.ready = threading.Event()   # set while llama-server answers /health; cleared when it dies

    def command(self) -> list[str]:
        return [
            str(self.exe),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", str(self.cfg.gpu_layers),
            "-c", str(self.cfg.context),
            "-fa", "on",
            "-ctk", self.cfg.kv_cache_type,
            "-ctv", self.cfg.kv_cache_type,
            "--parallel", "1",
            "--jinja",
            "--no-webui",
            "-lm", "none",  # no mmap: saves ~1.8 GB resident RAM (measured), weights live in VRAM
            "--cache-reuse", "256",  # reuse cached KV chunks after history trimming (KV shifting)
        ]

    def _spawn(self) -> None:
        if not self.exe.exists():
            raise FileNotFoundError(f"llama-server not found: {self.exe}")
        if not self.model.exists():
            raise FileNotFoundError(f"LLM model not found: {self.model}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"starting llama-server: {' '.join(self.command())}")
        with self.log_path.open("ab") as log:      # the child has its own handle: ours leaked one per restart
            self._proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT,
                                          **JOB.popen_kwargs())
        self._spawned_at = time.time()
        JOB.adopt(self._proc)

    def start(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        free_port(self.port, "llama-server")   # a stale llama-server of ours (e.g. from a killed backend)
        self._spawn()
        self.wait_ready(timeout_s)
        self._watchdog = threading.Thread(target=self._watch, name="llama-watchdog", daemon=True)
        self._watchdog.start()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._proc.returncode}; see {self.log_path}")
            if self._is_healthy():
                logger.info("llama-server is ready")
                self.ready.set()
                return
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not ready after {timeout_s}s; see {self.log_path}")

    def _is_healthy(self) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _watch(self) -> None:
        """Restart llama-server when it dies, waiting longer after every failed restart (it used to retry
        every 2 s forever), and keep watching a restart that is slower than READY_TIMEOUT_S: ``ready`` was
        then never set again although the server came up."""
        backoff = delay = RESTART_BACKOFF_S[0]
        while not self._stop.wait(delay):
            proc = self._proc
            if proc is not None and proc.poll() is None:
                delay = RESTART_BACKOFF_S[0]
                if self.ready.is_set():
                    backoff = RESTART_BACKOFF_S[0]
                elif self._is_healthy():
                    logger.info("llama-server is ready (late)")
                    self.ready.set()
                elif time.time() - self._spawned_at > HUNG_AFTER_S:
                    logger.error(f"llama-server not ready after {HUNG_AFTER_S:.0f} s: restarting it")
                    proc.kill()
                    delay = 1.0
                continue
            self.ready.clear()
            self.restarts += 1
            logger.error(f"llama-server died (code {proc.returncode if proc else None}); restart #{self.restarts}")
            if self._stop.is_set():
                return
            try:
                self._spawn()
                self.wait_ready(READY_TIMEOUT_S)
                backoff = delay = RESTART_BACKOFF_S[0]
            except Exception as exc:  # noqa: BLE001 - checked (and restarted) again later
                backoff = delay = min(RESTART_BACKOFF_S[1], backoff * 2)
                logger.error(f"llama-server restart failed: {exc}; next check in {delay:.0f} s")

    async def wait_until_ready(self, timeout_s: float) -> bool:
        """For requests that hit a crashed server: wait until the watchdog has it answering again."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.ready.is_set() and await asyncio.to_thread(self._is_healthy):
                return True
            await asyncio.sleep(0.25)
        return False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.info("llama-server stopped")
