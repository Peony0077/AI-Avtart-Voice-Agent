"""Supervisor for the llama.cpp ``llama-server`` process (P2).

Runs the prebuilt Windows binary as a child process, waits until it reports
healthy, restarts it if it crashes, and stops it on shutdown. Uses a plain
thread + ``subprocess.Popen`` (works with any asyncio event loop on Windows).
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import httpx
from loguru import logger

from app.core.config import Settings


class LlamaServerSupervisor:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.llm
        self.exe = settings.repo_path(self.cfg.server_exe)
        self.model = settings.model_file(self.cfg.model_path)
        self.log_path: Path = settings.log_dir / "llama-server.log"
        self.port = int(self.cfg.base_url.rsplit(":", 1)[1].split("/")[0])
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self.restarts = 0

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
        log = self.log_path.open("ab")
        logger.info(f"starting llama-server: {' '.join(self.command())}")
        self._proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def start(self, timeout_s: float = 180.0) -> None:
        if self._is_healthy():
            logger.info("llama-server already running (external) — reusing it")
            return
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
                return
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not ready after {timeout_s}s; see {self.log_path}")

    def _is_healthy(self) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _watch(self) -> None:
        while not self._stop.wait(2.0):
            if self._proc is not None and self._proc.poll() is not None:
                self.restarts += 1
                logger.error(f"llama-server died (code {self._proc.returncode}); restart #{self.restarts}")
                try:
                    self._spawn()
                    self.wait_ready(180.0)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"llama-server restart failed: {exc}")

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
