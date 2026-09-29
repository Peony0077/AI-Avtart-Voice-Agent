"""Cloned-voice TTS: client for the XTTS-v2 worker process (voice/xtts, separate venv).

Implements the same interface as KokoroTTS plus ``stream()`` so SpeechOutput can play the
first audio chunk of a clause before the whole clause is synthesised. No word timestamps:
heard-text accounting falls back to proportional estimates.
"""
from __future__ import annotations

import itertools
import json
import struct
import subprocess
import time
from collections.abc import Iterator

import numpy as np
from loguru import logger
from websockets.sync.client import connect

from app.core.config import Settings
from app.core.procjob import JOB, free_port
from app.tts.base import TTSResult

CHUNK = struct.Struct("<IIB")


class CloneTTS:
    name = "clone"
    sample_rate = 24000

    def __init__(self, settings: Settings, port: int = 8766) -> None:
        cfg = settings.tts
        if not cfg.clone_reference_wav:
            raise ValueError("tts.clone_reference_wav is required for the clone engine")
        self.url = f"ws://127.0.0.1:{port}"
        py = settings.repo_path("voice/xtts/.venv/Scripts/python.exe")
        script = settings.repo_path("voice/xtts/worker.py")
        ref = settings.repo_path(cfg.clone_reference_wav)
        log = (settings.log_dir / "xtts-worker.log").open("ab")
        t0 = time.perf_counter()
        free_port(port, "worker.py")
        self.proc = subprocess.Popen([str(py), str(script), "--ref", str(ref), "--port", str(port)],
                                     cwd=script.parent, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        JOB.adopt(self.proc)
        self.ws = self._connect(240)
        self._ids = itertools.count(1)
        self.load_s = round(time.perf_counter() - t0, 2)
        logger.info(f"clone TTS worker ready in {self.load_s}s")

    def _connect(self, timeout_s: float):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("XTTS worker exited; see logs/xtts-worker.log")
            try:
                return connect(self.url, max_size=None, open_timeout=2)
            except OSError:
                time.sleep(1.0)
        raise TimeoutError("XTTS worker did not start")

    def stream(self, text: str) -> Iterator[np.ndarray]:
        req = next(self._ids)
        self.ws.send(json.dumps({"type": "synth", "id": req, "text": text}))
        while True:
            msg = self.ws.recv()
            if isinstance(msg, bytes):
                rid, _seq, last = CHUNK.unpack_from(msg)
                if rid != req:
                    continue
                if last:
                    continue
                yield np.frombuffer(msg[CHUNK.size:], np.int16).astype(np.float32) / 32768.0
            else:
                data = json.loads(msg)
                if data.get("type") == "done" and data.get("id") == req:
                    self.last_stats = data
                    return

    def synthesize(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        parts = list(self.stream(text))
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return TTSResult(audio=audio, sample_rate=self.sample_rate, words=[],
                         synth_ms=round((time.perf_counter() - t0) * 1000, 1))

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.synthesize("Hello, I'm ready.")
        return round(time.perf_counter() - t0, 2)

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
