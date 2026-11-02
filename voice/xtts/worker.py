"""Voice-cloning TTS worker (XTTS-v2 via coqui-tts, idiap fork; model license CPML = non-commercial).

Separate venv/process because coqui-tts needs transformers 4.5x and other pins that must not
touch the backend environment.

Protocol (WebSocket, backend = client):
  → {"type":"synth","id":<int>,"text":"..."}           stream one clause
  ← binary: <u32 id><u32 seq><u8 last> + PCM16 mono 24 kHz chunk
  ← {"type":"done","id":..,"first_chunk_ms":..,"total_ms":..,"audio_ms":..}
  → {"type":"voice","wav":"path/to/reference.wav"}      switch the cloned voice

Run: voice\\xtts\\.venv\\Scripts\\python voice\\xtts\\worker.py --ref data\\voices\\me.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import websockets
from loguru import logger

MODEL_DIR = Path(os.environ.get("AVATAR_MODEL_HOME", "E:/AI-models")) / "tts" / "XTTS-v2"
CHUNK = struct.Struct("<IIB")


def _load_audio(audiopath, sampling_rate):
    """soundfile-based replacement for TTS's torchaudio.load (which needs FFmpeg DLLs via torchcodec)."""
    import soundfile as sf
    import torchaudio.functional as AF

    data, sr = sf.read(str(audiopath), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.mean(0, keepdims=True))
    if sr != sampling_rate:
        audio = AF.resample(audio, sr, sampling_rate)
    return audio.clamp(-1, 1)


class XTTS:
    def __init__(self, ref_wav: Path) -> None:
        import TTS.tts.models.xtts as xtts_module
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        xtts_module.load_audio = _load_audio

        t0 = time.perf_counter()
        cfg = XttsConfig()
        cfg.load_json(str(MODEL_DIR / "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=str(MODEL_DIR), use_deepspeed=False)
        self.model.cuda().eval()
        self.set_voice(ref_wav)
        self.load_s = round(time.perf_counter() - t0, 2)
        logger.info(f"XTTS-v2 ready in {self.load_s}s, voice {ref_wav}")

    @torch.inference_mode()
    def set_voice(self, ref_wav: Path) -> None:
        self.latent, self.speaker = self.model.get_conditioning_latents(audio_path=[str(ref_wav)])

    @torch.inference_mode()
    def stream(self, text: str):
        for chunk in self.model.inference_stream(text, "en", self.latent, self.speaker,
                                                 stream_chunk_size=20, enable_text_splitting=False):
            yield chunk.float().cpu().numpy()


async def serve(tts: XTTS, host: str, port: int) -> None:
    lock = asyncio.Lock()

    async def handle(ws) -> None:
        async for msg in ws:
            req = json.loads(msg)
            if req["type"] != "synth":
                continue
            async with lock:
                t0 = time.perf_counter()
                first = None
                samples = 0
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def produce() -> None:
                    # the end marker always follows: an XTTS error (CUDA out of memory, an over-long text)
                    # used to leave this handler - and the backend's recv() - waiting forever
                    try:
                        for audio in tts.stream(req["text"]):
                            loop.call_soon_threadsafe(queue.put_nowait, audio)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                fut = loop.run_in_executor(None, produce)
                seq = 0
                while (audio := await queue.get()) is not None:
                    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    samples += len(pcm)
                    first = first or (time.perf_counter() - t0) * 1000
                    await ws.send(CHUNK.pack(req["id"], seq, 0) + pcm.tobytes())
                    seq += 1
                error = None
                try:
                    await fut
                except Exception as exc:          # CUDA out of memory, an over-long text: this clause fails,
                    error = str(exc)[:200]        # the connection (and every later reply) stays usable
                    logger.exception("XTTS synthesis failed")
                await ws.send(CHUNK.pack(req["id"], seq, 1))
                await ws.send(json.dumps({"type": "done", "id": req["id"], "error": error,
                                          "first_chunk_ms": round(first or 0, 1),
                                          "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                                          "audio_ms": round(samples / 24)}))

    # origins=[None]: only clients without an Origin header (the backend) - a web page open in the
    # owner's browser could otherwise reach ws://127.0.0.1 and use the GPU (as in avatar/musetalk/worker.py)
    async with websockets.serve(handle, host, port, max_size=None, origins=[None]):
        logger.info(f"XTTS worker on ws://{host}:{port}")
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True, help="reference voice WAV (6-15 s, clean speech)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    tts = XTTS(args.ref)
    t0 = time.perf_counter()
    list(tts.stream("Warming up the voice."))
    logger.info(f"warm-up {time.perf_counter() - t0:.2f}s")
    asyncio.run(serve(tts, args.host, args.port))


if __name__ == "__main__":
    main()
