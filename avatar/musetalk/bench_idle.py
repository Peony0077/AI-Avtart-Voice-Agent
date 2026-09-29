"""Is MuseTalk slower right after the GPU was idle (as seen live)? Renders a 2.3 s clip
back-to-back vs after 3 s idle vs after 3 s idle with an in-process keep-warm loop."""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from common import AVATARS
from engine import Loop, MuseTalkEngine


def clip_fps(e, loop, pcm, b=8):
    t0 = time.perf_counter()
    chunks = e.audio_features(pcm)
    n = chunks.shape[0]
    first = None
    for s in range(0, n, b):
        idx, faces = e.faces(loop, chunks[s:s + b], s)
        for i, f in zip(idx, faces, strict=True):
            e.compose(loop, i, f)
        first = first or (time.perf_counter() - t0) * 1000
    total = time.perf_counter() - t0
    return round(n / total, 1), round(first)


def keep_warm(e, loop, stop):
    lat = loop.latents[:2].to(e.device, e.dtype)
    cond = torch.zeros(2, 50, 384, device=e.device, dtype=e.dtype)
    with torch.no_grad():
        while not stop.is_set():
            e.unet(lat, e.timesteps, encoder_hidden_states=e.pe(cond))
            torch.cuda.synchronize()
            time.sleep(0.03)


def main():
    e = MuseTalkEngine(fps=20)
    loop = Loop.load(AVATARS / "dev_girl" / "loops" / "friendly")
    e.warmup(loop)
    pcm = (np.random.default_rng(0).standard_normal(int(24000 * 2.3)) * 3000).astype(np.int16)
    for mode in ("back_to_back", "idle_3s", "idle_3s_keepwarm"):
        res = []
        for _ in range(4):
            if mode != "back_to_back":
                stop = threading.Event()
                th = threading.Thread(target=keep_warm, args=(e, loop, stop)) if mode.endswith("keepwarm") else None
                if th:
                    th.start()
                time.sleep(3.0)
                stop.set()
                if th:
                    th.join()
            res.append(clip_fps(e, loop, pcm))
        print(mode, "fps/first_ms:", res, flush=True)


if __name__ == "__main__":
    main()
