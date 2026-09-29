"""Profile one MuseTalk batch: whisper features, UNet, VAE decode, CPU blend, JPEG — per frame."""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import torch

from common import AVATARS
from engine import Loop, MuseTalkEngine


def timed(fn, reps=5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000 / reps


def main() -> None:
    torch.backends.cudnn.benchmark = "--bench" in sys.argv
    e = MuseTalkEngine(fps=20)
    loop = Loop.load(AVATARS / "dev_girl" / "loops" / "friendly")
    e.warmup(loop)
    chunks = e.audio_features((np.random.default_rng(0).standard_normal(24000 * 3) * 3000).astype(np.int16))
    for b in (4, 8):
        c = chunks[:b]
        idx = list(range(b))
        latent = loop.latents[idx].to(e.device, e.dtype)
        cond = e.pe(c.to(e.device, e.dtype))
        with torch.no_grad():
            pred, unet_ms = timed(lambda: e.unet(latent, e.timesteps, encoder_hidden_states=cond).sample)
            faces, vae_ms = timed(lambda: e.vae.decode_latents(pred))
        t0 = time.perf_counter()
        frames = [e._blend(loop, i, f) for i, f in zip(idx, faces, strict=True)]
        blend_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for f in frames:
            cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_ms = (time.perf_counter() - t0) * 1000
        print(f"batch {b}: per frame  unet {unet_ms / b:.1f} ms | vae {vae_ms / b:.1f} ms | "
              f"blend(cpu) {blend_ms / b:.1f} ms | jpeg {jpg_ms / b:.1f} ms | cudnn.benchmark={torch.backends.cudnn.benchmark}")


if __name__ == "__main__":
    main()
