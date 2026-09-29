"""Can the MuseTalk VAE decode get cheaper? Compares per-frame decode time and pixel error of:
  full decode · channels_last · partial decode of the lower latent rows (+ channels_last).
MuseTalk's blend mask only covers the lower half of the 256x256 face crop, so only the rows
below the mask boundary (plus a margin) matter.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from common import AVATARS
from engine import Loop, MuseTalkEngine


def decode(vae, latents, rows_from=0):
    lat = latents[:, :, rows_from:, :] / vae.scaling_factor
    img = vae.vae.decode(lat.to(vae.vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return (img.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype("uint8")[..., ::-1]


def bench(fn, reps=6):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000 / reps


def main() -> None:
    e = MuseTalkEngine(fps=20)
    loop = Loop.load(AVATARS / "dev_girl" / "loops" / "friendly")
    chunks = e.audio_features((np.random.default_rng(0).standard_normal(24000 * 2) * 3000).astype(np.int16))
    b = 4
    with torch.no_grad():
        latent = loop.latents[:b].to(e.device, e.dtype)
        pred = e.unet(latent, e.timesteps, encoder_hidden_states=e.pe(chunks[:b].to(e.device, e.dtype))).sample
        full, t_full = bench(lambda: decode(e.vae, pred))
        results = {"full": (t_full / b, 0.0)}
        for rows in (10, 12, 14):
            part, t_p = bench(lambda: decode(e.vae, pred, rows))
            y0 = rows * 8
            # compare only the area that gets blended (lower half of the crop: y >= 128)
            err = float(np.abs(part[:, 128 - y0:].astype(int) - full[:, 128:].astype(int)).mean())
            results[f"partial_rows{rows}"] = (t_p / b, err)
            if rows == 12:
                cv2.imwrite(str(AVATARS / "dev_girl" / "vae_compare.jpg"),
                            np.hstack([full[0][128:], part[0][128 - y0:], (np.abs(part[0][128 - y0:].astype(int) - full[0][128:].astype(int)) * 8).clip(0, 255).astype(np.uint8)]))
    for k, (ms, err) in results.items():
        print(f"{k:22} {ms:6.1f} ms/frame   mean abs pixel diff in blended area: {err:.2f}")


if __name__ == "__main__":
    main()
