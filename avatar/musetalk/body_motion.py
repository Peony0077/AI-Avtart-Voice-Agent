"""Breathing and a slight sway of the shoulders and body (avatar build post-process).

The face rig (LivePortrait) animates the head only; everything below the chin stayed frozen (owner:
"her shoulders and body don't move"). This moves the image below the chin: the shoulders rise a
little with each breath (7 breaths per 30 s loop, ~4.3 s each) and the body sways slowly from side
to side. The motion starts just below the face box and eases in over RAMP_PX, so the face, the neck
under the chin and the lip-sync area (which only ever changes pixels inside the face box) are
untouched, and lip-synced frames composed on these loop frames move with them. Periodic over the
loop, like the rest of its motion; keyed to the loop frame index, so gesture variants match their
base loop frame for frame.

Applied to the stored frames of a built avatar (no runtime cost); the original frames are kept as
*.orig, so re-running with other settings starts from them instead of moving the body twice:

    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\body_motion.py --id dev_girl_v2
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
from loguru import logger

BREATHS_PER_LOOP = 7      # 30 s loop -> one breath every ~4.3 s (~14 a minute, a calm rate)
BREATH_PX = 2.5           # shoulders rise this much on an in-breath (512 px frame)
SWAY_PX = 1.4             # slow side-to-side sway
RAMP_PX = 45              # the motion eases in over this many rows below the face box
GAP_PX = 2                # rows between the face box and the start of the motion


def offsets(i: int, n: int, breath_px: float = BREATH_PX, sway_px: float = SWAY_PX) -> tuple[float, float]:
    """(dx, dy) of the body at loop frame i of n: dy < 0 = up (in-breath). Periodic over n frames."""
    phase = 2 * math.pi * (i % n) / n
    breath = (1 - math.cos(BREATHS_PER_LOOP * phase)) / 2                       # 0 (out) .. 1 (in)
    sway = 0.6 * math.sin(3 * phase + 0.7) + 0.4 * math.sin(4 * phase + 2.1)     # 10 s and 7.5 s periods
    return sway_px * sway, -breath_px * breath


def move_body(frame: np.ndarray, face_bottom: int, dx: float, dy: float) -> np.ndarray:
    """The image below ``face_bottom`` shifted by (dx, dy), easing in from 0 at the face box."""
    import cv2

    h, w = frame.shape[:2]
    rows = np.arange(h, dtype=np.float32)
    weight = np.clip((rows - (face_bottom + GAP_PX)) / RAMP_PX, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    map_y = np.repeat((rows - dy * weight)[:, None], w, axis=1)
    map_x = np.arange(w, dtype=np.float32)[None, :] - (dx * weight)[:, None]
    return cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _apply_to_pickle(path: Path, key_idx: str | None, n: int, quality: int) -> int:
    import cv2

    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    with open(orig, "rb") as fh:
        d = pickle.load(fh)
    positions = d[key_idx] if key_idx else list(range(len(d["frames_jpg"])))
    out = []
    for jpg, pos, box in zip(d["frames_jpg"], positions, d["coords"], strict=True):
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        dx, dy = offsets(pos, n)
        moved = move_body(frame, int(box[3]), dx, dy)
        out.append(cv2.imencode(".jpg", moved, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
    d["frames_jpg"] = out
    with open(path, "wb") as fh:
        pickle.dump(d, fh)
    return len(out)


def apply_to_avatar(folder: Path, quality: int = 90) -> dict:
    """Move the body in every loop and gesture variant of a built seamless avatar (idempotent)."""
    meta = json.loads((folder / "avatar.json").read_text(encoding="utf-8"))
    if meta.get("pingpong", True):
        raise SystemExit("body motion needs a seamless (periodic) avatar; this one plays ping-pong")
    n = meta["frames_per_loop"]
    t0 = time.perf_counter()
    frames = 0
    for sub in sorted((folder / "loops").iterdir()):
        if (sub / "loop.pkl").exists():
            frames += _apply_to_pickle(sub / "loop.pkl", None, n, quality)
        elif (sub / "delta.pkl").exists():
            frames += _apply_to_pickle(sub / "delta.pkl", "frame_idx", n, quality)
    meta["body_motion"] = {"breaths_per_loop": BREATHS_PER_LOOP, "breath_px": BREATH_PX, "sway_px": SWAY_PX,
                           "ramp_px": RAMP_PX, "frames": frames, "s": round(time.perf_counter() - t0, 1)}
    (folder / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"body motion applied to {frames} frames in {meta['body_motion']['s']} s")
    return meta["body_motion"]


if __name__ == "__main__":
    from common import AVATARS

    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--jpeg", type=int, default=90)
    args = ap.parse_args()
    apply_to_avatar(AVATARS / args.id, args.jpeg)
