"""Build a talking avatar from ONE photo (offline, once per photo).

    photo ──LivePortrait face rig──► 5 short loops (neutral, friendly, happy, concerned, thinking)
          ──MuseTalk preparation──► face boxes, VAE latents, blend masks (cached)

Output: E:/AI-models/avatars/<avatar_id>/{avatar.json, preview/*.jpg, loops/<name>/...}

Usage (avatar venv):
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\build_avatar.py --photo me.jpg --id me
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import torch
from loguru import logger

from common import AVATARS
from motion import EXPRESSIONS, idle_curves, loop_params


def render_loops(photo: Path, fps: int, seconds: float, names: list[str], size: int) -> tuple[dict, dict]:
    from face_rig import FaceRig, load_photo

    t0 = time.perf_counter()
    rig = FaceRig()
    src = rig.load_source(load_photo(photo, size))
    curves = idle_curves(int(seconds * fps), fps)
    loops, timings = {}, {"rig_load_s": round(time.perf_counter() - t0, 2)}
    for name in names:
        t1 = time.perf_counter()
        frames = [cv2.cvtColor(rig.render(src, p), cv2.COLOR_RGB2BGR) for p in loop_params(curves, name)]
        loops[name] = frames
        timings[f"render_{name}_s"] = round(time.perf_counter() - t1, 2)
        logger.info(f"rendered loop {name}: {len(frames)} frames in {timings[f'render_{name}_s']}s")
    timings["render_ms_per_frame"] = round(1000 * sum(timings[f"render_{n}_s"] for n in names)
                                           / sum(len(f) for f in loops.values()), 1)
    timings["vram_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    del rig, src
    gc.collect()
    torch.cuda.empty_cache()
    return loops, timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=7.0, help="length of each loop before ping-pong")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--loops", default=",".join(EXPRESSIONS))
    args = ap.parse_args()

    out = AVATARS / args.id
    (out / "preview").mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.loops.split(",") if n.strip()]
    loops, timings = render_loops(args.photo, args.fps, args.seconds, names, args.size)
    for name, frames in loops.items():  # preview: first frame + a mid-blink-free frame
        cv2.imwrite(str(out / "preview" / f"{name}.jpg"), frames[len(frames) // 3])

    from engine import MuseTalkEngine
    from face_bbox import FaceBoxer

    t0 = time.perf_counter()
    engine = MuseTalkEngine(fps=args.fps)
    boxer = FaceBoxer()
    timings["musetalk_load_s"] = round(time.perf_counter() - t0, 2)
    for name, frames in loops.items():
        t1 = time.perf_counter()
        engine.prepare_loop(name, frames, boxer).save(out / "loops" / name)
        timings[f"prepare_{name}_s"] = round(time.perf_counter() - t1, 2)
    meta = {"id": args.id, "photo": str(args.photo), "fps": args.fps, "size": args.size,
            "loops": names, "frames_per_loop": 2 * int(args.seconds * args.fps), "timings": timings,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"avatar {args.id} built: {json.dumps(timings)}")


if __name__ == "__main__":
    main()
