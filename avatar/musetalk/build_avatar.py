"""Build a talking avatar from ONE photo (offline, once per photo).

    photo ──LivePortrait face rig──► 5 seamless 30 s loops (neutral, friendly, happy, concerned, thinking)
          ──LivePortrait face rig──► head-gesture windows (nods, tilt, shake) as small loop deltas
          ──MuseTalk preparation──► face boxes, VAE latents, blend masks (cached)

Output: E:/AI-models/avatars/<avatar_id>/{avatar.json, preview/*.jpg, loops/<name>/...}

Usage (avatar venv):
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\build_avatar.py --photo me.jpg --id me
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\build_avatar.py --id me --gestures-only   # add gestures
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\build_avatar.py --id me --update friendly,happy
        # re-render some expression loops (after changing motion.EXPRESSIONS) and their gesture variants
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
from motion import (
    EXPRESSIONS,
    GESTURE_GAP_FRAMES,
    GESTURE_GAP_FRAMES_LONG,
    GESTURE_VARIANTS,
    gesture_params,
    gesture_windows,
    idle_curves,
    loop_params,
)


def variant_name(expression: str, gesture: str) -> str:
    return f"{expression}__{gesture}"


def render_frames(photo: Path, fps: int, seconds: float, names: list[str], size: int,
                  gestures: dict[str, tuple[str, ...]], pingpong: bool = False) -> tuple[dict, dict, dict]:
    """Base loops (all frames) and gesture variants (only the frames inside gesture windows)."""
    from face_rig import FaceRig, load_photo

    def bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    rig = FaceRig()
    src = rig.load_source(load_photo(photo, size))
    curves = idle_curves(int(seconds * fps), fps, periodic=not pingpong)   # same seed: variants match the loops
    loops, variants, timings = {}, {}, {"rig_load_s": round(time.perf_counter() - t0, 2)}
    rendered = 0
    for name in names:
        t1 = time.perf_counter()
        loops[name] = [bgr(rig.render(src, p)) for p in loop_params(curves, name)]
        rendered += len(loops[name])
        timings[f"render_{name}_s"] = round(time.perf_counter() - t1, 2)
        logger.info(f"rendered loop {name}: {len(loops[name])} frames in {timings[f'render_{name}_s']}s")
    for gesture, expressions in gestures.items():
        for expression in expressions:
            t1 = time.perf_counter()
            params = gesture_params(curves, expression, gesture, fps, gap(pingpong))
            variants[(expression, gesture)] = {f: bgr(rig.render(src, p)) for f, p in params.items()}
            rendered += len(params)
            logger.info(f"rendered {variant_name(expression, gesture)}: {len(params)} frames "
                        f"in {time.perf_counter() - t1:.1f}s")
    timings["render_s"] = round(time.perf_counter() - t0, 1)
    timings["rendered_frames"] = rendered
    timings["vram_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    del rig, src
    gc.collect()
    torch.cuda.empty_cache()
    return loops, variants, timings


def gap(pingpong: bool) -> int:
    return GESTURE_GAP_FRAMES if pingpong else GESTURE_GAP_FRAMES_LONG


def gesture_meta(gestures: dict[str, tuple[str, ...]], n_forward: int, fps: int, pingpong: bool) -> dict:
    return {g: {"expressions": list(ex), "windows": gesture_windows(n_forward, fps, g, gap(pingpong))}
            for g, ex in gestures.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=30.0, help="length of each (seamless) loop")
    ap.add_argument("--pingpong", action="store_true", help="old style: short loops played forward and backward")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--jpeg", type=int, default=90, help="JPEG quality of the stored loop frames (= the idle video); "
                    "the same as the speech frames (avatar.jpeg_quality), so the picture does not change when she stops talking")
    ap.add_argument("--loops", default=",".join(EXPRESSIONS))
    ap.add_argument("--gestures", default=",".join(GESTURE_VARIANTS), help="head gestures to render ('' = none)")
    ap.add_argument("--gestures-only", action="store_true",
                    help="add gesture variants to an existing avatar (keeps its loops and existing variants)")
    ap.add_argument("--force", action="store_true", help="with --gestures-only: re-render existing variants too")
    ap.add_argument("--update", default="", help="re-render these loops of an existing avatar (and their gestures)")
    args = ap.parse_args()

    out = AVATARS / args.id
    meta: dict = {}
    update = [n.strip() for n in args.update.split(",") if n.strip()]
    if update or args.gestures_only:
        meta = json.loads((out / "avatar.json").read_text(encoding="utf-8"))
        args.photo = Path(meta["photo"])
        args.fps, args.size = meta["fps"], meta["size"]
        args.pingpong = meta.get("pingpong", True)
        args.seconds = meta["frames_per_loop"] / (2 if args.pingpong else 1) / meta["fps"]
        names: list[str] = [n for n in update if n in meta["loops"]]
        if update and not names:
            ap.error(f"--update: none of {update} is a loop of {args.id} ({meta['loops']})")
    else:
        if args.photo is None:
            ap.error("--photo is required")
        names = [n.strip() for n in args.loops.split(",") if n.strip()]
    available = set(meta.get("loops", [])) | set(names)
    wanted = [g.strip() for g in args.gestures.split(",") if g.strip()]
    gestures = {g: tuple(e for e in GESTURE_VARIANTS[g] if e in available) for g in wanted}
    to_render = gestures
    if update:                                    # the updated loops' variants (their base frames changed)
        to_render = {g: tuple(e for e in ex if e in names) for g, ex in gestures.items()}
    elif args.gestures_only and not args.force:   # only the variants that do not exist yet
        to_render = {g: tuple(e for e in ex if not (out / "loops" / variant_name(e, g) / "delta.pkl").exists())
                     for g, ex in gestures.items()}

    (out / "preview").mkdir(parents=True, exist_ok=True)
    loops, variants, timings = render_frames(args.photo, args.fps, args.seconds, names, args.size, to_render,
                                             args.pingpong)
    for name, frames in loops.items():  # preview: a mid-loop frame
        cv2.imwrite(str(out / "preview" / f"{name}.jpg"), frames[len(frames) // 3])

    from engine import Loop, MuseTalkEngine
    from face_bbox import FaceBoxer

    t0 = time.perf_counter()
    engine = MuseTalkEngine(fps=args.fps)
    boxer = FaceBoxer()
    timings["musetalk_load_s"] = round(time.perf_counter() - t0, 2)
    prepared: dict[str, Loop] = {}
    for name, frames in loops.items():
        t1 = time.perf_counter()
        prepared[name] = engine.prepare_loop(name, frames, boxer, jpeg_quality=args.jpeg, pingpong=args.pingpong)
        prepared[name].save(out / "loops" / name)
        timings[f"prepare_{name}_s"] = round(time.perf_counter() - t1, 2)
    t1 = time.perf_counter()
    for (expression, gesture), frames in variants.items():
        base = prepared.get(expression) or Loop.load(out / "loops" / expression)
        name = variant_name(expression, gesture)
        engine.prepare_delta(base, name, frames, boxer, jpeg_quality=args.jpeg).save(out / "loops" / name)
    timings["prepare_gestures_s"] = round(time.perf_counter() - t1, 1)

    n_forward = int(args.seconds * args.fps)
    if not (args.gestures_only or update):
        meta = {"id": args.id, "photo": str(args.photo), "fps": args.fps, "size": args.size, "loops": names,
                "frames_per_loop": 2 * n_forward if args.pingpong else n_forward, "pingpong": args.pingpong,
                "timings": timings}
    else:
        prefix = "update" if update else "gestures"
        meta.setdefault("timings", {}).update({f"{prefix}_{k}": v for k, v in timings.items()})
    if args.gestures_only or update:
        # merge: adding one gesture must not drop the others from avatar.json
        meta.setdefault("gestures", {}).update(gesture_meta(gestures, n_forward, args.fps, args.pingpong))
    else:
        meta["gestures"] = gesture_meta(gestures, n_forward, args.fps, args.pingpong)
    meta["built"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"avatar {args.id} built: {json.dumps(timings)}")


if __name__ == "__main__":
    main()
