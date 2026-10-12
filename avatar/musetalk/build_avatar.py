"""Build a talking avatar from ONE photo (offline, once per photo).

    photo ──LivePortrait face rig──► 5 seamless 30 s loops (neutral, friendly, happy, concerned, thinking)
          ──body_motion──► breathing and a slight sway below the chin
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
import numpy as np
import torch
from loguru import logger

from common import AVATARS
from motion import (
    EXPRESSIONS,
    GESTURE_GAP_FRAMES,
    GESTURE_GAP_FRAMES_LONG,
    GESTURE_VARIANTS,
    RigParams,
    gesture_delta,
    gesture_params,
    gesture_windows,
    idle_curves,
    loop_params,
)


def variant_name(expression: str, gesture: str) -> str:
    return f"{expression}__{gesture}"


# Rendered frames are kept JPEG-compressed (visually lossless) until their loop is prepared: raw, the
# v3 build (6 loops + 11 000 gesture frames, 0.8 MB each) needed ~9 GB of RAM and was killed on this
# 16 GB PC; compressed ~1.4 GB, and only one loop is decoded at a time.
KEEP_QUALITY = 97


def keep(img_bgr) -> bytes:
    return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, KEEP_QUALITY])[1].tobytes()


def unpack(frame):
    """A kept frame (bytes) as BGR; frames that are already arrays pass through."""
    if isinstance(frame, bytes | bytearray):
        return cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    return frame


def render_frames(photo: Path, fps: int, seconds: float, names: list[str], size: int,
                  gestures: dict[str, tuple[str, ...]], pingpong: bool = False, sr: str | None = None) -> tuple[dict, dict, dict]:
    """Base loops (all frames) and gesture variants (only the frames inside gesture windows). ``sr``: the
    Real-ESRGAN model for the photo and LivePortrait's face crops (upscale.py), None = bicubic."""
    from face_rig import FaceRig, load_photo

    def bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    rig = FaceRig(upscaler=sr)
    src = rig.load_source(load_photo(photo, size, rig.sr))
    curves = idle_curves(int(seconds * fps), fps, periodic=not pingpong)   # same seed: variants match the loops
    loops, variants, timings = {}, {}, {"rig_load_s": round(time.perf_counter() - t0, 2)}
    rendered = 0
    for name in names:
        t1 = time.perf_counter()
        loops[name] = [keep(bgr(rig.render(src, p))) for p in loop_params(curves, name)]
        rendered += len(loops[name])
        timings[f"render_{name}_s"] = round(time.perf_counter() - t1, 2)
        logger.info(f"rendered loop {name}: {len(loops[name])} frames in {timings[f'render_{name}_s']}s")
    for gesture, expressions in gestures.items():
        for expression in expressions:
            t1 = time.perf_counter()
            params = gesture_params(curves, expression, gesture, fps, gap(pingpong))
            variants[(expression, gesture)] = {f: keep(bgr(rig.render(src, p))) for f, p in params.items()}
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


def video_params(expression: str) -> RigParams:
    """Rig parameters for a video frame: relative to the frame itself (its own pose, gaze and blinks
    stay), only the expression's offsets are added."""
    e = EXPRESSIONS[expression]
    gx, gy = e.get("gaze", (0.0, 0.0))
    return RigParams(pitch=e.get("pitch", 0.0), yaw=e.get("yaw", 0.0), roll=e.get("roll", 0.0), eye_open=e.get("eye", 1.0),
                     gaze_x=gx, gaze_y=gy, smile=e["smile"], eyebrow=e["eyebrow"])


def render_video_frames(video: Path, fps: int, seconds: float, start_s: float, names: list[str], size: int,
                        gestures: dict[str, tuple[str, ...]], sr: str | None = None) -> tuple[dict, dict, dict, dict, int]:
    """Loops and gesture variants from a video (video_source.py): each frame through the face rig relative
    to itself, the loop seam cross-faded. Returns (loops, variants, timings, video info, frames in which
    gesture windows may lie - clear of the seam)."""
    from face_bbox import FaceBoxer
    from face_rig import FaceRig
    from video_source import SEAM_FRAMES, close_loop, read_segment

    def bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    boxer = FaceBoxer()
    body, lead_in, info = read_segment(video, fps, start_s, seconds, size, boxer)
    del boxer
    n = len(body)
    n_win = n - SEAM_FRAMES
    windows = {g: gesture_windows(n_win, fps, g, GESTURE_GAP_FRAMES_LONG) for g in gestures}
    deltas = {(g, s): gesture_delta(g, fps, seed=s) for g in gestures for s, _ in windows[g]}
    rig = FaceRig(upscaler=sr)
    timings = {"rig_load_s": round(time.perf_counter() - t0, 2)}
    loops = {name: [] for name in names}
    lead = {name: [] for name in names}
    variants = {(e, g): {} for g, ex in gestures.items() for e in ex}
    src, missed = None, 0
    for i, frame in enumerate(lead_in + body):
        try:
            src = rig.load_source(frame)
        except ValueError:                       # no face found (blur): the previous frame's rig, this image
            if src is None:
                raise
            src = {**src, "img": frame}
            missed += 1
        k = i - len(lead_in)
        for name in names:
            (lead if k < 0 else loops)[name].append(bgr(rig.render(src, video_params(name))))
        if k < 0:
            continue
        for g, expressions in gestures.items():
            for s, e in windows[g]:
                if s <= k < e:
                    d = deltas[(g, s)]
                    for expression in expressions:
                        p = video_params(expression)
                        p.pitch += float(d["pitch"][k - s])
                        p.yaw += float(d["yaw"][k - s])
                        p.roll += float(d["roll"][k - s])
                        variants[(expression, g)][k] = bgr(rig.render(src, p))
        if k % 100 == 0:
            logger.info(f"video frame {k}/{n} rendered ({time.perf_counter() - t0:.0f} s)")
    for name in names:
        loops[name] = close_loop(loops[name], lead[name])
    timings.update({"render_s": round(time.perf_counter() - t0, 1), "video_frames": n, "frames_without_face": missed,
                    "rendered_frames": n * len(names) + sum(len(v) for v in variants.values())})
    del rig, src
    gc.collect()
    torch.cuda.empty_cache()
    return loops, variants, timings, info, n_win


def gap(pingpong: bool) -> int:
    return GESTURE_GAP_FRAMES if pingpong else GESTURE_GAP_FRAMES_LONG


def gesture_meta(gestures: dict[str, tuple[str, ...]], n_forward: int, fps: int, pingpong: bool) -> dict:
    return {g: {"expressions": list(ex), "windows": gesture_windows(n_forward, fps, g, gap(pingpong))}
            for g, ex in gestures.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path)
    ap.add_argument("--video", type=Path, help="build from a video of a real person (video_source.py) instead of a photo")
    ap.add_argument("--start-s", type=float, default=1.0, help="with --video: where the loop segment starts")
    ap.add_argument("--id", required=True)
    # 25 fps: MuseTalk's training frame rate (the first avatars used 20); 1024x1024 with Real-ESRGAN (owner,
    # 2026-10-10: "the resolution is not good" - 512x512 was enlarged 2-3x on the page)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=30.0, help="length of each (seamless) loop")
    ap.add_argument("--pingpong", action="store_true", help="old style: short loops played forward and backward")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--sr", default="x2plus", help="Real-ESRGAN model for the photo and the face crops (upscale.py; "
                    "used when they would be enlarged); '' = bicubic")
    ap.add_argument("--jpeg", type=int, default=92, help="JPEG quality of the stored loop frames (= the idle video); "
                    "the same as the speech frames (avatar.jpeg_quality), so the picture does not change when she stops talking")
    ap.add_argument("--loops", default=",".join(EXPRESSIONS))
    ap.add_argument("--gestures", default=",".join(GESTURE_VARIANTS), help="head gestures to render ('' = none)")
    ap.add_argument("--gestures-only", action="store_true",
                    help="add gesture variants to an existing avatar (keeps its loops and existing variants)")
    ap.add_argument("--force", action="store_true", help="with --gestures-only: re-render existing variants too")
    ap.add_argument("--update", default="", help="re-render these loops of an existing avatar (and their gestures)")
    ap.add_argument("--no-body-motion", action="store_true", help="keep the shoulders still (body_motion.py)")
    args = ap.parse_args()

    out = AVATARS / args.id
    meta: dict = {}
    update = [n.strip() for n in args.update.split(",") if n.strip()]
    if update or args.gestures_only:
        meta = json.loads((out / "avatar.json").read_text(encoding="utf-8"))
        if meta.get("video"):
            ap.error(f"{args.id} was built from a video: build it again with --video")
        args.photo = Path(meta["photo"])
        args.fps, args.size = meta["fps"], meta["size"]
        args.pingpong = meta.get("pingpong", True)
        args.seconds = meta["frames_per_loop"] / (2 if args.pingpong else 1) / meta["fps"]
        names: list[str] = [n for n in update if n in meta["loops"]]
        if update and not names:
            ap.error(f"--update: none of {update} is a loop of {args.id} ({meta['loops']})")
    else:
        if args.photo is None and args.video is None:
            ap.error("--photo or --video is required")
        if args.video is not None and args.pingpong:
            ap.error("--video makes seamless loops (no --pingpong)")
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
    video_info, n_win = None, None
    if args.video is not None:
        loops, variants, timings, video_info, n_win = render_video_frames(
            args.video, args.fps, args.seconds, args.start_s, names, args.size, to_render, args.sr or None)
    else:
        loops, variants, timings = render_frames(args.photo, args.fps, args.seconds, names, args.size, to_render,
                                                 args.pingpong, args.sr or None)
    for name, frames in loops.items():  # preview: a mid-loop frame
        cv2.imwrite(str(out / "preview" / f"{name}.jpg"), unpack(frames[len(frames) // 3]))

    from engine import Loop, MuseTalkEngine
    from face_bbox import FaceBoxer

    t0 = time.perf_counter()
    engine = MuseTalkEngine(fps=args.fps)
    boxer = FaceBoxer()
    timings["musetalk_load_s"] = round(time.perf_counter() - t0, 2)
    prepared: dict[str, Loop] = {}
    # (the frames are freed while preparing; --gestures-only renders no loop: only a video build needs it)
    n_loop = len(next(iter(loops.values()))) if loops else 0
    for name in list(loops):
        t1 = time.perf_counter()
        frames = [unpack(f) for f in loops.pop(name)]
        prepared[name] = engine.prepare_loop(name, frames, boxer, jpeg_quality=args.jpeg, pingpong=args.pingpong)
        del frames
        prepared[name].save(out / "loops" / name)
        timings[f"prepare_{name}_s"] = round(time.perf_counter() - t1, 2)
    t1 = time.perf_counter()
    for (expression, gesture) in list(variants):
        frames = {f: unpack(img) for f, img in variants.pop((expression, gesture)).items()}
        base = prepared.get(expression) or Loop.load(out / "loops" / expression)
        name = variant_name(expression, gesture)
        engine.prepare_delta(base, name, frames, boxer, jpeg_quality=args.jpeg).save(out / "loops" / name)
        del frames
    timings["prepare_gestures_s"] = round(time.perf_counter() - t1, 1)

    n_forward = int(args.seconds * args.fps) if video_info is None else n_loop
    if not (args.gestures_only or update):
        meta = {"id": args.id, "photo": str(args.photo) if args.photo else None, "fps": args.fps, "size": args.size,
                "motion": 3,     # motion.py v3: lively head (short moves, micro-motion, follows the eyes), stronger faces
                "loops": names, "frames_per_loop": 2 * n_forward if args.pingpong else n_forward,
                "pingpong": args.pingpong, "timings": timings}
        if video_info is not None:
            meta["video"] = video_info
    else:
        prefix = "update" if update else "gestures"
        meta.setdefault("timings", {}).update({f"{prefix}_{k}": v for k, v in timings.items()})
    if args.gestures_only or update:
        # merge: adding one gesture must not drop the others from avatar.json
        meta.setdefault("gestures", {}).update(gesture_meta(gestures, n_forward, args.fps, args.pingpong))
    else:   # video: windows stay clear of the cross-faded seam
        meta["gestures"] = gesture_meta(gestures, n_win or n_forward, args.fps, args.pingpong)
    meta["built"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "avatar.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not args.pingpong and not args.no_body_motion and video_info is None:   # breathing and sway below the chin
        # (a video already has the person's own body motion)
        from body_motion import apply_to_avatar
        apply_to_avatar(out, args.jpeg)
    logger.info(f"avatar {args.id} built: {json.dumps(timings)}")


if __name__ == "__main__":
    main()
