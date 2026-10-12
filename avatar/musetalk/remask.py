"""Recompute the blend masks of a built avatar with engine.contour_safe_mask (no new build).

Avatars built before 2026-10-12 used MuseTalk's "jaw" mask, which reached past the jawline: while the avatar
talked, a grey ghost outline showed along the jaw (owner: "her right cheek has an issue"). This rewrites only the
masks of every loop and gesture variant - frames, face boxes and latents stay - each file replaced atomically.
Restart the backend afterwards (the worker loads the masks at start).

    avatar/musetalk/.venv/bin/python avatar/musetalk/remask.py --id dev_girl_hd
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger

from common import AVATARS, WEIGHTS, chdir
from engine import contour_safe_mask


def remask_file(path, fp, pool: ThreadPoolExecutor) -> int:
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    def one(k: int):
        frame = cv2.imdecode(np.frombuffer(data["frames_jpg"][k], np.uint8), cv2.IMREAD_COLOR)
        return contour_safe_mask(frame, data["coords"][k], fp)

    # a ping-pong loop stores each frame twice (forward and backward): its mask is computed once
    unique: dict[bytes, int] = {}
    for k, jpg in enumerate(data["frames_jpg"]):
        unique.setdefault(jpg, k)
    results = dict(zip(unique.values(), pool.map(one, unique.values()), strict=True))
    masks, boxes = [], []
    for jpg in data["frames_jpg"]:
        mask, box = results[unique[jpg]]
        masks.append(cv2.imencode(".png", mask)[1].tobytes())
        boxes.append(tuple(int(v) for v in box))
    data["masks_png"], data["mask_boxes"] = masks, boxes
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(data, fh)
    os.replace(tmp, path)
    return len(unique)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="avatar id (folder under MODEL_HOME/avatars)")
    args = ap.parse_args()
    folder = AVATARS / args.id / "loops"
    files = sorted(folder.glob("*/loop.pkl")) + sorted(folder.glob("*/delta.pkl"))
    if not files:
        raise SystemExit(f"no loops in {folder}")
    with chdir(WEIGHTS):
        from musetalk.utils.face_parsing import FaceParsing

        fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    t0, total = time.perf_counter(), 0
    # the parsing runs on the GPU one frame at a time; cropping, morphology, blur and PNG run in threads
    with ThreadPoolExecutor(max_workers=6) as pool:
        for path in files:
            t1 = time.perf_counter()
            n = remask_file(path, fp, pool)
            total += n
            logger.info(f"{path.parent.name}: {n} masks in {time.perf_counter() - t1:.1f}s")
    logger.info(f"avatar {args.id}: {total} masks in {time.perf_counter() - t0:.0f}s - restart the backend to use them")


if __name__ == "__main__":
    main()
