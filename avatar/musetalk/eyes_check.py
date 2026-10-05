"""Eye opening of every expression loop compared with the idle face (CPU, no GPU needed).

    .venv\\Scripts\\python eyes_check.py [--avatar ID] [--every 20] [--sheet eyes.jpg]

Replies are spoken with the expression loops (motion.EXPRESSIONS), the idle video is "friendly". When an
expression's eyes are much smaller or larger than the idle face's, her eyes visibly shrink or widen
whenever she starts to talk (owner, 2026-10-05: the "happy" smile narrowed them by 19 %). Per loop: eye
opening = eyelid gap / eye width from the 68 DWPose face landmarks (the model the build uses for face
boxes), median over frames without blinks, and the difference to the idle loop. --sheet writes the
eye region of every loop side by side to look at. Exit code 1 when a loop differs by more than --max.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from common import AVATARS, WEIGHTS
from face_bbox import _dwpose

EYES = ((36, 37, 38, 39, 40, 41), (42, 43, 44, 45, 46, 47))   # corner, upper, upper, corner, lower, lower


def eye_opening(lm: np.ndarray) -> float:
    """Mean over both eyes of (eyelid gaps) / (eye width)."""
    out = []
    for a, b, c, d, e, f in EYES:
        width = np.linalg.norm(lm[a] - lm[d])
        out.append((np.linalg.norm(lm[b] - lm[f]) + np.linalg.norm(lm[c] - lm[e])) / (2 * width))
    return float(np.mean(out))


def measure(session: ort.InferenceSession, frames_jpg: list[bytes], every: int) -> tuple[float, np.ndarray, np.ndarray]:
    values, crop, lm_first = [], None, None
    for k in range(0, len(frames_jpg), every):
        img = cv2.imdecode(np.frombuffer(frames_jpg[k], np.uint8), cv2.IMREAD_COLOR)
        lm = _dwpose(session, img)[23:91]
        values.append(eye_opening(lm))
        if crop is None:
            lm_first, crop = lm, img
    v = np.array(values)
    open_eyes = v[v > 0.75 * np.median(v)]               # without blinks
    return float(np.median(open_eyes)), crop, lm_first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="dev_girl_v3")
    ap.add_argument("--every", type=int, default=20, help="measure every n-th frame")
    ap.add_argument("--idle", default="friendly")
    ap.add_argument("--max", type=float, default=0.10, help="largest allowed difference to the idle loop")
    ap.add_argument("--sheet", type=Path, default=None)
    args = ap.parse_args()
    avatar = args.avatar
    meta = json.loads((AVATARS / avatar / "avatar.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(WEIGHTS / "models/dwpose/dw-ll_ucoco_384.onnx"), providers=["CPUExecutionProvider"])
    results, crops = {}, []
    for name in meta["loops"]:
        with open(AVATARS / avatar / "loops" / name / "loop.pkl", "rb") as fh:
            frames = pickle.load(fh)["frames_jpg"]
        results[name], img, lm = measure(session, frames, args.every)
        y = int(np.mean(lm[36:48, 1]))
        x1, x2 = int(lm[36, 0]) - 25, int(lm[45, 0]) + 25
        crops.append(cv2.putText(cv2.resize(img[y - 35:y + 35, x1:x2], (300, 90)), name, (5, 15),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1))
    idle = results.get(args.idle)
    narrowest = 0.0
    for name, value in results.items():
        diff = value / idle - 1.0 if idle else 0.0
        if name != "thinking":                    # thinking looks away on purpose
            narrowest = max(narrowest, -diff)
        print(f"{name:12s} eye opening {value:.3f}  {diff:+.0%} vs {args.idle}")
    if args.sheet:
        cv2.imwrite(str(args.sheet), np.vstack(crops))
        print(f"eye regions: {args.sheet}")
    # narrower eyes read as squinting the moment she starts to talk; wider ones (neutral = the photo's own
    # eyes, "interested") read as attention
    if narrowest > args.max:
        print(f"an expression's eyes are {narrowest:.0%} smaller than the idle face's (more than {args.max:.0%})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
