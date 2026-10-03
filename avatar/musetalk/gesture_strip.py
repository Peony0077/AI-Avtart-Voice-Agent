"""Contact sheet of the rendered head gestures (visual check after build_avatar.py).

    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\gesture_strip.py --id dev_girl --out logs\\gestures.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import AVATARS
from engine import Loop, LoopDelta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="dev_girl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=7, help="frames shown per gesture")
    args = ap.parse_args()
    folder = AVATARS / args.id
    meta = json.loads((folder / "avatar.json").read_text(encoding="utf-8"))
    rows = []
    for gesture, gm in meta.get("gestures", {}).items():
        expression = gm["expressions"][0]
        base = Loop.load(folder / "loops" / expression)
        variant = base.with_delta(LoopDelta.load(folder / "loops" / f"{expression}__{gesture}"))
        s, e = gm["windows"][0]
        picks = np.linspace(s - 1, e, args.frames).round().astype(int)
        tiles = []
        for f in picks:
            img = cv2.resize(variant.frame(int(f)), (192, 192))
            cv2.putText(img, f"{gesture} {f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
