"""An avatar from a short video of a real person instead of one photo (avatar build input).

HeyGen builds its custom "video avatars" from ~2 minutes of footage: the real person's own head
motion, blinks and body motion instead of motion generated from a photo. Here: a segment of the video
(30 s by default) becomes the loops. Every frame goes through the face rig *relative to itself* - its
own pose stays, the expression offsets of each loop (smile, eyebrows, eyes; motion.EXPRESSIONS) and the
head-gesture deltas are added, and the lips are closed (the person may talk in the video: during
speech the lip sync draws the mouth anyway). The segment is chosen so that its last frame looks like
its first; the frames just before the segment are cross-faded into its end, so the loop repeats
without a jump.

Recording: front-facing, head and shoulders, even light, plain background, 1-3 minutes, talking
naturally with pauses; the person's consent is required (it is their face and identity).

    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\build_avatar.py --video data\\avatars\\me\\video.mp4 --id me
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEAM_FRAMES = 8          # frames cross-faded at the loop seam (400 ms at 20 fps)
SEAM_SEARCH_S = 3.0      # the loop may end up to this much earlier, where it best matches its start
FACE_HEIGHT_SHARE = 0.56  # framing like the photo avatars: face box height / frame height


def crop_box(face_boxes: list[tuple], frame_w: int, frame_h: int) -> tuple[int, int, int]:
    """Square crop (x0, y0, side) around the median face box, framed like the photo avatars (head and
    shoulders); kept inside the frame where possible."""
    b = np.median(np.array(face_boxes, dtype=np.float32), axis=0)
    face_h = b[3] - b[1]
    side = int(min(face_h / FACE_HEIGHT_SHARE, frame_w, frame_h))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + 0.06 * side
    x0 = int(np.clip(cx - side / 2, 0, frame_w - side))
    y0 = int(np.clip(cy - side / 2, 0, frame_h - side))
    return x0, y0, side


def loop_length(thumbs: np.ndarray, min_len: int, seam: int = SEAM_FRAMES) -> int:
    """Loop length n (min_len <= n <= len(thumbs) - seam) whose frame n best matches frame 0, so the
    seam joins similar poses. ``thumbs``: small grayscale frames [N, h, w]."""
    first = thumbs[0].astype(np.float32)
    last = len(thumbs) - seam
    if last <= min_len:
        return max(1, last)
    errors = [float(np.mean(np.abs(thumbs[n].astype(np.float32) - first))) for n in range(min_len, last + 1)]
    return min_len + int(np.argmin(errors))


def seam_weights(k: int = SEAM_FRAMES) -> np.ndarray:
    """Cross-fade weights for the last k loop frames: how much of the frames *before* the loop start
    they take (eases in, never 0 or 1)."""
    x = np.arange(1, k + 1, dtype=np.float32) / (k + 1)
    return x * x * (3 - 2 * x)


def close_loop(frames: list[np.ndarray], lead_in: list[np.ndarray]) -> list[np.ndarray]:
    """The last len(lead_in) frames blended towards ``lead_in`` (the frames right before the loop's
    first frame), so the loop flows from its end into its start like the video itself."""
    k = len(lead_in)
    out = list(frames)
    for j, w in enumerate(seam_weights(k)):
        a = frames[len(frames) - k + j].astype(np.float32)
        b = lead_in[j].astype(np.float32)
        out[len(frames) - k + j] = np.clip(a * (1 - w) + b * w + 0.5, 0, 255).astype(np.uint8)
    return out


def read_segment(path: Path, fps: int, start_s: float, seconds: float, size: int, boxer) -> tuple[list, list, dict]:
    """Frames (RGB, size x size, framed on the face) of the loop and the SEAM_FRAMES lead-in frames
    just before it, sampled at ``fps``; the loop end is chosen to match its start."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0 = max(start_s, SEAM_FRAMES / fps)                       # room for the lead-in frames
    wanted = [t0 - (SEAM_FRAMES - j) / fps for j in range(SEAM_FRAMES)] + \
             [t0 + k / fps for k in range(int(seconds * fps) + SEAM_FRAMES)]
    frames, i = [], 0
    while len(frames) < len(wanted):
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        while len(frames) < len(wanted) and wanted[len(frames)] <= t + 0.5 / src_fps:
            frames.append(bgr)
    cap.release()
    if len(frames) < SEAM_FRAMES + 2 * fps:
        raise ValueError(f"{path}: too short (need at least {t0 + 2:.1f} s)")
    h, w = frames[0].shape[:2]
    probe = frames[SEAM_FRAMES::max(1, fps // 2)]                   # face boxes on 2 frames a second
    x0, y0, side = crop_box([b for b in boxer.boxes(probe) if b != (0.0, 0.0, 0.0, 0.0)], w, h)
    framed = [cv2.cvtColor(cv2.resize(f[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB) for f in frames]
    lead_in, body = framed[:SEAM_FRAMES], framed[SEAM_FRAMES:]
    thumbs = np.stack([cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_RGB2GRAY) for f in body])
    n = loop_length(thumbs, min_len=max(2 * fps, len(body) - SEAM_FRAMES - int(SEAM_SEARCH_S * fps)))
    info = {"video": str(path), "source_fps": round(src_fps, 2), "start_s": round(t0, 2), "frames": n,
            "crop": [x0, y0, side]}
    return body[:n], lead_in, info
