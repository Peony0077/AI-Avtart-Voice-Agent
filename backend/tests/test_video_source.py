"""Avatar from a video (avatar/musetalk/video_source.py): framing, loop length, seam cross-fade."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from video_source import FACE_HEIGHT_SHARE, close_loop, crop_box, loop_length, seam_weights  # noqa: E402


def test_crop_frames_the_face_like_the_photo_avatars_and_stays_inside_the_video():
    boxes = [(560, 200, 760, 470), (565, 205, 765, 475), (555, 198, 755, 468)]   # 1280x720 video
    x0, y0, side = crop_box(boxes, 1280, 720)
    assert side == int(270 / FACE_HEIGHT_SHARE) or side == 720
    assert 0 <= x0 and x0 + side <= 1280 and 0 <= y0 and y0 + side <= 720
    assert x0 < 560 and x0 + side > 765                               # the face is inside, centred
    x0, y0, side = crop_box([(10, 10, 300, 600)], 640, 640)            # face near the edge, big
    assert (x0, y0) == (0, 0) and side <= 640


def test_loop_ends_where_the_video_looks_like_its_start():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (16, 16)).astype(np.uint8)
    thumbs = np.stack([np.clip(base.astype(int) + 40 * np.sin(i / 7), 0, 255).astype(np.uint8) for i in range(140)])
    n = loop_length(thumbs, min_len=60)
    assert 60 <= n <= 140 - 8
    assert abs(np.sin(n / 7)) < 0.15                                  # back near the start pose (sin = 0)


def test_seam_cross_fade_leads_from_the_end_into_the_start():
    w = seam_weights(8)
    assert 0 < w[0] < w[-1] < 1 and np.all(np.diff(w) > 0)
    frames = [np.full((2, 2, 3), 200, np.uint8) for _ in range(20)]
    lead_in = [np.full((2, 2, 3), 0, np.uint8) for _ in range(8)]       # the frames just before the start
    out = close_loop(frames, lead_in)
    assert out[11][0, 0, 0] == 200                                    # untouched before the seam
    assert out[12][0, 0, 0] > out[19][0, 0, 0]                         # fading towards the lead-in ...
    assert out[19][0, 0, 0] < 40                                      # ... almost there at the last frame
