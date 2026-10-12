"""HD avatar (owner, 2026-10-10: "the quality is bad, the resolution is not good"): H.264 High profile,
the lower face pasted at the frame's own resolution (also when MuseTalk's crop box leaves the frame), and
expression fades by absolute frame number. The compositing tests need OpenCV (the avatar venv)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from h264 import encoder_args, pick_encoder  # noqa: E402


def test_both_encoders_use_the_high_profile():
    for enc in ("nvenc", "libx264"):
        args = encoder_args(enc)
        assert args[args.index("-profile:v") + 1] == "high" and args[args.index("-level") + 1] == "4.1"
    assert "ultrafast" not in encoder_args("libx264")          # no deblocking, no CABAC: blocky faces
    assert pick_encoder("libx264") == "libx264"


def _engine_and_loop(frame: np.ndarray, coords: tuple, mask_box: tuple):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torch")
    from engine import Loop, MuseTalkEngine

    eng = object.__new__(MuseTalkEngine)       # compose() needs no model
    eng.sharpen = 0.0
    xs, ys, xe, ye = mask_box
    mask = np.zeros((ye - ys, xe - xs), np.uint8)
    mask[(ye - ys) // 2:, :] = 255                  # MuseTalk's mask: the lower half of its crop box
    loop = Loop("t", [cv2.imencode(".png", frame)[1].tobytes()], [coords], None,
                [cv2.imencode(".png", mask)[1].tobytes()], [mask_box], size=frame.shape[1::-1])
    return eng, loop


def test_a_crop_box_past_the_top_left_edge_is_clipped():
    frame = np.full((300, 300, 3), 100, np.uint8)
    eng, loop = _engine_and_loop(frame, (10, 5, 110, 125), (-30, -40, 150, 140))   # 1.5x box leaves the frame
    face = np.full((160, 256, 3), 200, np.uint8)
    out = eng.compose(loop, 0, face)
    assert out.shape == frame.shape
    assert out[100, 60].mean() > 150                     # the lower face was pasted (not wrapped / failed)
    assert (out[200:, :] == 100).all()                   # nothing outside the mask box changed


def test_a_super_resolved_lower_face_is_pasted_at_full_size_and_leaves_the_upper_face():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    coords = (300, 300, 732, 780)                         # a 432 px face box, as in the 1024 avatar
    eng, loop = _engine_and_loop(frame, coords, (192, 216, 840, 864))
    face = np.full((320, 512, 3), 128, np.uint8)          # Real-ESRGAN output: 2x the 256-wide decode
    out = eng.compose(loop, 0, face)
    top = 300 + round(480 * (256 - 160) / 256)
    assert (out[:top] == frame[:top]).all()               # the upper face stays as it is (no 256 px round trip)
    assert abs(int(out[700, 516].mean()) - 128) <= 2     # inside the mask: the generated lower face


def test_expression_fades_count_absolute_frames():
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    from worker import Turn

    turn = Turn(1, base_idx=100, loop="happy", fade_from="friendly", fade_end=108, fade_total=8)
    assert turn.fade_weight(100) == pytest.approx(1 - 8 / 9)
    assert turn.fade_weight(104) == pytest.approx(1 - 4 / 9)   # also when frames 101-103 were skipped (stride)
    assert turn.fade_weight(108) is None
    turn.fade_end = 130                                         # never more than fade_total frames from the end
    assert turn.fade_weight(100) == pytest.approx(1 - 8 / 9)


def test_the_blend_mask_stays_inside_the_face_outline():
    """Owner, 2026-10-12: "while she talks, her right cheek has an issue" - MuseTalk's jaw mask reached past the
    jawline and the generated (narrower) jaw showed as a ghost outline. The mask is ~0 on the outline at the sides,
    full over the mouth, empty in the upper half, and reaches below the outline only under the middle of the chin."""
    pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from engine import CHIN_DROP, CHIN_HALF_WIDTH, contour_safe_mask

    class FakeParsing:                        # BiSeNet stand-in: label 1 (skin) inside an ellipse of the 512 input
        preprocess = staticmethod(lambda img: torch.zeros(3, 512, 512))

        def __init__(self):
            yy, xx = np.mgrid[:512, :512]
            self.face = ((xx - 256) / 120.0) ** 2 + ((yy - 256) / 150.0) ** 2 <= 1.0   # chin at y = 406
            self.net = self

        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, img):
            logits = torch.zeros(1, 19, 512, 512)
            logits[0, 1][torch.from_numpy(self.face)] = 1.0
            logits[0, 0][torch.from_numpy(~self.face)] = 1.0
            return [logits]

    frame = np.zeros((1024, 1024, 3), np.uint8)
    face_box = (342, 342, 682, 682)                       # crop box 1.5x: ~512 px (the parsing scale); the face
    #                                                      box ends at y ~427 there, below the chin as in real frames
    mask, crop_box = contour_safe_mask(frame, face_box, FakeParsing())
    assert crop_box == (257, 257, 767, 767) and mask.shape == (510, 510)
    m = mask.astype(np.float32) / 255
    s = mask.shape[0] / 512
    row = int(330 * s)                                    # a row in the lower half
    half = 120 * np.sqrt(1 - ((330 - 256) / 150) ** 2)    # the outline's x at that row
    assert m[row, int((256 - half) * s)] < 0.05 and m[row, int((256 + half) * s) - 1] < 0.05   # outline: the frame's own
    assert m[int(360 * s), int(256 * s)] > 0.95           # the mouth / chin area: lip sync
    assert m[: int(240 * s)].max() < 0.05                 # the upper half
    below = int((406 + CHIN_DROP // 2) * s)               # just below the chin's outline (y = 406)
    assert m[below, int(256 * s)] > 0.3                   # the jaw may drop in the middle ...
    assert m[below, int((256 - CHIN_HALF_WIDTH - 40) * s)] < 0.05   # ... but not at the sides
