"""Face bounding boxes for MuseTalk without mmcv/mmpose.

Adapted from OpenAvatarChat (Apache-2.0),
src/handlers/avatar/musetalk/musetalk_utils_preprocessing.py: DWPose runs through
ONNX Runtime (CPU) and the S3FD detector comes from MuseTalk's face_detection
package. Works on in-memory BGR frames instead of files.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
import torch
from loguru import logger

from common import WEIGHTS

_INPUT = (288, 384)  # (w, h) RTMPose-L wholebody 384x288
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)
_SPLIT = 2.0
PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def _warp_matrix(center, scale, out_size):
    src_w = scale[0]
    dst_w, dst_h = out_size
    src_dir = np.array([0.0, src_w * -0.5])
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = src[1] + np.r_[-(src[0] - src[1])[1], (src[0] - src[1])[0]]
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = dst[1] + np.r_[-(dst[0] - dst[1])[1], (dst[0] - dst[1])[0]]
    return cv2.getAffineTransform(src, dst)


def _dwpose(session: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = np.array([w * 0.5, h * 0.5], np.float32)
    scale = np.array([w * 1.25, h * 1.25], np.float32)
    aspect = _INPUT[0] / _INPUT[1]
    scale = np.array([scale[0], scale[0] / aspect] if scale[0] > scale[1] * aspect else [scale[1] * aspect, scale[1]],
                     np.float32)
    warped = cv2.warpAffine(img, _warp_matrix(center, scale, _INPUT), _INPUT, flags=cv2.INTER_LINEAR)
    x = ((cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1)[None]
    simcc_x, simcc_y = session.run(None, {session.get_inputs()[0].name: x})[:2]
    locs = np.stack([simcc_x[0].argmax(-1), simcc_y[0].argmax(-1)], -1).astype(np.float32) / _SPLIT
    return locs / np.array(_INPUT, np.float32) * scale + center - scale / 2   # (133, 2)


class FaceBoxer:
    def __init__(self) -> None:
        from musetalk.utils.face_detection import FaceAlignment, LandmarksType

        self.session = ort.InferenceSession(str(WEIGHTS / "models/dwpose/dw-ll_ucoco_384.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.fa = FaceAlignment(LandmarksType._2D, flip_input=False,
                                device="cuda" if torch.cuda.is_available() else "cpu")

    def boxes(self, frames_bgr: list[np.ndarray], bbox_shift: int = 0) -> list[tuple]:
        out = []
        for frame in frames_bgr:
            lm = _dwpose(self.session, frame)[23:91].astype(np.int32)   # 68 face landmarks
            det = self.fa.get_detections_for_batch(np.asarray([frame]))[0]
            if det is None:
                out.append(PLACEHOLDER)
                continue
            half = lm[29].copy()
            half[1] += bbox_shift
            upper = half[1] - (lm[:, 1].max() - half[1])
            box = (int(lm[:, 0].min()), int(upper), int(lm[:, 0].max()), int(lm[:, 1].max()))
            if box[3] - box[1] <= 0 or box[2] - box[0] <= 0 or box[0] < 0:
                logger.warning(f"bad landmark box {box}, using detector box {det}")
                box = tuple(int(v) for v in det)
            out.append(box)
        return out
