"""Procedural face rig on top of LivePortrait (MIT; KlingAIResearch/LivePortrait).

Given ONE portrait photo, renders frames with explicit per-frame control of
head pose (pitch/yaw/roll), eye openness (blinks), eyeball direction (gaze),
smile and eyebrows — while keeping the lips closed. The composition mirrors
LivePortrait's own ``GradioPipeline.execute_image_retargeting``.
Note: LivePortrait's face detector uses InsightFace models (non-commercial use only).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from common import WEIGHTS  # also sets up sys.path for vendor/LivePortrait
from motion import RigParams

from src.config.crop_config import CropConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import paste_back, prepare_paste_back  # noqa: E402
from src.utils.cropper import Cropper  # noqa: E402
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio  # noqa: E402

LP = WEIGHTS / "liveportrait"


class FaceRig:
    """``upscaler``: a Real-ESRGAN model (upscale.py) that doubles LivePortrait's 512x512 face crop before it is
    pasted back - for avatars above 512x512, where the crop covers ~1100 px and was enlarged ~2x. ``detail``:
    its share of the result (the rest bicubic: some skin texture stays)."""

    def __init__(self, device_id: int = 0, upscaler: str | None = None, detail: float = 0.7) -> None:
        inf = InferenceConfig(
            checkpoint_F=str(LP / "liveportrait/base_models/appearance_feature_extractor.pth"),
            checkpoint_M=str(LP / "liveportrait/base_models/motion_extractor.pth"),
            checkpoint_G=str(LP / "liveportrait/base_models/spade_generator.pth"),
            checkpoint_W=str(LP / "liveportrait/base_models/warping_module.pth"),
            checkpoint_S=str(LP / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
            flag_use_half_precision=True, device_id=device_id,
        )
        self.crop_cfg = CropConfig(insightface_root=str(LP / "insightface"),
                                   landmark_ckpt_path=str(LP / "liveportrait/landmark.onnx"), device_id=device_id)
        self.wrapper = LivePortraitWrapper(inference_cfg=inf)
        self.cropper = Cropper(crop_cfg=self.crop_cfg, device_id=device_id)
        self.mask_crop = inf.mask_crop
        self.device = self.wrapper.device
        self.sr = None
        self.detail = detail
        if upscaler:
            from upscale import Upscaler
            self.sr = Upscaler(upscaler, device=f"cuda:{device_id}")

    @torch.no_grad()
    def load_source(self, img_rgb: np.ndarray) -> dict:
        crop = self.cropper.crop_source_image(img_rgb, self.crop_cfg)
        if crop is None:
            raise ValueError("no face found in the photo")
        w = self.wrapper
        I_s = w.prepare_source(crop["img_crop_256x256"])
        info = w.get_kp_info(I_s)
        return {
            "img": img_rgb, "info": info, "f_s": w.extract_feature_3d(I_s), "x_s": w.transform_keypoint(info),
            "R_s": get_rotation_matrix(info["pitch"], info["yaw"], info["roll"]),
            "lmk": crop["lmk_crop"], "M_c2o": crop["M_c2o"],
            "mask": prepare_paste_back(self.mask_crop, crop["M_c2o"], dsize=(img_rgb.shape[1], img_rgb.shape[0])),
            "eye_ratio": float(np.mean(calc_eye_close_ratio(crop["lmk_crop"][None]))),   # mean of left/right
            "lip_ratio": float(np.mean(calc_lip_close_ratio(crop["lmk_crop"][None]))),
        }

    @torch.no_grad()
    def render(self, src: dict, p: RigParams) -> np.ndarray:
        w, dev, info = self.wrapper, self.device, src["info"]
        R_d = get_rotation_matrix(info["pitch"] + p.pitch, info["yaw"] + p.yaw, info["roll"] + p.roll)
        R_new = (R_d @ src["R_s"].permute(0, 2, 1)) @ src["R_s"]
        delta = info["exp"].clone().to(dev)
        if p.gaze_x or p.gaze_y:
            _eyeball(delta, p.gaze_x, p.gaze_y)
        if p.smile:
            _smile(delta, p.smile)
        if p.eyebrow:
            _eyebrow(delta, p.eyebrow)
        t = info["t"].clone()
        t[..., 2] = 0  # no depth translation (as in LivePortrait's video pipeline and transform_keypoint)
        x_d = info["scale"] * (info["kp"] @ R_new + delta) + t
        # eyes: target openness relative to the photo; lips: keep closed
        eye_target = max(0.0, src["eye_ratio"] * p.eye_open)
        if abs(p.eye_open - 1.0) > 1e-3:
            ratio = w.calc_combined_eye_ratio([[eye_target]], src["lmk"])
            x_d = x_d + w.retarget_eye(src["x_s"], ratio)
        lip = w.calc_combined_lip_ratio([[0.0]], src["lmk"])
        x_d = x_d + w.retarget_lip(src["x_s"], lip)
        x_d = w.stitching(src["x_s"], x_d)
        out = w.warp_decode(src["f_s"], src["x_s"], x_d)
        if self.sr is None or not self._needs_sr(src):
            crop_rgb = w.parse_output(out["out"])[0]
            return paste_back(crop_rgb, src["M_c2o"], src["img"], src["mask"])
        # the crop doubled (Real-ESRGAN, blended with bicubic) and pasted with the matrix for a 2x crop
        x = out["out"].float().clamp(0, 1)                                   # [1,3,512,512] RGB
        up = torch.nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        hr = (self.detail * self.sr.tensor(x).float() + (1 - self.detail) * up).clamp(0, 1)
        crop_rgb = hr[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        m = src["M_c2o"] @ np.diag([0.5, 0.5, 1.0]).astype(np.float32)
        h, w_ = src["img"].shape[:2]
        pasted = cv2.warpAffine(crop_rgb, m[:2], (w_, h), flags=cv2.INTER_CUBIC)
        mask = src["mask"]
        return np.clip(mask * pasted + (1 - mask) * src["img"], 0, 255).astype(np.uint8)

    @staticmethod
    def _needs_sr(src: dict) -> bool:
        """The 512x512 crop covers more than ~600 px of the photo: it would be enlarged."""
        m = src["M_c2o"]
        return float(np.hypot(m[0, 0], m[1, 0])) * 512 > 600


# ---- expression deltas: copied from LivePortrait src/gradio_pipeline.py (MIT), which
# ---- imports gradio at module level (not installed in the avatar venv)
def _eyeball(d: torch.Tensor, x: float, y: float) -> None:
    if x > 0:
        d[0, 11, 0] += x * 0.0007
        d[0, 15, 0] += x * 0.001
    else:
        d[0, 11, 0] += x * 0.001
        d[0, 15, 0] += x * 0.0007
    d[0, 11, 1] += y * -0.001
    d[0, 15, 1] += y * -0.001
    blink = -y / 2.0
    d[0, 11, 1] += blink * -0.001
    d[0, 13, 1] += blink * 0.0003
    d[0, 15, 1] += blink * -0.001
    d[0, 16, 1] += blink * 0.0003


def _smile(d: torch.Tensor, s: float) -> None:
    d[0, 20, 1] += s * -0.01
    d[0, 14, 1] += s * -0.02
    d[0, 17, 1] += s * 0.0065
    d[0, 17, 2] += s * 0.003
    d[0, 13, 1] += s * -0.00275
    d[0, 16, 1] += s * -0.00275
    d[0, 3, 1] += s * -0.0035
    d[0, 7, 1] += s * -0.0035


def _eyebrow(d: torch.Tensor, e: float) -> None:
    if e > 0:
        d[0, 1, 1] += e * 0.001
        d[0, 2, 1] += e * -0.001
    else:
        d[0, 1, 0] += e * -0.001
        d[0, 2, 0] += e * 0.001
        d[0, 1, 1] += e * 0.0003
        d[0, 2, 1] += e * -0.0003


def load_photo(path, size: int = 512, upscaler=None, detail: float = 0.7) -> np.ndarray:
    """Read a photo as RGB, center-crop to a square and resize to ``size``. A photo smaller than ``size`` is
    enlarged with ``upscaler`` (upscale.Upscaler) when given - hair, clothes and background come straight
    from it - else bicubic."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    bgr = bgr[y0:y0 + s, x0:x0 + s]
    while upscaler is not None and bgr.shape[0] < size:
        big = upscaler.images([bgr])[0]
        soft = cv2.resize(bgr, big.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        bgr = cv2.addWeighted(big, detail, soft, 1 - detail, 0)
    interp = cv2.INTER_AREA if bgr.shape[0] >= size else cv2.INTER_CUBIC
    bgr = cv2.resize(bgr, (size, size), interpolation=interp)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
