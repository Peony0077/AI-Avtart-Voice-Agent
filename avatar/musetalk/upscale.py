"""Super-resolution for the avatar (Real-ESRGAN networks, BSD-3-Clause, xinntao/Real-ESRGAN).

Two places where the picture used to be stretched (owner, 2026-10-10: "the quality is bad, the resolution
is not good"):
  * build time (face_rig.py): LivePortrait draws its face crop at 512x512 for an area ~2.3x the face, so
    in a 1024x1024 avatar the face was that crop enlarged ~2x. RealESRGAN_x2plus doubles the crop first.
  * run time (engine.py): MuseTalk draws the lower face at 256x256 and it was enlarged to the face box.
    The same network doubles it before it is pasted, so the mouth is as crisp as the face around it.

The network definitions follow Real-ESRGAN / BasicSR (RRDBNet, SRVGGNetCompact) so the published weights
load unchanged; BasicSR itself is not used (it imports removed torchvision modules).
Weights (download_weights.py): MODEL_HOME/avatar/models/realesrgan/{RealESRGAN_x2plus,realesr-general-x4v3}.pth
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import WEIGHTS

ESRGAN_DIR = WEIGHTS / "models" / "realesrgan"


class _DenseBlock(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf: int, gc: int = 32) -> None:
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _DenseBlock(nf, gc), _DenseBlock(nf, gc), _DenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator; scale 2 works on the 2x pixel-unshuffled input (as RealESRGAN_x2plus)."""

    def __init__(self, scale: int = 2, num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.scale = scale
        in_ch = 3 * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.pixel_unshuffle(x, 2) if self.scale == 2 else F.pixel_unshuffle(x, 4) if self.scale == 1 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """The compact Real-ESRGAN network (realesr-general-x4v3): ~10x lighter than RRDBNet."""

    def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = 4) -> None:
        super().__init__()
        self.upscale = upscale
        layers: list[nn.Module] = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


MODELS = {
    # name: (file, constructor, scale)
    "x2plus": ("RealESRGAN_x2plus.pth", lambda: RRDBNet(scale=2), 2),
    "general-x4v3": ("realesr-general-x4v3.pth", lambda: SRVGGNetCompact(upscale=4), 4),
}


class Upscaler:
    """Batched super-resolution of BGR uint8 images on the GPU (fp16)."""

    def __init__(self, model: str = "x2plus", device: str = "cuda") -> None:
        file, make, self.scale = MODELS[model]
        path = ESRGAN_DIR / file
        if not path.exists():
            raise FileNotFoundError(f"{path} (run download_weights.py)")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("params_ema") or state.get("params") or state
        net = make()
        net.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.net = net.to(self.device, torch.float16).eval()
        self.name = model

    @torch.no_grad()
    def tensor(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] RGB in 0..1 (any float dtype, on the device) -> [B,3,sH,sW] RGB 0..1 fp16.
        Padded to even sizes for the 2x network (pixel unshuffle)."""
        h, w = x.shape[-2:]
        ph, pw = (h % 2, w % 2) if self.scale == 2 else (0, 0)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = self.net(x.to(self.device, torch.float16)).clamp_(0, 1)
        return y[..., : h * self.scale, : w * self.scale]

    def images(self, imgs_bgr: list[np.ndarray] | np.ndarray, batch: int = 4) -> list[np.ndarray]:
        """BGR uint8 images (same size within a batch) -> upscaled BGR uint8 images."""
        out: list[np.ndarray] = []
        for i in range(0, len(imgs_bgr), batch):
            chunk = np.stack(imgs_bgr[i:i + batch])[..., ::-1]                  # BGR -> RGB
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device).permute(0, 3, 1, 2).half() / 255
            y = self.tensor(x).mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
            out += [np.ascontiguousarray(f) for f in y]
        return out
