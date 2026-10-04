"""MuseTalk 1.5 real-time lip-sync engine (MuseTalk: MIT, TMElyralab).

Loop preparation and frame generation follow MuseTalk's realtime_inference.py and
OpenAvatarChat's MuseTalkAlgoV15 (Apache-2.0), rewritten for this project:
  * the UNet is built on the meta device and filled straight from a memory-mapped
    checkpoint in fp16 on the GPU (the fp32 file is 3.2 GB; RAM on this PC is scarce)
  * loop frames and masks are kept JPEG/PNG-compressed in RAM
  * frames are produced in batches with a pts on the turn's audio timeline
"""
from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.signal import resample_poly

from common import WEIGHTS, chdir

EXTRA_MARGIN = 10     # MuseTalk 1.5: extra chin area below the landmark box
DECODE_FROM_ROW = 12  # latent row (x8 px) from which the face is decoded: 96 px margin above the blend line
PARSING_MODE = "jaw"


LIP_SHARPEN = 0.25   # unsharp mask on the generated face: as crisp as the photo's (Laplacian var 19 -> 26, photo 24)


def _lip_weight(lab: np.ndarray) -> np.ndarray:
    """Soft lip mask of a lower face in LAB: the reddest pixels (skin dominates, so the median a* is
    skin), not the dark inside of an open mouth."""
    a, lightness = lab[..., 1], lab[..., 0]
    w = np.clip((a - np.median(a[::4, ::4]) - 6.0) / 8.0, 0, 1) * np.clip((lightness - 70.0) / 30.0, 0, 1)
    return cv2.GaussianBlur(w, (5, 5), 0)


def refine_face(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """MuseTalk's lower face (BGR) made to look like the avatar's own (``ref``: the same rows of the
    loop frame). MuseTalk draws the lips pinker and matte, so they changed colour whenever the avatar
    started or stopped speaking (owner's test); their colour is moved to the avatar's lip colour
    (lip colour difference a*b* 5.1 -> 1.3 on 80 speech frames, stable from frame to frame), and the
    slightly soft output is sharpened to the photo's crispness."""
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    wg, wr = _lip_weight(g), _lip_weight(r)
    if wg.sum() > 50 and wr.sum() > 50:
        mg = (g * wg[..., None]).sum((0, 1)) / wg.sum()
        mr = (r * wr[..., None]).sum((0, 1)) / wr.sum()
        g += wg[..., None] * ((mr - mg) * np.array([0.7, 1.0, 1.0], np.float32))   # lightness only partly
        gen = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if LIP_SHARPEN:
        gen = cv2.addWeighted(gen, 1 + LIP_SHARPEN, cv2.GaussianBlur(gen, (0, 0), 1.2), -LIP_SHARPEN, 0)
    return gen


@dataclass
class Loop:
    name: str
    frames_jpg: list[bytes]            # full frames (BGR JPEG), in play order (ping-pong or seamless)
    coords: list[tuple]                # face box per frame
    latents: torch.Tensor              # [N, 8, 32, 32] fp16 (cpu)
    masks_png: list[bytes]             # blend masks (grayscale PNG)
    mask_boxes: list[tuple]
    size: tuple[int, int] = (512, 512)
    _decoded: dict = field(default_factory=dict, repr=False)
    latent_overrides: dict = field(default_factory=dict, repr=False)   # gesture variants: replaced rows
    pingpong: bool = True              # first avatars: forward + backward; new ones: seamless forward loop

    def latent(self, i: int) -> torch.Tensor:
        row = self.latent_overrides.get(i)
        return self.latents[i] if row is None else row

    def __len__(self) -> int:
        return len(self.frames_jpg)

    def frame(self, i: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.frames_jpg[i % len(self)], np.uint8), cv2.IMREAD_COLOR)

    def mask(self, i: int) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.masks_png[i % len(self)], np.uint8), cv2.IMREAD_GRAYSCALE)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.latents, folder / "latents.pt")
        with open(folder / "loop.pkl", "wb") as fh:
            pickle.dump({"name": self.name, "frames_jpg": self.frames_jpg, "coords": self.coords,
                         "masks_png": self.masks_png, "mask_boxes": self.mask_boxes, "size": self.size,
                         "pingpong": self.pingpong}, fh)

    @classmethod
    def load(cls, folder: Path) -> Loop:
        with open(folder / "loop.pkl", "rb") as fh:
            d = pickle.load(fh)
        return cls(latents=torch.load(folder / "latents.pt"), **d)

    def with_delta(self, delta: LoopDelta) -> Loop:
        """This loop with some frames replaced (gesture variants). Unchanged frames are shared with
        this loop, so a variant costs only its replaced frames in RAM."""
        n2 = len(self)
        frames, coords = list(self.frames_jpg), list(self.coords)
        masks, mboxes = list(self.masks_png), list(self.mask_boxes)
        overrides = {}
        for j, f in enumerate(delta.frame_idx):
            for pos in ((f, n2 - 1 - f) if self.pingpong else (f,)):   # and its ping-pong mirror
                frames[pos], coords[pos] = delta.frames_jpg[j], delta.coords[j]
                masks[pos], mboxes[pos] = delta.masks_png[j], delta.mask_boxes[j]
                overrides[pos] = delta.latents[j]
        # the latent tensor is shared with the base loop; only the replaced rows are stored
        return Loop(delta.name, frames, coords, self.latents, masks, mboxes, size=self.size, latent_overrides=overrides,
                    pingpong=self.pingpong)


@dataclass
class LoopDelta:
    """Frames of a gesture variant that differ from its base loop (forward-half indices)."""
    name: str
    base: str
    frame_idx: list[int]
    frames_jpg: list[bytes]
    coords: list[tuple]
    latents: torch.Tensor
    masks_png: list[bytes]
    mask_boxes: list[tuple]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.latents, folder / "latents.pt")
        with open(folder / "delta.pkl", "wb") as fh:
            pickle.dump({"name": self.name, "base": self.base, "frame_idx": self.frame_idx,
                         "frames_jpg": self.frames_jpg, "coords": self.coords, "masks_png": self.masks_png,
                         "mask_boxes": self.mask_boxes}, fh)

    @classmethod
    def load(cls, folder: Path) -> LoopDelta:
        with open(folder / "delta.pkl", "rb") as fh:
            d = pickle.load(fh)
        return cls(latents=torch.load(folder / "latents.pt"), **d)


class MuseTalkEngine:
    def __init__(self, fps: int = 20, batch_size: int = 4, device: str = "cuda") -> None:
        from diffusers import UNet2DConditionModel
        from transformers import WhisperModel

        t0 = time.perf_counter()
        self.fps, self.batch_size, self.device = fps, batch_size, torch.device(device)
        self.dtype = torch.float16
        with chdir(WEIGHTS):
            from musetalk.models.unet import PositionalEncoding
            from musetalk.models.vae import VAE
            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.utils.face_parsing import FaceParsing

            cfg = json.loads(Path("models/musetalkV15/musetalk.json").read_text())
            with torch.device("meta"):
                unet = UNet2DConditionModel(**cfg)
            state = torch.load("models/musetalkV15/unet.pth", map_location="cpu", mmap=True, weights_only=True)
            state = {k: v.to(self.device, self.dtype) for k, v in state.items()}
            unet.load_state_dict(state, assign=True, strict=True)
            del state
            self.unet = unet.to(self.device, self.dtype).eval()
            self.vae = VAE(model_path="models/sd-vae", use_float16=True)
            self.vae.vae = self.vae.vae.to(self.device, self.dtype).eval()
            self.vae.device = self.device
            self.pe = PositionalEncoding(d_model=384).to(self.device, self.dtype)
            self.audio = AudioProcessor(feature_extractor_path="models/whisper")
            self.whisper = WhisperModel.from_pretrained("models/whisper", torch_dtype=self.dtype).to(self.device).eval()
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        self.timesteps = torch.tensor([0], device=self.device)
        self.load_s = round(time.perf_counter() - t0, 2)
        logger.info(f"MuseTalk engine ready in {self.load_s}s (fps={fps}, batch={batch_size})")

    # ------------------------------------------------------------------ preparation
    @torch.no_grad()
    def _prepare_frames(self, frames_bgr: list[np.ndarray], boxer, fallback: list[tuple] | None = None
                        ) -> tuple[list[tuple], list[torch.Tensor], list[bytes], list[tuple]]:
        """Face box, VAE latent and blend mask per frame. A frame whose face is not detected reuses
        the previous box (or ``fallback[i]``, already adjusted, for gesture frames)."""
        from musetalk.utils.blending import get_image_prepare_material

        boxes = boxer.boxes(frames_bgr)
        h = frames_bgr[0].shape[0]
        coords, latents, masks_png, mask_boxes = [], [], [], []
        last_good = None
        for i, (box, frame) in enumerate(zip(boxes, frames_bgr, strict=True)):
            if box == (0.0, 0.0, 0.0, 0.0) and fallback is not None:
                x1, y1, x2, y2 = fallback[i]
            else:
                if box == (0.0, 0.0, 0.0, 0.0):
                    box = last_good
                x1, y1, x2, y2 = box
                y2 = min(y2 + EXTRA_MARGIN, h)
                y1, x1 = max(y1, 0), max(x1, 0)
                last_good = box
            coords.append((x1, y1, x2, y2))
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.vae.get_latents_for_unet(crop).to("cpu", self.dtype))
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=PARSING_MODE)
            masks_png.append(cv2.imencode(".png", mask)[1].tobytes())
            mask_boxes.append(tuple(int(v) for v in crop_box))
        return coords, latents, masks_png, mask_boxes

    def prepare_loop(self, name: str, frames_bgr: list[np.ndarray], boxer, jpeg_quality: int = 93,
                     pingpong: bool = True) -> Loop:
        """pingpong: append the frames backwards so a non-periodic loop repeats without a jump (each
        JPEG is shared by both directions); seamless loops (motion.idle_curves periodic) play forward."""
        t0 = time.perf_counter()
        h, w = frames_bgr[0].shape[:2]
        coords, latents, masks_png, mask_boxes = self._prepare_frames(frames_bgr, boxer)
        jpgs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes() for f in frames_bgr]
        if pingpong:
            loop = Loop(name, jpgs + jpgs[::-1], coords + coords[::-1], torch.cat(latents + latents[::-1]),
                        masks_png + masks_png[::-1], mask_boxes + mask_boxes[::-1], size=(w, h))
        else:
            loop = Loop(name, jpgs, coords, torch.cat(latents), masks_png, mask_boxes, size=(w, h), pingpong=False)
        logger.info(f"loop {name}: {len(loop)} frames prepared in {time.perf_counter() - t0:.1f}s")
        return loop

    def prepare_delta(self, base: Loop, name: str, frames: dict[int, np.ndarray], boxer,
                      jpeg_quality: int = 93) -> LoopDelta:
        """Gesture variant of ``base``: only the frames inside the gesture windows are prepared."""
        t0 = time.perf_counter()
        idx = sorted(frames)
        imgs = [frames[f] for f in idx]
        coords, latents, masks_png, mask_boxes = self._prepare_frames(imgs, boxer, [base.coords[f] for f in idx])
        jpgs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes() for f in imgs]
        delta = LoopDelta(name, base.name, idx, jpgs, coords, torch.cat(latents), masks_png, mask_boxes)
        logger.info(f"gesture variant {name}: {len(idx)} frames prepared in {time.perf_counter() - t0:.1f}s")
        return delta

    # ------------------------------------------------------------------ real time
    @torch.no_grad()
    def audio_features(self, pcm16: np.ndarray, sample_rate: int = 24000) -> torch.Tensor:
        """PCM16 mono → whisper-tiny features per video frame: [n_frames, 50, 384]."""
        audio = pcm16.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            g = math.gcd(16000, sample_rate)
            audio = resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)
        feats = self.audio.feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_features
        return self.audio.get_whisper_chunk([feats.to(self.dtype)], self.device, self.dtype, self.whisper,
                                            len(audio), fps=self.fps, audio_padding_length_left=2,
                                            audio_padding_length_right=2)

    @torch.no_grad()
    def faces(self, loop: Loop, chunks: torch.Tensor, start_idx: int,
              sources: list[Loop] | None = None, frame_idx: list[int] | None = None) -> tuple[list[int], np.ndarray]:
        """GPU stage: audio chunks [B,50,384] → lower part of the 256x256 face crops (BGR uint8).

        Frames start_idx, start_idx + 1, ... of the loop, or the frames ``frame_idx`` (not consecutive
        when the worker renders every 2nd or 3rd frame to keep up).

        Only latent rows >= DECODE_FROM_ROW are decoded: MuseTalk's blend mask covers the lower half
        of the crop (y >= 128), so decoding the top is wasted work. Measured on the RTX 3060:
        44.7 → 27.7 ms/frame, mean abs difference 1.9/255 in the blended area (profile_vae.py).
        """
        b = chunks.shape[0]
        idx = [i % len(loop) for i in (frame_idx if frame_idx is not None else range(start_idx, start_idx + b))]
        if sources is None:
            latent = loop.latents[idx].to(self.device, self.dtype)
        else:   # per-frame source loop (base loop, or its gesture variant inside a gesture window)
            latent = torch.stack([src.latent(i) for src, i in zip(sources, idx, strict=True)]).to(self.device, self.dtype)
        cond = self.pe(chunks.to(self.device, self.dtype))
        pred = self.unet(latent, self.timesteps, encoder_hidden_states=cond).sample
        lat = pred[:, :, DECODE_FROM_ROW:, :] / self.vae.scaling_factor
        img = self.vae.vae.decode(lat.to(self.vae.vae.dtype)).sample
        img = (img / 2 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8)
        return idx, img.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]   # RGB → BGR

    def compose(self, loop: Loop, i: int, face_bottom: np.ndarray) -> np.ndarray:
        """CPU stage: paste the generated lower face into loop frame i (MuseTalk blending)."""
        frame = loop.frame(i)
        x1, y1, x2, y2 = loop.coords[i]
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_AREA)
        h = face_bottom.shape[0]
        crop[256 - h:] = refine_face(np.ascontiguousarray(face_bottom), crop[256 - h:])   # top rows keep the original
        face = cv2.resize(crop, (x2 - x1, y2 - y1))
        xs, ys, xe, ye = loop.mask_boxes[i]
        body = frame[ys:ye, xs:xe]
        large = body.copy()
        large[y1 - ys:y2 - ys, x1 - xs:x2 - xs] = face
        m = loop.mask(i).astype(np.float32)[..., None] / 255.0
        hh, ww = min(large.shape[0], m.shape[0]), min(large.shape[1], m.shape[1])
        frame[ys:ys + hh, xs:xs + ww] = (large[:hh, :ww] * m[:hh, :ww] + body[:hh, :ww] * (1 - m[:hh, :ww])).astype(np.uint8)
        return frame

    def generate(self, loop: Loop, chunks: torch.Tensor, start_idx: int) -> list[np.ndarray]:
        """Both stages, sequentially (tests/benchmarks; the worker pipelines them)."""
        idx, faces = self.faces(loop, chunks, start_idx)
        return [self.compose(loop, i, f) for i, f in zip(idx, faces, strict=True)]

    def warmup(self, loop: Loop) -> float:
        t0 = time.perf_counter()
        chunks = self.audio_features(np.zeros(24000, np.int16))
        self.generate(loop, chunks[: self.batch_size], 0)
        torch.cuda.synchronize()
        return round(time.perf_counter() - t0, 2)
