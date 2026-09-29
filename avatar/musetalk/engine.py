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


@dataclass
class Loop:
    name: str
    frames_jpg: list[bytes]            # full frames (BGR JPEG), ping-pong order
    coords: list[tuple]                # face box per frame
    latents: torch.Tensor              # [N, 8, 32, 32] fp16 (cpu)
    masks_png: list[bytes]             # blend masks (grayscale PNG)
    mask_boxes: list[tuple]
    size: tuple[int, int] = (512, 512)
    _decoded: dict = field(default_factory=dict, repr=False)

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
                         "masks_png": self.masks_png, "mask_boxes": self.mask_boxes, "size": self.size}, fh)

    @classmethod
    def load(cls, folder: Path) -> Loop:
        with open(folder / "loop.pkl", "rb") as fh:
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
    def prepare_loop(self, name: str, frames_bgr: list[np.ndarray], boxer, jpeg_quality: int = 93) -> Loop:
        from musetalk.utils.blending import get_image_prepare_material

        t0 = time.perf_counter()
        boxes = boxer.boxes(frames_bgr)
        coords, latents = [], []
        h, w = frames_bgr[0].shape[:2]
        last_good = None
        for box, frame in zip(boxes, frames_bgr, strict=True):
            if box == (0.0, 0.0, 0.0, 0.0):
                box = last_good
            x1, y1, x2, y2 = box
            y2 = min(y2 + EXTRA_MARGIN, h)
            y1, x1 = max(y1, 0), max(x1, 0)
            coords.append((x1, y1, x2, y2))
            last_good = box
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.vae.get_latents_for_unet(crop).to("cpu", self.dtype))
        # ping-pong so the loop can repeat without a jump
        frames = frames_bgr + frames_bgr[::-1]
        coords = coords + coords[::-1]
        latents = latents + latents[::-1]
        masks_png, mask_boxes = [], []
        for frame, box in zip(frames[: len(frames_bgr)], coords[: len(frames_bgr)], strict=True):
            mask, crop_box = get_image_prepare_material(frame, list(box), fp=self.fp, mode=PARSING_MODE)
            masks_png.append(cv2.imencode(".png", mask)[1].tobytes())
            mask_boxes.append(tuple(int(v) for v in crop_box))
        masks_png += masks_png[::-1]
        mask_boxes += mask_boxes[::-1]
        frames_jpg = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes() for f in frames]
        loop = Loop(name, frames_jpg, coords, torch.cat(latents), masks_png, mask_boxes, size=(w, h))
        logger.info(f"loop {name}: {len(loop)} frames prepared in {time.perf_counter() - t0:.1f}s")
        return loop

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
    def faces(self, loop: Loop, chunks: torch.Tensor, start_idx: int) -> tuple[list[int], np.ndarray]:
        """GPU stage: audio chunks [B,50,384] → lower part of the 256x256 face crops (BGR uint8).

        Only latent rows >= DECODE_FROM_ROW are decoded: MuseTalk's blend mask covers the lower half
        of the crop (y >= 128), so decoding the top is wasted work. Measured on the RTX 3060:
        44.7 → 27.7 ms/frame, mean abs difference 1.9/255 in the blended area (profile_vae.py).
        """
        b = chunks.shape[0]
        idx = [(start_idx + i) % len(loop) for i in range(b)]
        latent = loop.latents[idx].to(self.device, self.dtype)
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
        crop[256 - face_bottom.shape[0]:] = face_bottom        # top rows keep the original pixels
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
