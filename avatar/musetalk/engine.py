"""MuseTalk 1.5 real-time lip-sync engine (MuseTalk: MIT, TMElyralab).

Loop preparation and frame generation follow MuseTalk's realtime_inference.py and
OpenAvatarChat's MuseTalkAlgoV15 (Apache-2.0), rewritten for this project:
  * the UNet is built on the meta device and filled straight from a memory-mapped
    checkpoint in fp16 on the GPU (the fp32 file is 3.2 GB; RAM on this PC is scarce)
  * loop frames and masks are kept JPEG/PNG-compressed in RAM
  * frames are produced in batches with a pts on the turn's audio timeline
  * for avatars larger than 512x512 the generated lower face (256 px wide) is doubled by Real-ESRGAN
    (upscale.py) and pasted at the frame's own resolution: enlarged with bilinear interpolation, it was a
    soft patch in a sharp face
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
import torch.nn.functional as F
from loguru import logger
from scipy.signal import resample_poly

import lipsync
from common import WEIGHTS, chdir

EXTRA_MARGIN = 10     # MuseTalk 1.5: extra chin area below the landmark box
DECODE_FROM_ROW = 12  # latent row (x8 px) from which the face is decoded: 96 px margin above the blend line


LIP_SHARPEN = 0.25   # unsharp mask on the generated face: as crisp as the photo's (Laplacian var 19 -> 26, photo 24)

# Blend mask (owner, 2026-10-12: "while she talks, her right cheek has an issue"). MuseTalk's "jaw" mask grows the
# face region downwards by ~21 px of its 512 px parsing (so the jaw can move) and narrows it only sideways: along
# a slanted jawline it reached ~30 px past the face into the hair, and MuseTalk draws the jaw up to ~12 px narrower
# than the frame's own - both outlines showed, a grey ghost band along the jaw (sharper, so more visible, in the
# HD avatar). This mask fades the generated face out INSIDE the face's outline, so the outline is always the
# frame's own; only under the middle of the chin may it reach further down (the jaw drops when the mouth opens).
MASK_INSET = 12        # px (512 parsing) inside the outline: the narrower jaw (~8) and the blur's reach (~4)
CHIN_DROP = 14         # px (512 parsing) the generated chin may reach below the outline ...
CHIN_HALF_WIDTH = 50   # ... within this many px either side of the centre (~15 % of the face's width)
MASK_BLUR = 0.05       # Gaussian kernel, share of the crop's size (MuseTalk: 0.1)
FACE_LABELS = (1, 11, 12, 13)   # BiSeNet: skin, mouth, upper lip, lower lip (not the nose, neck, ears or hair)


def contour_safe_mask(frame_bgr: np.ndarray, face_box, fp) -> tuple[np.ndarray, tuple]:
    """(blend mask, crop box) for loop frame ``frame_bgr`` - a drop-in for MuseTalk's
    get_image_prepare_material(mode="jaw"): the same crop box (1.5x the face box), lower half only, limited to
    the face box, but kept inside the face's outline (see MASK_INSET)."""
    from musetalk.utils.blending import get_crop_box
    from PIL import Image

    x, y, x1, y1 = (int(v) for v in face_box)
    crop_box, _ = get_crop_box([x, y, x1, y1], 1.5)
    xs, ys, xe, ye = crop_box
    large = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])).crop(crop_box)   # padded past the frame
    w, h = large.size
    with torch.no_grad():
        img = fp.preprocess(large.resize((512, 512), Image.BILINEAR)).unsqueeze(0)
        img = img.to(next(fp.net.parameters()).device)
        parsing = fp.net(img)[0].squeeze(0).argmax(0).cpu().numpy()
    face = np.isin(parsing, FACE_LABELS).astype(np.uint8)
    k = 2 * MASK_INSET + 1
    region = cv2.erode(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # under the middle of the chin: the face itself and up to CHIN_DROP px below it
    dropped = face.copy()
    for d in range(1, CHIN_DROP + 1):
        dropped[d:] |= face[:-d]
    middle = slice(256 - CHIN_HALF_WIDTH, 256 + CHIN_HALF_WIDTH)
    region[:, middle] = np.maximum(region[:, middle], dropped[:, middle])
    region[:256] = 0                                                     # the upper half stays the frame's own
    fx0, fy0 = round((x - xs) * 512 / w), round((y - ys) * 512 / h)       # and outside the face box too
    fx1, fy1 = round((x1 - xs) * 512 / w), round((y1 - ys) * 512 / h)
    inside = np.zeros_like(region)
    inside[max(fy0, 0):fy1, max(fx0, 0):fx1] = 1
    mask = cv2.resize((region & inside) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    blur = int(MASK_BLUR * h) // 2 * 2 + 1
    return cv2.GaussianBlur(mask, (blur, blur), 0), tuple(crop_box)


def _lip_weight(lab: np.ndarray) -> np.ndarray:
    """Soft lip mask of a lower face in LAB: the reddest pixels (skin dominates, so the median a* is
    skin), not the dark inside of an open mouth."""
    a, lightness = lab[..., 1], lab[..., 0]
    w = np.clip((a - np.median(a[::4, ::4]) - 6.0) / 8.0, 0, 1) * np.clip((lightness - 70.0) / 30.0, 0, 1)
    return cv2.GaussianBlur(w, (5, 5), 0)


def _lip_halves(lab: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Centre row of the lips and the mean colour of the upper and of the lower lip."""
    rows = np.arange(lab.shape[0], dtype=np.float32)[:, None]
    cy = float((w * rows).sum() / w.sum())
    up, lo = w * (rows < cy), w * (rows >= cy)
    return (cy, (lab * up[..., None]).sum((0, 1)) / max(float(up.sum()), 1e-6),
            (lab * lo[..., None]).sum((0, 1)) / max(float(lo.sum()), 1e-6))


def _lip_shift(g: np.ndarray, r: np.ndarray, wg: np.ndarray, wr: np.ndarray) -> np.ndarray:
    """Per-row LAB shift towards the avatar's own lips: upper and lower lip matched separately (MuseTalk
    draws the lower lip pinker than the upper one; one shift for both left it pink), blended across the
    lip centre line; lightness only partly."""
    cy, gu, gl = _lip_halves(g, wg)
    _, ru, rl = _lip_halves(r, wr)
    gain = np.array([0.7, 1.0, 1.0], np.float32)
    t = np.clip((np.arange(g.shape[0], dtype=np.float32) - (cy - 4)) / 8, 0, 1)
    t = (t * t * (3 - 2 * t))[:, None, None]
    return (1 - t) * ((ru - gu) * gain) + t * ((rl - gl) * gain)


def _teeth(g: np.ndarray, wg: np.ndarray) -> None:
    """Teeth (bright, nearly colourless pixels next to the lips) a little whiter and crisper; MuseTalk
    draws them as a soft grey strip. In place, LAB."""
    lightness, a, b = g[..., 0], g[..., 1], g[..., 2]
    near_lips = cv2.dilate((wg > 0.3).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
    chroma = np.abs(a - 128) + np.abs(b - 128)
    wt = np.clip((lightness - 120) / 40, 0, 1) * np.clip(1 - (chroma - 14) / 14, 0, 1) * near_lips
    wt = cv2.GaussianBlur(wt.astype(np.float32), (5, 5), 0)
    if wt.sum() > 5:
        detail = lightness - cv2.GaussianBlur(lightness, (0, 0), 1.5)
        g[..., 0] = lightness + wt * (10 + 0.6 * detail)
        g[..., 1] = 128 + (a - 128) * (1 - 0.4 * wt)
        g[..., 2] = 128 + (b - 128) * (1 - 0.4 * wt)


def refine_face(gen: np.ndarray, ref: np.ndarray, sharpen: float = LIP_SHARPEN) -> np.ndarray:
    """MuseTalk's lower face (BGR) made to look like the avatar's own (``ref``: the same rows of the
    loop frame). MuseTalk draws the lips pinker and matte, so they changed colour whenever the avatar
    started or stopped speaking (owner's test); their colour is moved to the avatar's lip colour, the
    upper and the lower lip separately (a*b* difference upper 5.7 -> 2.7, lower 6.8 -> 1.4 on 96 speech
    frames; one shift for both left the lower lip at 2.6), the teeth are made a little whiter and
    crisper, and a soft decoder output (SD VAE) is sharpened to the photo's crispness. ~7 ms CPU."""
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    wg, wr = _lip_weight(g), _lip_weight(r)
    if wg.sum() > 50 and wr.sum() > 50:
        g += wg[..., None] * _lip_shift(g, r, wg, wr)
        _teeth(g, wg)
        gen = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if sharpen:
        gen = cv2.addWeighted(gen, 1 + sharpen, cv2.GaussianBlur(gen, (0, 0), 1.2), -sharpen, 0)
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
        (folder / "loop.pkl.orig").unlink(missing_ok=True)   # body_motion's copy of the previous build
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
        (folder / "delta.pkl.orig").unlink(missing_ok=True)  # body_motion's copy of the previous build
        with open(folder / "delta.pkl", "wb") as fh:
            pickle.dump({"name": self.name, "base": self.base, "frame_idx": self.frame_idx,
                         "frames_jpg": self.frames_jpg, "coords": self.coords, "masks_png": self.masks_png,
                         "mask_boxes": self.mask_boxes}, fh)

    @classmethod
    def load(cls, folder: Path) -> LoopDelta:
        with open(folder / "delta.pkl", "rb") as fh:
            d = pickle.load(fh)
        return cls(latents=torch.load(folder / "latents.pt"), **d)


SR_MIN_FACE = 320    # face boxes wider than this (px) get the super-resolved lower face (avatars > 512x512)
SR_DETAIL = 0.7      # share of the Real-ESRGAN picture (rest: bicubic) - keeps some skin texture, as the loops


class MuseTalkEngine:
    def __init__(self, fps: int = 20, batch_size: int = 4, device: str = "cuda", decoder: str = "vae",
                 upscaler: str | None = None) -> None:
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
        # decoder "taesd": the tiny SD autoencoder (madebyollin/taesd, MIT) decodes MuseTalk's latents in
        # 2.2 ms instead of 26.4 ms per frame (UNet 13.8 ms), so lip sync renders ~2.5x faster and keeps
        # up while the LLM shares the GPU. Same look (PSNR 30.9 dB against the SD VAE, lips a little
        # crisper: Laplacian variance 33 vs 21, photo 25), so its output is not sharpened.
        self.tiny = None
        self.sharpen = LIP_SHARPEN
        if decoder == "taesd":
            path = WEIGHTS / "models" / "taesd"
            if (path / "config.json").exists():
                from diffusers import AutoencoderTiny
                self.tiny = AutoencoderTiny.from_pretrained(str(path), torch_dtype=self.dtype).to(self.device).eval()
                self.sharpen = 0.0
            else:
                logger.warning(f"TAESD not found in {path}: using the SD VAE decoder")
        self.decoder = "taesd" if self.tiny is not None else "vae"
        # Real-ESRGAN for the lower face of large avatars (the worker turns it on: use_sr)
        self.sr = None
        if upscaler:
            from upscale import Upscaler
            self.sr = Upscaler(upscaler, device=device)
        self.use_sr = False
        self.load_s = round(time.perf_counter() - t0, 2)
        logger.info(f"MuseTalk engine ready in {self.load_s}s (fps={fps}, batch={batch_size}, decoder={self.decoder}, "
                    f"upscaler={upscaler})")

    # ------------------------------------------------------------------ preparation
    @torch.no_grad()
    def _prepare_frames(self, frames_bgr: list[np.ndarray], boxer, fallback: list[tuple] | None = None
                        ) -> tuple[list[tuple], list[torch.Tensor], list[bytes], list[tuple]]:
        """Face box, VAE latent and blend mask per frame. A frame whose face is not detected reuses
        the previous box (or ``fallback[i]``, already adjusted, for gesture frames)."""
        boxes = boxer.boxes(frames_bgr)
        h = frames_bgr[0].shape[0]
        coords, latents, masks_png, mask_boxes = [], [], [], []
        # a frame without a detected face takes the previous box - the first ones the first detected box
        last_good = next((b for b in boxes if b != (0.0, 0.0, 0.0, 0.0)), None)
        if last_good is None and fallback is None:
            raise ValueError("no face found in any frame")
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
            mask, crop_box = contour_safe_mask(frame, (x1, y1, x2, y2), self.fp)
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
        # windows centred on each frame's middle as in training (lipsync.py); MuseTalk's get_whisper_chunk
        # centred them 45 ms early at our 20 fps
        hidden = self.whisper.encoder(feats.to(self.device, self.dtype), output_hidden_states=True).hidden_states
        f = torch.stack(hidden, dim=2)[:, : math.floor(len(audio) / 16000 * 50)]       # [1, T, 5, 384]
        pad = torch.zeros_like(f[:, :1]).expand(-1, lipsync.PAD, -1, -1)
        f = torch.cat([pad, f, pad], 1)
        n = lipsync.frame_count(len(audio), self.fps)
        if n <= 0:
            return torch.zeros(0, lipsync.WINDOW * f.shape[2], f.shape[3], device=self.device, dtype=self.dtype)
        idx = torch.tensor(lipsync.window_starts(n, self.fps), device=self.device)[:, None] \
            + torch.arange(lipsync.WINDOW, device=self.device)
        return f[0, idx.clamp_(0, f.shape[1] - 1)].flatten(1, 2)                     # [n, 50, 384]

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
        if self.tiny is not None:                     # TAESD decodes the UNet's (scaled) latents directly
            img = self.tiny.decode(pred[:, :, DECODE_FROM_ROW:, :]).sample
        else:
            lat = pred[:, :, DECODE_FROM_ROW:, :] / self.vae.scaling_factor
            img = self.vae.vae.decode(lat.to(self.vae.vae.dtype)).sample
        img = (img.float() / 2 + 0.5).clamp(0, 1)
        if self.sr is not None and self.use_sr:      # 256 -> 512 wide, before it is pasted into a large face box
            up = F.interpolate(img, scale_factor=2, mode="bicubic", align_corners=False)
            img = (SR_DETAIL * self.sr.tensor(img).float() + (1 - SR_DETAIL) * up).clamp(0, 1)
        img = img.mul(255).round().to(torch.uint8)
        return idx, img.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]   # RGB → BGR

    def compose(self, loop: Loop, i: int, face_bottom: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
        """CPU stage: paste the generated lower face into loop frame i (MuseTalk blending).
        ``background``: another picture of frame i to paste into (two expressions blended) - one paste,
        so the mouth cannot appear twice.

        The lower face is pasted at the frame's own resolution and the rest of the face box is left as it
        is (it used to go through a 256x256 copy and back). The lip colour correction (refine_face) works
        on the 256-wide picture as before; its change is smooth, so it is carried over enlarged."""
        frame = loop.frame(i) if background is None else background.copy()
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = loop.coords[i]
        w, h = x2 - x1, y2 - y1
        rows = face_bottom.shape[0] * 256 // face_bottom.shape[1]          # rows of the 256x256 crop it covers
        top = y1 + round(h * (256 - rows) / 256)                            # first frame row of the lower face
        small = face_bottom if face_bottom.shape[1] == 256 else cv2.resize(face_bottom, (256, rows), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(frame[top:y2, x1:x2], (256, rows), interpolation=cv2.INTER_AREA)
        refined = refine_face(np.ascontiguousarray(small), ref, self.sharpen if face_bottom.shape[1] == 256 else 0.0)
        interp = cv2.INTER_AREA if face_bottom.shape[1] >= w else cv2.INTER_CUBIC
        gen = cv2.resize(face_bottom, (w, y2 - top), interpolation=interp).astype(np.float32)
        delta = refined.astype(np.float32) - small.astype(np.float32)
        gen = np.clip(gen + cv2.resize(delta, (w, y2 - top), interpolation=cv2.INTER_LINEAR), 0, 255).astype(np.uint8)
        # MuseTalk's crop box (1.5x the face box) may reach past the frame edges: clip it, and the mask with it
        # (frame[ys:ye] with a negative ys wrapped around, and the paste failed)
        xs, ys, xe, ye = loop.mask_boxes[i]
        cx0, cy0, cx1, cy1 = max(xs, 0), max(ys, 0), min(xe, fw), min(ye, fh)
        body = frame[cy0:cy1, cx0:cx1]
        large = body.copy()
        py0, py1 = max(top, cy0), min(y2, cy1)
        px0, px1 = max(x1, cx0), min(x2, cx1)
        large[py0 - cy0:py1 - cy0, px0 - cx0:px1 - cx0] = gen[py0 - top:py1 - top, px0 - x1:px1 - x1]
        m = loop.mask(i)[cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs].astype(np.float32)[..., None] / 255.0
        hh, ww = min(large.shape[0], m.shape[0]), min(large.shape[1], m.shape[1])
        frame[cy0:cy0 + hh, cx0:cx0 + ww] = (large[:hh, :ww] * m[:hh, :ww] + body[:hh, :ww] * (1 - m[:hh, :ww])).astype(np.uint8)
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
