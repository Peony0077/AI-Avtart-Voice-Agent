"""H.264 video for the browser (owner, 2026-10-05: "delay and non-smooth" through the tunnel).

The avatar used to go out as one JPEG per frame: 41 KB x 20 fps = 6.7 Mbit/s even while idle. Through
the internet tunnel this PC's link carried ~4.7 Mbit/s, so frames queued, the voice (same WebSocket)
waited behind them and a reply started playing seconds late - or not at all. H.264 at ~1 Mbit/s looks
the same here (PSNR 38-39 dB against the rendered frames) at a seventh of the bandwidth, like the video
streams of HeyGen and Anam. The browser decodes it with WebCodecs (AvatarStage).

One ffmpeg process per stream (NVENC, else libx264; High profile, no B-frames): raw BGR frames in, FLV out.
Encoder (measured 2026-10-10, 1024x1024 at 25 fps, 2.5 Mbit/s, PSNR against the source): libx264 ultrafast
baseline (the first setting: no deblocking, CAVLC) 45.3 dB; libx264 veryfast High 47.8 dB; NVENC p4 High
49.8 dB at under one CPU core (x264: 3-8 cores). NVENC at 1.5 Mbit/s (48.5 dB) beats ultrafast at 2.5. FLV gives every frame with its exact length, as soon as it is encoded
(1.9 ms p50 after the frame was written, measured; MP4 and Matroska held every frame until the next one
arrived). The frames are rewritten as Annex-B access units for the browser, SPS and PPS before every
keyframe. A raw H.264 stream was used first, split at access unit delimiters with "no bytes for 4 ms =
the frame is complete": under load the pipe delivered a frame in pieces more than 4 ms apart, a frame
was cut in two, the browser's decoder failed and waited for the next keyframe - the end of a short
reply never appeared and the picture froze for 0.9 s (e2e phase6_speech_end).
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
from loguru import logger

START_CODE = b"\x00\x00\x00\x01"


@functools.cache
def ffmpeg_exe() -> str | None:
    """ffmpeg on PATH, else the one bundled with imageio-ffmpeg (avatar venv), else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def is_keyframe(au: bytes) -> bool:
    """An IDR slice (NAL unit type 5) in the access unit: a decoder can start here."""
    i = 0
    while True:
        i = au.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= len(au):
            return False
        if au[i + 3] & 0x1F == 5:
            return True
        i += 3


class FlvReader:
    """FLV bytes in (any pieces), complete H.264 frames out: ``feed(data)`` -> [(keyframe, annexb_au)]."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._header = False
        self._nal_len = 4
        self._params = b""            # SPS and PPS (Annex-B), put before every keyframe

    def feed(self, data: bytes) -> list[tuple[bool, bytes]]:
        self._buf += data
        out: list[tuple[bool, bytes]] = []
        if not self._header:
            if len(self._buf) < 9:
                return out
            if self._buf[:3] != b"FLV":
                raise ValueError("not an FLV stream")
            size = int.from_bytes(self._buf[5:9], "big")
            if len(self._buf) < size + 4:
                return out
            del self._buf[:size + 4]                      # header and PreviousTagSize0
            self._header = True
        while len(self._buf) >= 11:
            size = int.from_bytes(self._buf[1:4], "big")
            if len(self._buf) < 11 + size + 4:            # tag header, data, PreviousTagSize
                break
            kind, data = self._buf[0], bytes(self._buf[11:11 + size])
            del self._buf[:11 + size + 4]
            if kind == 9 and len(data) > 5 and data[0] & 0x0F == 7:   # video, AVC
                frame = self._video(data)
                if frame is not None:
                    out.append(frame)
        return out

    def _video(self, data: bytes) -> tuple[bool, bytes] | None:
        key, packet, body = data[0] >> 4 == 1, data[1], data[5:]
        if packet == 0:                                   # AVCDecoderConfigurationRecord
            self._nal_len = (body[4] & 3) + 1
            params, i = bytearray(), 5
            for mask in (0x1F, 0xFF):                     # the SPS (5-bit count), then the PPS
                count, i = body[i] & mask, i + 1
                for _ in range(count):
                    n = int.from_bytes(body[i:i + 2], "big")
                    params += START_CODE + body[i + 2:i + 2 + n]
                    i += 2 + n
            self._params = bytes(params)
            return None
        if packet != 1:
            return None
        au, i, n = bytearray(self._params if key else b""), 0, self._nal_len
        while i + n <= len(body):
            size = int.from_bytes(body[i:i + n], "big")
            au += START_CODE + body[i + n:i + n + size]
            i += n + size
        return key, bytes(au)


def encoder_args(encoder: str) -> list[str]:
    """High profile (CABAC, 8x8 transform, deblocking) at level 4.1: frames up to 2048x1024 at 30 fps.
    The browser decodes it (lib/h264.js asks WebCodecs for avc1.640029)."""
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-zerolatency", "1", "-delay", "0",
                "-rc", "cbr", "-profile:v", "high", "-level", "4.1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "high", "-level", "4.1"]


@functools.cache
def pick_encoder(wanted: str = "auto") -> str:
    """"nvenc" when asked for or ("auto") when this ffmpeg and GPU can open an NVENC session, else "libx264"."""
    exe = ffmpeg_exe()
    if wanted == "libx264" or exe is None:
        return "libx264"
    try:
        probe = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=black:s=256x256:r=25:d=0.2", *encoder_args("nvenc"), "-f", "null", "-"],
                               capture_output=True, timeout=20, **_no_window())
        if probe.returncode == 0:
            return "nvenc"
        logger.warning(f"NVENC unavailable ({probe.stderr.decode(errors='replace').strip()[-200:]}): libx264")
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"NVENC probe failed ({exc!r}): libx264")
    return "libx264"


def _no_window() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class H264Stream:
    """One H.264 stream: ``encode(frame, meta)`` in order; ``on_au(meta, access_unit, keyframe)`` is called
    (from a reader thread) for every frame, in the same order. The first frame is a keyframe."""

    def __init__(self, on_au: Callable[[object, bytes, bool], None], size: tuple[int, int] = (512, 512),
                 fps: int = 20, kbps: int = 1000, gop: int = 40, encoder: str = "libx264") -> None:
        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError("ffmpeg not found (PATH or imageio-ffmpeg)")
        w, h = size
        self.size = size
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "pipe:0", *encoder_args(pick_encoder(encoder)), "-b:v", f"{kbps}k",
               "-maxrate", f"{kbps}k", "-bufsize", f"{max(1, kbps // 2)}k", "-g", str(gop), "-bf", "0",
               "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-flush_packets", "1",
               "-flvflags", "no_duration_filesize", "-f", "flv", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0, **_no_window())
        self.stderr_tail: deque[str] = deque(maxlen=20)   # why an encoder stopped (it used to go to DEVNULL)
        threading.Thread(target=self._drain_stderr, daemon=True, name="h264-err").start()
        self.on_au = on_au
        self._meta: deque = deque()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.closed = False
        self._closed_by_us = False
        self.frames_in = 0
        self.frames_out = 0
        self.bytes_out = 0
        threading.Thread(target=self._read, daemon=True, name="h264-read").start()

    def encode(self, frame_bgr: np.ndarray, meta: object = None) -> bool:
        """Blocking (~1 ms): call it from a worker thread. False if the stream is closed."""
        if self.closed:
            return False
        data = np.ascontiguousarray(frame_bgr).tobytes()
        with self._write_lock:
            with self._lock:
                self._meta.append(meta)
            try:
                self.proc.stdin.write(data)
            except (OSError, ValueError):
                self.closed = True
                return False
        self.frames_in += 1
        return True

    def close(self) -> None:
        """No more frames: the encoder writes out what it has and exits (its frames still arrive)."""
        self._closed_by_us = True
        if self.closed:
            return
        self.closed = True
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    @property
    def failed(self) -> bool:
        """The encoder exited on its own (not closed by us): its frames will not come."""
        return self.proc.poll() is not None and not self._closed_by_us

    def _drain_stderr(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            self.stderr_tail.append(line.decode(errors="replace").rstrip())

    def _read(self) -> None:
        flv = FlvReader()
        while True:
            try:
                chunk = self.proc.stdout.read(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            try:
                frames = flv.feed(chunk)
            except ValueError:
                break
            for key, au in frames:
                with self._lock:
                    meta = self._meta.popleft() if self._meta else None
                self.frames_out += 1
                self.bytes_out += len(au)
                try:
                    self.on_au(meta, au, key)
                except Exception:  # noqa: BLE001 - a consumer error must not stop the stream
                    pass
        self.closed = True
        if not self._closed_by_us:
            self.proc.wait()
            logger.warning(f"H.264 encoder exited ({self.proc.returncode}): {' | '.join(self.stderr_tail) or 'no message'}")
