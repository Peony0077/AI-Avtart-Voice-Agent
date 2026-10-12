# Third-party notices

Components this project uses, and their licenses. The project began as a personal, non-commercial
project; with lead generation and the customer link it is now used for a business, so read the next
section first.

## Licences and commercial use

Most components are permissively licensed (MIT, Apache-2.0, BSD). **Two are licensed for non-commercial
use only, and the standard setup uses both:**

| Component | License | Where it is used | Before commercial use |
|---|---|---|---|
| **InsightFace buffalo_l** (`det_10g.onnx` face detector, `2d106det.onnx` landmarks; shipped in the KlingTeam/LivePortrait weights) | non-commercial research only | LivePortrait's face cropper (`avatar/musetalk/face_rig.py`) in **every avatar build** - `build_avatar.py` from a photo and from a video. Every avatar shown by the `avatar` and `avatar_clone` profiles was rendered with it (the live worker does not run it) | replace the cropper's detector with a commercially usable one and rebuild the avatars, or obtain a commercial license from InsightFace |
| **XTTS-v2** (Coqui Public Model License, CPML) | non-commercial only | the `clone` profile (TTS in `voice/xtts/worker.py`); `scripts/pick_clone_voice.py` - its speaker encoder scores the Kokoro voices, a step of the **`avatar_clone`** setup; `voice/xtts/bench_clone.py` (benchmarks) | do not use the `clone` profile; for `avatar_clone` choose the Kokoro base voice by ear instead of running `pick_clone_voice.py`. Coqui closed in 2024, so a commercial license may not be obtainable |

Profiles `default`, `lite` and the voice part of `avatar` / `avatar_clone` (Kokoro, OpenVoice converter,
Whisper, Qwen, bge) use only permissively licensed models.

Also check before a business use:

- **Likeness and voice** (below): the development photo and voice are for development only; use a real
  person's face or voice only with their written consent.
- **LiteAvatar weights** (renderer comparison only, not part of the app): no license stated.
- **Cloudflare quick tunnels** (`scripts/tunnel.ps1`, `scripts/tunnel.sh`): Cloudflare offers them for
  testing and development; serve customers through a tunnel of your own Cloudflare account or your own
  HTTPS server, under Cloudflare's terms.
- **Copyleft programs used unmodified**: espeak-ng (GPL-3.0) as a separate library, ffmpeg with x264
  (GPL-2.0+) as an external program. This project does not ship them; their terms apply mainly when they
  are distributed (for example bundled in an installer or an appliance).

This is a summary, not legal advice.

## Code reused or adapted

| Component | Where | License | Notes |
|---|---|---|---|
| OpenAvatarChat (HumanAIGC-Engineering) | `avatar/musetalk/face_bbox.py`, ideas in `avatar/musetalk/engine.py` | Apache-2.0 | DWPose-via-ONNX preprocessing and MuseTalk real-time loop design adapted; modified. |
| MuseTalk (TMElyralab) | `avatar/musetalk/vendor/MuseTalk` (fetched by `setup.ps1` / `setup.sh`, not committed) | MIT | Pinned at `0a89dec`. |
| LivePortrait (KlingAIResearch) | `avatar/musetalk/vendor/LivePortrait` (fetched, not committed); `face_rig.py` mirrors its retargeting composition | MIT | Pinned at `9b294b3`. |
| LiteAvatar (HumanAIGC) — renderer comparison only | `avatar/liteavatar/vendor/lite-avatar` (fetched by `setup.ps1`, not committed; includes FunASR code, MIT) | MIT | Pinned at `5b7ec85`; weights (Paraformer, audio2mouth, sample avatar) from ModelScope LiteAvatarGallery — license not stated [unverified]; used for a local benchmark only. |
| OpenVoice (MyShell.ai) | `backend/app/tts/openvoice/` (models, modules, attentions, commons, transforms) | MIT | Vendored at `74a1d14`, only imports changed; license text in that folder. `converted_tts.py` re-implements its converter API without librosa/watermarking. |
| Real-ESRGAN (xinntao) / BasicSR (XPixelGroup) | `avatar/musetalk/upscale.py` | BSD-3-Clause / Apache-2.0 | The RRDBNet and SRVGGNetCompact network definitions follow theirs so the published weights load; BasicSR itself is not used. |

## Models (downloaded to MODEL_HOME - `E:\AI-models` / `~/AI-models` - never committed)

Pinned to a commit (`scripts/download_models.py`, `avatar/musetalk/download_weights.py`) or a SHA-256.

| Model | License | Commercial? |
|---|---|---|
| Qwen3-4B-Instruct-2507 (GGUF by unsloth) | Apache-2.0 | yes |
| Whisper large-v3-turbo (OpenAI) | MIT | yes |
| whisper-small.en (OpenAI) — `lite` profile | Apache-2.0 | yes |
| Kokoro-82M incl. voices (hexgrad) | Apache-2.0 | yes |
| **XTTS-v2 (Coqui)** — `clone` profile, `pick_clone_voice.py` (`avatar_clone` setup) | **Coqui Public Model License (CPML)** | **no** — non-commercial only |
| OpenVoice V2 tone-colour converter (MyShell.ai) — `avatar_clone` profile | MIT | yes |
| Silero VAD | MIT | yes |
| bge-small-en-v1.5 (BAAI) | MIT | yes |
| MuseTalk 1.5 UNet | MIT | yes |
| sd-vae-ft-mse (Stability AI) | MIT | yes |
| TAESD tiny autoencoder (madebyollin/taesd) — MuseTalk's default latent decoder | MIT | yes |
| Real-ESRGAN weights `RealESRGAN_x2plus`, `realesr-general-x4v3` (xinntao) — avatar sharpening | BSD-3-Clause | yes |
| whisper-tiny (OpenAI) | Apache-2.0 | yes |
| DWPose (yzd-v) | Apache-2.0 | yes |
| face-parse-bisent | WTFPL | yes |
| S3FD (face-alignment) | BSD-3-Clause | yes |
| LivePortrait weights | MIT | yes |
| **InsightFace buffalo_l (used by LivePortrait's cropper in every avatar build)** | **non-commercial research only** | **no** — replace the detector and rebuild the avatars before any commercial use |
| en_core_web_sm (spaCy) | MIT | yes |

## Programs downloaded or called (not committed)

| Program | License | How it is used |
|---|---|---|
| llama.cpp `llama-server` (ggml-org), prebuilt release b11242 | MIT | separate process (`scripts/get_llama_cpp.ps1`, `scripts/setup_linux.sh` → `tools/llama.cpp`) |
| NVIDIA CUDA runtime (cudart, cuBLAS) bundled with that llama.cpp release | NVIDIA CUDA Toolkit EULA (redistributable runtime libraries) | loaded by `llama-server` only |
| cloudflared (Cloudflare) 2026.10.0 | Apache-2.0 | separate process for the quick tunnel (`scripts/tunnel.ps1` / `tunnel.sh` → `tools/cloudflared`) |
| ffmpeg with libx264 | LGPL-2.1+ / GPL-2.0+ (x264 builds are GPL) | external program installed by the user, called for the avatar's H.264 video; not linked or shipped |

## Runtimes / libraries (selection)

PyTorch (BSD-3) · transformers / diffusers / accelerate (Apache-2.0) ·
FastAPI (MIT) · uvicorn (BSD-3) · onnxruntime (MIT) · OpenCV (Apache-2.0) · espeak-ng via
espeakng-loader (GPL-3.0, used as a separate library by Kokoro's G2P) · React (MIT) · Vite (MIT) ·
Playwright (Apache-2.0) · LanceDB (Apache-2.0) · sentence-transformers (Apache-2.0) ·
SciPy (BSD-3) · cryptography (Apache-2.0 / BSD) · pypdf (BSD-3) · python-docx (MIT) ·
coqui-tts (MPL-2.0, XTTS worker venv only, unmodified) · torchcodec (BSD-3, XTTS worker venv) ·
diffusers' AutoencoderTiny (Apache-2.0, loads TAESD).

## Likeness and voice

The development avatar photo (`girl.png`) comes from OpenAvatarChat's resources (Apache-2.0 repository)
and appears to be AI-generated. The development "cloned" voice (`data/voices/dev_emma.wav`) is Kokoro's
stock voice bf_emma, so no real person's voice is cloned. Use only photos and voices of people who have
agreed to it.
