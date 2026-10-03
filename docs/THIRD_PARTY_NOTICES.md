# Third-party notices

This is a personal, non-commercial project. Components and their licenses:

## Code reused or adapted

| Component | Where | License | Notes |
|---|---|---|---|
| OpenAvatarChat (HumanAIGC-Engineering) | `avatar/musetalk/face_bbox.py`, ideas in `avatar/musetalk/engine.py` | Apache-2.0 | DWPose-via-ONNX preprocessing and MuseTalk real-time loop design adapted; modified. |
| MuseTalk (TMElyralab) | `avatar/musetalk/vendor/MuseTalk` (fetched by `setup.ps1`, not committed) | MIT | Pinned at `0a89dec`. |
| LivePortrait (KlingAIResearch) | `avatar/musetalk/vendor/LivePortrait` (fetched, not committed); `face_rig.py` mirrors its retargeting composition | MIT | Pinned at `9b294b3`. |
| LiteAvatar (HumanAIGC) — renderer comparison only | `avatar/liteavatar/vendor/lite-avatar` (fetched by `setup.ps1`, not committed; includes FunASR code, MIT) | MIT | Pinned at `5b7ec85`; weights (Paraformer, audio2mouth, sample avatar) from ModelScope LiteAvatarGallery — license not stated [unverified]; used for a local benchmark only. |
| OpenVoice (MyShell.ai) | `backend/app/tts/openvoice/` (models, modules, attentions, commons, transforms) | MIT | Vendored at `74a1d14`, only imports changed; license text in that folder. `converted_tts.py` re-implements its converter API without librosa/watermarking. |

## Models (downloaded to `E:\AI-models`, never committed)

| Model | License | Commercial? |
|---|---|---|
| Qwen3-4B-Instruct-2507 (GGUF by unsloth) | Apache-2.0 | yes |
| Whisper large-v3-turbo, whisper-small.en (OpenAI) | MIT | yes |
| Kokoro-82M incl. voices (hexgrad) | Apache-2.0 | yes |
| **XTTS-v2 (Coqui)** — `clone` profile | **Coqui Public Model License (CPML)** | **no** — non-commercial only |
| OpenVoice V2 tone-colour converter (MyShell.ai) — `avatar_clone` profile | MIT | yes |
| Silero VAD | MIT | yes |
| bge-small-en-v1.5 (BAAI) | MIT | yes |
| MuseTalk 1.5 UNet | MIT | yes |
| sd-vae-ft-mse (Stability AI) | MIT | yes |
| whisper-tiny (OpenAI) | Apache-2.0 | yes |
| DWPose (yzd-v) | Apache-2.0 | yes |
| face-parse-bisent | WTFPL | yes |
| S3FD (face-alignment) | BSD-3-Clause | yes |
| LivePortrait weights | MIT | yes |
| **InsightFace buffalo_l (used by LivePortrait's cropper)** | **non-commercial research only** | **no** — replace the detector before any commercial use |
| en_core_web_sm (spaCy) | MIT | yes |

## Runtimes / libraries (selection)

llama.cpp (MIT) · PyTorch (BSD-3) · transformers / diffusers / accelerate (Apache-2.0) ·
FastAPI (MIT) · uvicorn (BSD-3) · onnxruntime (MIT) · OpenCV (Apache-2.0) · espeak-ng via
espeakng-loader (GPL-3.0, used as a separate library by Kokoro's G2P) · React (MIT) · Vite (MIT) ·
Playwright (Apache-2.0) · LanceDB (Apache-2.0) · sentence-transformers (Apache-2.0) ·
SciPy (BSD-3) · cryptography (Apache-2.0 / BSD) · pypdf (BSD-3) · python-docx (MIT) ·
coqui-tts (MPL-2.0, XTTS worker venv only, unmodified) · torchcodec (BSD-3, XTTS worker venv) ·
cloudflared (Apache-2.0, downloaded by `scripts/tunnel.ps1`, not committed).

## Likeness and voice

The development avatar photo (`girl.png`) comes from OpenAvatarChat's resources (Apache-2.0 repository)
and appears to be AI-generated. The development "cloned" voice (`data/voices/dev_emma.wav`) is Kokoro's
stock voice bf_emma, so no real person's voice is cloned. Use only photos and voices of people who have
agreed to it.
