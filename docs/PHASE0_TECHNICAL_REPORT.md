# Phase 0 Technical Report — Self-Hosted Real-Time Conversational Avatar

**Date:** 2026-09-29 · **Machine:** POOH-PC · **Status:** DRAFT FOR APPROVAL
Nothing has been installed and no model weights have been downloaded. Only code repositories (OpenAvatarChat, MuseTalk) were shallow-cloned into a temporary scratch folder for reading, and public metadata (Hugging Face / GitHub APIs) was queried.

**Evidence legend**

| Tag | Meaning |
|---|---|
| **[M]** | Measured on POOH-PC during this investigation |
| **[V]** | Verified from a primary source (repo file, model card, release/API metadata) |
| **[E]** | Estimate, with the reasoning stated |
| **[U]** | Unknown — must be benchmarked or confirmed |

---

## 0. Executive summary

1. **The GPU is shared with the desktop.** ~1.8–1.9 GiB of the 12 GiB is already used by Windows/apps [M], so the real budget is **~10 GiB**. **System RAM is the second hard limit**: 15.7 GiB total with only **1.5–1.9 GiB free** during tests [M].
2. **OpenAvatarChat → recommendation B: fork / reuse components** (plus an optional *sandboxed* reference install later). It cannot be used directly in our environment: it is a single-process, single-environment design pinned to `torch==2.8.0+cu128` and `numpy<=1.26.4` (would replace our working torch), its presets depend on paid Alibaba DashScope/Bailian APIs (including the interrupt judge), tooling is Linux-first, the UI is Vue/TypeScript/Electron, and its authors measured **~2.2 s** average response on an RTX 4090. Its **MuseTalk handler is the most valuable reusable piece**: it replaces MuseTalk's mmcv/mmpose preprocessing with an ONNX DWPose model, removing the Windows compile problem.
3. **MuseTalk 1.5** is commercially clean (code MIT, weights MIT/Apache-2.0/WTFPL) [V] and realistic, but it is a **mouth-region inpainter**: blinking, gaze and head motion come only from a **reference video loop**; a single still image gives a frozen head; there is **no native emotion control**. Expected **~15–21 fps on RTX 3060** standalone [E, from 42 fps on 3080 Ti [V]] → **25 fps is at risk, 20 fps is borderline**. Must be benchmarked.
4. **Avatar benchmark shortlist:** MuseTalk 1.5 (mandatory, native Windows) · **LAM** (single image → Gaussian-splat head rendered *in the browser*, blendshape control of blink/gaze/emotion, near-zero server VRAM) · **Ditto** (single image → audio-driven head motion; TensorRT; WSL2). Secondary: **AVTR-1** (published 206 ms per 5-frame chunk on RTX 3060 Ti ≈ 24 fps, has "listening" behaviour, but custom license + non-commercial InsightFace dependency) and **SoulX-FlashHead Lite** (Apache-2.0, 96 fps on RTX 4090, but needs a flash-attn build, 1.3B DiT, estimated 10–18 fps on 3060 and it renders continuously even when idle).
5. **Recommended live stack for Phases 1–3** (all Apache-2.0 / MIT): Silero VAD (CPU) → Whisper large-v3-turbo via `transformers` (GPU, inside the existing venv) → **Qwen3-4B-Instruct-2507 Q4_K_M** on **llama-server** (prebuilt Windows CUDA binary, separate process) → **Kokoro-82M** (word timestamps) → browser.
6. **Processes:** backend (existing `.venv`, torch 2.9.1+cu130 untouched) · `llama-server.exe` · avatar worker in **its own venv**. Native Windows for all three. WSL2 only for engines that require TensorRT/flash-attn builds — WSL2 shares the same 12 GB GPU and is capped at **7.6 GiB RAM** [M].
7. **Latency:** ≤ 1.2 s end-of-speech → first audio is achievable for the audio-only pipeline with tuning [E]; with a neural avatar expect **1.3–1.8 s** until optimised [E], because audio must wait for its lip-synced video frames.
8. **Top risks:** system RAM (upgrade to 32 GB strongly recommended), MuseTalk fps on a 3060, limited emotion control with lip-sync-only renderers, model licenses (InsightFace, non-commercial TTS), self-echo during barge-in.

---

## 1. Current environment

| Item | Value | Src |
|---|---|---|
| Host / OS | POOH-PC, Windows 11 Pro 10.0.26200 | M |
| CPU | Intel Core i5-12400, 6 cores / 12 threads | M |
| System RAM | 15.7 GiB total; **1.5–1.9 GiB free** during tests (Chrome, Telegram, Slack, Battle.net, WPS Office, Bitdefender, etc.); page file 34 GB | M |
| GPU | RTX 3060 12 GB (12,288 MiB), compute capability 8.6, bf16 supported, WDDM, **drives the display** | M |
| Driver | 610.60, CUDA UMD 13.3 | M |
| VRAM already used | 1,761–1,904 MiB (nvidia-smi). `torch.cuda.mem_get_info` reports 10.99 GiB free — WDDM accounting differs; plan with the conservative figure (~10.2 GiB usable) | M |
| GPU throughput | fp16 4096² matmul: 13.8–14.8 TFLOPS; 6 GiB single allocation succeeded | M |
| Python env | `.venv`: Python 3.11.15 (uv-managed CPython), torch 2.9.1+cu130, torchaudio 2.9.1+cu130, torchvision 0.24.1+cu130, numpy 2.4.6, cuDNN 9.12 | M |
| Build toolchain | **No MSVC (`cl`), no `cmake`, no `nvcc`** | M |
| Tools present | Node 24.11.1, npm 11.6.2 (nvm4w), ffmpeg 9.0, git 2.52, git-lfs, uv 0.12.4 | M |
| Not present | Ollama, Docker, espeak-ng | M |
| Disk | One 1 TB NVMe SSD (Dahua C900 PLUS); partitions C: 124 GB free, D: 51, E: 135, F: 24 | M |
| WSL2 | Ubuntu 24.04.3, kernel 6.18; RAM cap **7.6 GiB** (no `.wslconfig`); sees the same RTX 3060; Python 3.12.3; no CUDA toolkit/uv/conda; **NAT** networking. Was stopped before and after the check | M |
| Audio devices | USB Audio Device, Realtek, NVIDIA HDMI, virtual cables (VB-Audio, Voicemod, Voice.ai) | M |
| Remote access | AnyDesk running. A remote viewer's microphone is *not* this PC's microphone | M |
| MCP | No MCP servers configured for this project; no Playwright MCP in the current session | M |
| Repository | 2 commits (`55dea38`, `a01765a added gitignore`), working tree clean before this report. `README.md` is **UTF-16LE** (git treats it as binary). `.gitignore` exists but does not yet include `weights/`, `checkpoints/`, `cache/`, or unanchored `models/` and `dist/` | M |

**Implications:** every design decision below assumes ~10 GiB VRAM, ≤ 3 GPU processes, no source compilation on Windows, and that other desktop apps must be closed for demos until RAM is upgraded.

---

## 2. OpenAvatarChat assessment

**Repository:** [HumanAIGC-Engineering/OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) — **Apache-2.0** [V], v0.6.0 (2026-04), last push 2026-07-31, ~3.8k stars [V].

### 2.1 What it already solves [V — read from source]

| Area | Implementation | Key files |
|---|---|---|
| Handler engine | YAML-configured handlers (client, VAD, ASR, LLM/agent, TTS, avatar, logic, manager). Lifecycle: `load → create_context → start_context → handle / on_signal → destroy_context` | `src/chat_engine/common/handler_base.py`, `core/handler_manager.py` |
| Concurrency | **One Python process**; every handler gets a pump thread + `queue.Queue`; RTC client uses asyncio queues | `core/chat_session.py`, `handlers/client/rtc_client/client_handler_rtc.py` |
| Stream graph + cancellation | `ChatStream` objects with ancestry; `cancel_stream_chain` cancels a stream and everything derived from it — functionally a turn-scoped cancel | `core/stream_manager.py` |
| Interruption | Dedicated `InterruptHandler` receives INTERRUPT signals (client or handler), cancels `CLIENT_PLAYBACK` chains, logs to history. "Duplex" Silero VAD keeps running during playback. Manual + duplex interrupt for all avatars since v0.6.0 | `handlers/logic/interrupt/interrupt_handler.py`, `handlers/vad/silerovad/duplex_vad_handler.py` |
| Interrupt decision | `SemanticTurnDetector` asks an LLM (`qwen-plus` on **DashScope** in presets, 3 s timeout) whether new speech is a real interruption | `handlers/llm/semantic_turn_detector/` |
| End-of-turn | pipecat **smart-turn v3.1** ONNX on CPU | `handlers/vad/smart_turn_eou/` |
| ASR | SenseVoiceSmall (local), Bailian ASR (cloud) | `handlers/asr/` |
| LLM | OpenAI-compatible (can target a local llama-server), Dify, Qwen-Omni (cloud), Agent (beta, MCP tools) | `handlers/llm/`, `handlers/agent/` |
| TTS | CosyVoice local, Bailian CosyVoice (cloud), Edge TTS (Microsoft online service — free but not self-hosted) | `handlers/tts/` |
| Avatars | LiteAvatar (2D), LAM (client-side Gaussian WebGL), MuseTalk 1.5, SoulX-FlashHead Lite, "without_avatar" | `handlers/avatar/` |
| Transport | WebRTC via aiortc + a fastrtc/gradio-webrtc fork; coturn TURN; self-signed SSL. Also WebSocket clients: `ws_client` (binary protocol + Opus) and `ws_lam_client` | `handlers/client/`, `service/rtc_service/` |
| Measured latency (authors) | **~2.2 s** average end-of-speech → avatar speech, i9-13900KF + **RTX 4090**, including RTC and VAD delay | `docs/en/guide/how-it-works.md` |

### 2.2 Dependencies and platform [V — `pyproject.toml` and handler pyprojects]

- Python `>=3.11.7,<3.12`; **`torch==2.8.0`, `torchvision==0.23.0`, `torchaudio==2.8.0` from the cu128 index**; **`numpy<=1.26.4`**; `gradio~=5.9.1`; `aiortc>=1.12`; `onnxruntime-gpu~=1.20.2`; `protobuf<5`; `setuptools<70`. Requires a driver that supports CUDA ≥ 12.8 (ours does).
- Handler extras: MuseTalk → `diffusers~=0.30.2`, `transformers~=4.44.1`, `tensorflow~=2.16.1` (not imported by the handler code — looks vestigial), `onnxruntime-gpu`. FlashHead → **`flash-attn>=2.6` (compiled by `install.py`)**, `xformers`, `xfuser`, `mediapipe`. CosyVoice local → **`pynini==2.1.6` (fails on Windows via PyPI; docs say use Conda)**, `openai-whisper`, `lightning`. LiteAvatar → `funasr`, `transformers==4.44.1`.
- `install.py` merges all selected handler dependencies into **one** uv environment.
- **Windows:** documentation is Linux-first (`apt`, bash scripts, CUDA 12.8 Docker image, coturn script). Windows is mentioned only for installing uv, the pynini workaround, and a third-party one-click package. Docker is optional.
- **GPU:** FlashHead preset sets `concurrent_limit: 1` ("high VRAM"); LiteAvatar ≈ 3 GB VRAM per session; other VRAM figures undocumented [U].
- **WebUI:** [OpenAvatarChat-WebUI](https://github.com/HumanAIGC-Engineering/OpenAvatarChat-WebUI) is **Vue 3 + TypeScript + Electron** (ant-design-vue); LAM rendering uses the npm package `gaussian-splat-renderer-for-lam` [V].

### 2.3 Recommendation: **B — FORK / REUSE COMPONENTS**

| Reason | Detail |
|---|---|
| Environment conflict | Installing it into our `.venv` would replace torch 2.9.1+cu130 with 2.8.0+cu128 and downgrade numpy 2.4.6 → ≤1.26.4. Forbidden by our rules. |
| No dependency/process isolation | All handlers share one process and one env, so MuseTalk/FlashHead pins must coexist with ASR/TTS pins, and one crash ends the session. We need the avatar in its own process/venv. |
| Paid APIs in presets | Default LLM, TTS and the **semantic interrupt judge** call DashScope. Local replacements exist for LLM/ASR/TTS, but the interrupt judge adds an LLM round trip (≤ 3 s timeout) to every barge-in. We will use a local heuristic instead. |
| Latency | 2.2 s average on a 4090 vs our 1.2 s target on a 3060 → we need tighter streaming (clause-level TTS, prompt caching, explicit turn state). |
| Frontend | Vue/TS/Electron vs our React + Vite + JavaScript requirement. |
| Platform | Linux-first tooling; flash-attn and pynini builds fail on this Windows machine. |
| License | Apache-2.0 allows reuse with attribution and NOTICE — reuse is legally straightforward. |

**Reuse (with attribution):**
- `handlers/avatar/musetalk/*` — ONNX DWPose preprocessing (no mmcv/mmpose), `MuseTalkAlgoV15` (fp16, warm-up, UNet/VAE pipelined threads, idle frames, blending profiler), fps ↔ sample-rate alignment validator.
- `duplex_vad_handler.py` parameters (speaking threshold 0.25, start delay 1024 samples, etc.) as a starting point.
- smart-turn EOU integration.
- Stream-chain cancellation design → our `turn_id` + cancel scope.
- `ws_client` binary protocol / Opus codec ideas; `ws_lam_client` + LAM renderer for the LAM benchmark.
- FlashHead streaming processor (motion-frame chunking, frame metronome) if FlashHead ever becomes viable.

**Do not reuse:** gradio/fastrtc UI, DashScope/Bailian handlers, single-env `install.py`.

**Optional later:** a *sandboxed* OpenAvatarChat install (own venv under `tools/oac/`, never the main env), pointed at our local llama-server, as an A/B reference. Needs large downloads → wait for approval.

---

## 3. MuseTalk assessment

| Item | Value | Src |
|---|---|---|
| Repository | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk), ~6.6k stars. **Last code push 2025-09-26**; HF weights repo updated 2026-09-22 | V |
| Version | 1.5 (released 2025-03-28) | V |
| Code license | MIT — README: "no limitation for both academic and commercial usage" | V |
| Weight licenses | MuseTalk UNet: MIT (HF card) · sd-vae-ft-mse: MIT · whisper-tiny: Apache-2.0 · DWPose: Apache-2.0 · face-parse-bisent: WTFPL · S3FD face detector: via face-alignment (BSD-3) [E — verify] · SyncNet (LatentSync, OpenRAIL++) is **training only**, not needed | V |
| Official environment | Python 3.10, torch 2.0.1+cu118, mmengine / mmcv 2.0.1 / mmdet 3.1.0 / mmpose 1.1.0 via `mim`, numpy 1.23.5, tensorflow 2.12, diffusers 0.30.2, transformers 4.39.2 | V |
| Where mmcv/mmpose are needed | Only `musetalk/utils/preprocessing.py` and `scripts/preprocess.py` (avatar **preparation**). The real-time loop needs torch + diffusers + transformers (`WhisperModel`) + opencv | V (grep) |
| OpenAvatarChat variant | Runs DWPose through ONNX Runtime (`dw-ll_ucoco_384.onnx`, 128 MiB) → **no mmcv**; torch 2.8.0+cu128 | V |
| Real-time mechanics | Preparation: video → frames → face bbox/landmarks → face-parsing masks → cached VAE latents; the loop plays frames forward then reversed. Per frame: whisper-tiny features → UNet single step (fp16) → VAE decode 256×256 → blend into the source frame. **Idle = replay source frames, no inference** | V |
| Published fps | 30+ fps on Tesla V100; [LiveTalking](https://github.com/lipku/LiveTalking): **42 fps 3080 Ti, 45 fps 3090, 72 fps 4090**, recommends 3080 Ti or better | V |
| RTX 3060 fps | **~15–21 fps** standalone (3080 Ti ≈ 2–2.4× a 3060 in compute and memory bandwidth) | E |
| VRAM | UNet 3.24 GB fp32 file → ~1.6 GiB fp16; plus VAE, whisper-tiny, CUDA context, batch activations → **2.8–3.8 GiB** | E |
| System RAM | Cached reference frames/masks: 20 s of 720p ≈ 1.4 GB of frames alone → **2–3 GiB** (use ≤ 512p and 10–20 s loops) | E |
| Download for inference | ~4 GB: v1.5 UNet 3.24 GB + VAE 0.32 GB + whisper-tiny 0.15 GB + DWPose ONNX 0.13 GB + face-parse 0.09 GB + S3FD ≈ 0.09 GB (v1.0 UNet, another 3.24 GB, is not needed) | V sizes / E total |

**Capabilities vs our requirements**

| Requirement | MuseTalk 1.5 |
|---|---|
| Real-time lip sync | Yes on stronger GPUs; **on a 3060 must be benchmarked (25 fps at risk)** |
| Single image initialisation | Technically works, but the head is frozen except the mouth → **not acceptable** for our goal |
| Reference video initialisation | Yes — the intended input (10–30 s, 25 fps, frontal face, neutral, natural blinks/small nods) |
| Blink / gaze / head motion | Only as recorded in the reference loop |
| Emotion / expression control | **None natively.** Workaround: several reference loops (neutral, friendly, concerned, listening, thinking) cross-faded by the BehaviorManager |
| Upper body | Whatever is in the reference video (static loop) |
| Streaming audio input | Yes via the OpenAvatarChat processor (chunked; fps must divide 24000, e.g. 20/24/25) |
| Latency | Batch 2 → first frame ≈ 100–150 ms after audio arrives [E] |
| Long-session stability | Unknown on Windows [U] → 30-minute soak test |
| Coexistence with other models | Needs its **own venv/process** (numpy ≤1.26, diffusers 0.30, transformers 4.44). VRAM is tight (see §11) |

**Conclusion:** MuseTalk is a strong, license-clean **lip-sync layer**, but it is not a complete "single image → living face" solution on its own. The cleanest high-quality path is to **record a 20–30 s reference video of the person** (license-clean, best realism) and generate MuseTalk's mouth region on top. Single-image support would need an offline motion generator to produce the loops (§4, hybrid H2), subject to that generator's license.

**Runtime:** native Windows, `avatar/musetalk/.venv` (Python 3.11 + torch 2.8.0+cu128 prebuilt wheels, the combination OpenAvatarChat tested), preprocessing via ONNX DWPose. WSL2 is the fallback. Do **not** use the official mmcv/`mim` path on Windows.

---

## 4. Alternative avatar comparison

### 4.1 Facts (repository, license, platform)

| Engine | Repo · last push | Code license | Weights license | Commercial use | OS / runtime | Download |
|---|---|---|---|---|---|---|
| **MuseTalk 1.5** | TMElyralab/MuseTalk · 2025-09 | MIT | MIT (+MIT/Apache/WTFPL deps) | OK (verify S3FD) | **Windows native** (ONNX preprocessing path) | ~4 GB |
| **LAM + Audio2Expression** | aigc3d/LAM · 2026-06; aigc3d/LAM_Audio2Expression · 2025-10 | Apache-2.0 | audio2exp Apache-2.0; LAM-20K: no license tag on HF [U]; wav2vec2-base Apache-2.0 | Likely OK (confirm LAM-20K) | Asset generation: Linux, Windows installer claimed, CUDA extensions; or their HF/ModelScope Space. Runtime: **browser WebGL** + small server model | 2.2 GB (generation) + 0.38 + 0.7 GB |
| **Ditto** | antgroup/ditto-talkinghead · 2025-11 | Apache-2.0 | HF card has no license tag [U] | Confirm | Linux; TensorRT 8.6.1 engines (Ampere+ prebuilt); PyTorch path exists | 4.5 GB |
| **AVTR-1** | avaturn-live/avtr-1 · 2026-05 | AVTR-1 Community License | same | Only if annual revenue < USD 10M; **InsightFace parts non-commercial** | Linux (Pixi), CUDA 12 + TensorRT 10 | 1.8 GB (gated) |
| **SoulX-FlashHead Lite** | Soul-AILab/SoulX-FlashHead · 2026-05 | Apache-2.0 | Apache-2.0 | OK | Linux; Python 3.10, torch 2.7.1 cu128, **flash-attn 2.8.0.post2 required** | 13.3 GB repo (Lite subset smaller) |
| **LiteAvatar** | HumanAIGC/lite-avatar · 2025-06 | MIT | Gallery avatars (ModelScope) [U] | Likely OK | Windows native; **CPU 30 fps** | small |
| **LivePortrait / FasterLivePortrait / JoyVASA** | KlingAIResearch/LivePortrait · 2026-06; warmshao/FasterLivePortrait (MIT) · 2025-06; jdh-algo/JoyVASA (MIT) · 2026-04 | MIT (LivePortrait LICENSE: NOASSERTION on API) | LivePortrait MIT, but uses **InsightFace** detection models (non-commercial) | Blocked for commercial use unless InsightFace is licensed/replaced | Windows one-click package (video-driven only; not audio-driven); FasterLivePortrait uses TensorRT | ~2 GB |
| **EchoMimic V3** | antgroup/echomimic_v3 · 2026-03 | Apache-2.0 | Apache-2.0 | OK | Linux; 1.3B Wan DiT, **offline** | 6.7 GB |
| **Hallo 2/3** | fudan-generative-ai | MIT | MIT | OK | Linux; diffusion, **offline** | 12 GB |
| **LiveAvatar (Quark)** | Alibaba-Quark/LiveAvatar | Apache-2.0 | Apache-2.0 | OK | 14B diffusion, **single 80 GB GPU** | — |
| **Wav2Lip** | Rudrabha/Wav2Lip | Custom | **Non-commercial** | ❌ | — | — |

### 4.2 Capabilities and performance

| Engine | Input | Single image | Head motion | Blink | Gaze | Emotion control | Upper body | Output | Published speed | RTX 3060 (est.) | VRAM (est.) | Coexists with LLM+STT+TTS? | Integration | OAC handler |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MuseTalk 1.5 | ref. video + audio | ✗ frozen head | from video | from video | from video | ✗ (loop switching) | from video | 256² face in source frame | 42 fps @ 3080 Ti [V] | **15–21 fps** | 2.8–3.8 GiB | Tight | Medium (reuse OAC code) | ✓ |
| LAM | 1 image → 3D Gaussian head | ✓ | ✓ (pose) | ✓ (ARKit) | ✓ (eye blendshapes) | ✓ (blendshapes) | ✗ head only | browser canvas | 110+ fps WebGL on a phone [V] | 60 fps in browser [E] | server 0.3–1 GiB | **Easy** | Medium-high (asset generation, renderer) | ✓ (client-side) |
| Ditto | 1 image + audio | ✓ | ✓ | ✓ [E] | [U] | partial [U] | ✗ | [U] | "real-time" on high-end GPUs [U] | 10–20 fps with TRT | 3–5 GiB | Tight | High (TRT, WSL2) | ✗ |
| AVTR-1 | 1 image + speech + listening audio | ✓ | ✓ | [U] | [U] | ✗ | ✗ | [U] | 206 ms / 5 frames on 3060 **Ti** (≈24 fps); 232 ms on 4060 [V] | **18–22 fps** | [U] | Tight | High (TRT, WSL2, license) | ✗ |
| FlashHead Lite | 1 image + audio | ✓ | ✓ | ✓ | [U] | ✗ | ✗ | 512² [V] | 96 fps @ 4090 [V] | **10–18 fps**, and busy even when idle | 6–9 GiB | ✗ unlikely | High (flash-attn, WSL2) | ✓ |
| LiteAvatar | gallery avatar only | ✗ (no public creator) | subtle | ✓? | — | ✗ | ✗ | [U] | 30 fps on CPU [V] | 25–30 fps CPU | 0 (CPU) / ~3 GB GPU [V] | ✓ | Low | ✓ |
| LivePortrait + JoyVASA | image + driving video / audio→motion | ✓ | ✓ | ✓ | ✓ | ✓ (expression retarget) | ✗ | 512² | real-time claims with TRT [U] | 15–25 fps (TRT) | 2–4 GiB | Tight | High | ✗ |
| EchoMimic V3 / Hallo / LiveAvatar | — | ✓ | ✓ | ✓ | — | — | partial | — | offline | < 2 fps | > 10 GiB | ✗ | — | ✗ |

The RTX 3060 estimates scale published numbers by relative compute and bandwidth (4090 ≈ 6× a 3060 on fp16 tensor throughput; 3080 Ti ≈ 2–2.4×). **No winner is declared until benchmarks exist.**

### 4.3 Benchmark shortlist

1. **MuseTalk 1.5** — mandatory; native Windows; best quality-per-VRAM for video realism.
2. **LAM** — full behaviour control (blink/gaze/emotion via blendshapes) with almost no server GPU; realism is 3D Gaussian rather than video.
3. **Ditto** — single image → natural audio-driven head motion; WSL2 + TensorRT.
   - 3b. **AVTR-1** if its license is acceptable (published 3060 Ti numbers, listening behaviour).
   - Optional probe: **FlashHead Lite** in WSL2 — expected to fail real-time on a 3060; worth a quick check only if everything else is unsatisfying.

### 4.4 Hybrid designs

| ID | Design | Pros | Cons |
|---|---|---|---|
| H1 | **Reference-video MuseTalk.** Record 20–30 s of the person; also record short state loops (listening, thinking, speaking-neutral, friendly, concerned). The BehaviorManager cross-fades loops; MuseTalk inpaints the mouth only while SPEAKING. | Most realistic; cheapest GPU; license-clean | Emotion granularity limited to recorded loops; needs a recording session |
| H2 | **Single image → offline motion generator → MuseTalk.** Image → LivePortrait (driving template) or Ditto offline → idle/listening loops → MuseTalk preparation. | Meets the "from one image" goal; runtime cost the same as H1 | Inherits the generator's license (InsightFace for LivePortrait); motion quality varies |
| H3 | **LAM.** Image → Gaussian head asset (offline) → browser renderer; server streams ARKit blendshapes (audio2expression) plus BehaviorManager offsets. | Full blink/gaze/emotion/head control; server VRAM ≈ 0 | Head only; realism below video; asset generation needs Linux or their Space |
| H4 | **Fully generative (Ditto / AVTR-1 / FlashHead).** | Single image, natural motion | Highest GPU cost → likely forces a smaller LLM or CPU STT |

---

## 5. STT recommendation

| Model | License | Languages | Size | Runs on | VRAM (est.) | Latency, ≤ 5 s utterance (est.) | Notes |
|---|---|---|---|---|---|---|---|
| **Whisper large-v3-turbo** (`transformers`) | MIT | ~99 | 1.51 GiB [V] | GPU, in backend venv | 1.8–2.2 GiB | 150–300 ms | Uses existing torch cu130 → no DLL mixing. **Phase 2 default.** |
| Whisper small | Apache-2.0 | ~99 | 0.9 GiB fp16 | GPU/CPU | 0.6–0.8 GiB | 100–200 ms GPU | Lower accuracy; VRAM saver |
| distil-large-v3.5 | MIT | English | 1.5 GiB | GPU | ~1.7 GiB | 100–250 ms | English only |
| **Parakeet-TDT-0.6B-v3** (sherpa-onnx / onnx-asr) | CC-BY-4.0 (attribution) | 25 European | 3.0 GiB [V] | **CPU** ONNX | 0 | 200–400 ms (CPU) | Frees ~2 GiB VRAM for the avatar → **Phase 4 fallback** |
| faster-whisper (CTranslate2 4.8.2) | MIT | ~99 | — | GPU (CUDA 12 build) / CPU int8 | ~1 GiB int8 | 100–200 ms | CTranslate2 still ships CUDA 12 builds; cuDNN optional since 4.6.3 [V]. DLL names are versioned (`cublas64_12` vs torch's `…_13`), so coexistence in one process is **plausible but unverified** → isolated test |
| SenseVoiceSmall (FunASR) | FunASR Model License (attribution; permissive-ish) | zh, en, ja, ko, yue | 0.87 GiB [V] | GPU/CPU | ~1 GiB | ~70 ms per 10 s audio (published) | Very fast; OpenAvatarChat's default; limited languages |
| Moonshine | MIT | English | small | CPU | 0 | fast | English only |
| Kyutai STT 1B | CC-BY-4.0 | en, fr | 2.2 GiB | GPU | ~2.5 GiB | streaming | Linux-oriented server |

**Recommendation:** Whisper large-v3-turbo via `transformers` for Phases 2–3 (VAD-segmented, not token-streaming). If the avatar needs the VRAM, switch to Parakeet v3 on CPU or Whisper small. Evaluate faster-whisper only in an isolated test. **The final pick depends on which languages you need (see Appendix C).**

---

## 6. LLM recommendation

**Runtime:** llama.cpp `llama-server` (MIT), prebuilt Windows binaries. Latest checked: **b11242 (2026-09-28)** with `llama-b11242-bin-win-cuda-12.4-x64.zip` (252 MiB) + `cudart-…-cuda-12.4-x64.zip` (373 MiB), and `…-cuda-13.4-x64.zip` (146 MiB) + cudart (404 MiB) [V]. The driver exposes CUDA 13.3 [M] → the **CUDA 12.4 build is the conservative choice**; the 13.4 build relies on CUDA minor-version compatibility → test both. Separate process, OpenAI-compatible streaming HTTP on `127.0.0.1`, no Python dependency impact.

| Model | License | Q4_K_M file | Notes | Verdict |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | Apache-2.0 | **2.33 GiB** [V] | Non-thinking instruct variant; mature llama.cpp support; multilingual | **Phase 1 default** |
| Qwen3.5-4B (Feb 2026) | Apache-2.0 | 2.55 GiB (+0.63 GiB optional vision projector) [V] | Newer; registered as image-text-to-text architecture. Reasoning-mode default and llama.cpp speed [U] | Benchmark vs Qwen3-4B-2507 |
| Qwen3.5-2B | Apache-2.0 | ~1.3 GiB [E] | Fallback if VRAM is short | Benchmark |
| Phi-4-mini-instruct | MIT | ~2.4 GiB [E] | Alternative | Optional |
| SmolLM3-3B / Ministral-3-3B-2512 / Granite-4.0-micro | Apache-2.0 | ~2 GiB [E] | Alternatives | Optional |
| Llama-3.2-3B-Instruct | Llama 3.2 Community License (gated) | — | Extra license terms and attribution | Not default |
| Gemma 3 / 3n 4B | Gemma Terms of Use (gated) | — | Use restrictions | Not default |
| Qwen3-8B / Qwen3.5-9B | Apache-2.0 | ~5 GiB | Only if the avatar leaves ≥ 6 GiB free | Later |

**KV-cache math (Qwen3-4B):** 36 layers × 8 KV heads × 128 dims × 2 (K, V) × 2 bytes = **144 KiB/token** → 4k context ≈ 0.56 GiB (f16) or ≈ 0.3 GiB (q8_0).

**Initial server settings** (to verify against `llama-server --help` of the downloaded build): full GPU offload, context 4096, flash-attention on, q8_0 K/V cache, 1 slot, prompt caching on, bind 127.0.0.1.

**Prompt layout for cache reuse:** `[system][rolling summary][recent turns…][latest user message + retrieved context]` — putting RAG context in the *latest* message keeps the long prefix cacheable across turns.

**Behaviour metadata:** the model emits one metadata line before the speech text, e.g. `<meta>{"emotion":"empathetic","gesture":"small_nod","intensity":0.45}</meta>`, then plain speech. This keeps the speech streamable (unlike putting speech inside a JSON field). A llama.cpp GBNF grammar can enforce the header shape; the backend strips it before TTS. Thinking/reasoning output must be disabled for latency.

**Expected:** TTFT 150–400 ms with a prefix-cache hit; 60–100 tokens/s decode [E].

---

## 7. TTS recommendation

| Model | License / commercial | Languages | Cloning | Weights | VRAM (est.) | First audio | Streaming | Word timestamps | Windows | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | Apache-2.0 ✓ | en-US/en-GB best; es, fr, hi, it, ja, pt, zh via espeak-ng | ✗ (54 preset voices) | 0.33 GiB [V] | 0.4–0.6 GiB, or CPU | 50–150 ms GPU per clause [E] | Clause/sentence chunks | **✓** from the duration predictor (`KPipeline.join_timestamps`) [V] | ✓ (needs espeak-ng installer, GPL-3.0, separate program) | **Phase 2 default** |
| **Qwen3-TTS-12Hz-0.6B** (Base / CustomVoice) | Apache-2.0 ✓ | 10 (zh, en, ja, ko, de, fr, ru, pt, es, it) | ✓ 3-second | 2.3 GiB [V] | 2–3 GiB | "as low as 97 ms" (GPU unspecified) [V] | ✓ dual-track streaming | ✗ | Python 3.12 recommended, flash-attn optional [V]; Windows [U] | **Benchmark** (multilingual/cloning) |
| CosyVoice 3 (Fun-CosyVoice3-0.5B-2512) | Apache-2.0 ✓ | 9 + dialects | ✓ | 9.1 GiB repo [V] | 2.5–4 GiB | ~150 ms published | ✓ bi-streaming | ✗ | Pins torch 2.3.1; TensorRT/deepspeed Linux-only; pynini needs Conda on Windows [V] | Benchmark only in separate env |
| Chatterbox-Turbo | MIT ✓ | English | ✓ | 3.8 GiB repo [V] | 1.5–2.5 GiB | [U] | community forks [U] | ✗ | [U] | Benchmark (English quality) |
| Chatterbox (multilingual) | MIT ✓ | 23 | ✓ | 12.9 GiB repo [V] | 3–4 GiB | [U] | [U] | ✗ | [U] | Optional |
| VibeVoice-Realtime-0.5B | MIT ✓ | English | ✗ | 1.9 GiB [V] | ~2 GiB | [U] | ✓ | ✗ | [U] | Optional |
| Kyutai TTS 1.6B | CC-BY-4.0 ✓ (attribution) | en, fr | ✓ | 3.8 GiB [V] | ~4 GiB | low | ✓ | ✓ (word-level) [E] | Linux-oriented | Optional |
| NeuTTS Air | Apache-2.0 ✓ (gated) | English | ✓ | 2.8 GiB [V] | CPU/GGUF | [U] | [U] | ✗ | [U] | Optional |
| MeloTTS | MIT ✓ | several | ✗ | 0.19 GiB | CPU | fast | chunks | ✗ | ✓ | Lightweight fallback |
| Fish Speech / OpenAudio S1-mini | **CC-BY-NC-SA-4.0 / Fish Audio Research License** ✗ | many | ✓ | — | — | — | — | — | — | **Not recommended** (non-commercial) |
| F5-TTS | **CC-BY-NC-4.0** ✗ | — | ✓ | — | — | — | — | — | — | **Not recommended** |
| XTTS-v2 | **Coqui Public Model License** (non-commercial) ✗ | — | ✓ | — | — | — | — | — | — | **Not recommended** |
| IndexTTS-2 | bilibili Model Use License (royalty-free with conditions, e.g. large-MAU clause) ⚠ | en, zh | ✓ | 5.5 GiB | high | — | — | — | — | Legal review first |
| Orpheus 3B | Apache-2.0 | English | ✓ | 52.8 GiB repo | > 6 GiB | — | — | — | — | Too heavy |

**Recommendation:** Kokoro for Phase 2 (English, fastest, timestamps). Benchmark **Qwen3-TTS-0.6B** as the multilingual + voice-cloning option and **Chatterbox-Turbo** for English quality.

The neural avatars are **audio-driven**, so word timestamps are not needed for lip sync. They are needed for recording *what the user actually heard* when interrupted. Without them we fall back to sentence-level accounting (every TTS chunk is one clause, and the browser reports playback position).

---

## 8. RAG recommendation

| Stage | Choice | License | Notes |
|---|---|---|---|
| Parse PDF | `pypdf` (or `pdfplumber`) | BSD-3 / MIT | **Avoid PyMuPDF (AGPL-3.0)** unless AGPL is acceptable |
| Parse DOCX | `python-docx` | MIT | |
| Markdown / TXT | `markdown-it-py` / stdlib | MIT | |
| Web pages | `trafilatura` | Apache-2.0 [E — verify current version] | Later |
| Heavy layout (optional) | `docling` | MIT | Pulls torch models; later |
| Chunking | Heading-aware, 300–500 tokens, 10–15% overlap; metadata: source, page, section, chunk_id, content hash | — | |
| Embeddings (CPU) | `bge-small-en-v1.5` (33M, 384-d) for English · `multilingual-e5-small` (118M) for multilingual · `Qwen3-Embedding-0.6B` if quality needs it | MIT / MIT / Apache-2.0 | Avoid EmbeddingGemma (Gemma terms). Run via sentence-transformers (Apache-2.0) or ONNX |
| Vector store (dev) | **LanceDB** (embedded, built-in full-text search → hybrid) or Chroma | Apache-2.0 | Both have Windows wheels |
| Vector store (later) | PostgreSQL + pgvector | PostgreSQL License | |
| Rerank (optional) | `cross-encoder/ms-marco-MiniLM-L6-v2` (English) or `bge-reranker-v2-m3` (multilingual, slow on CPU) | Apache-2.0 | |
| Grounding | top-k (default 4), similarity threshold, "insufficient evidence" path: the avatar says it doesn't know / asks to clarify; never invents facts | — | |
| Citations | Source metadata sent to the UI text panel (not spoken, or spoken briefly) | — | |

**Latency targets [E]:** query embedding ≤ 20 ms, search ≤ 10 ms, rerank ≤ 80 ms (if enabled) — all on CPU.

---

## 9. Dependency compatibility matrix

| Component | Environment | Python | torch / CUDA | Compile on Windows? | Conflicts with main `.venv`? | Status |
|---|---|---|---|---|---|---|
| Backend core (FastAPI, asyncio, pydantic) | main `.venv` | 3.11.15 | — | No | No | ✅ now |
| Silero VAD | main (onnxruntime CPU or torch) | 3.11 | CPU | No | No | ✅ Phase 2 |
| smart-turn v3 | main (onnxruntime CPU) | 3.11 | CPU | No | needs `transformers>=4.48` | ⏳ benchmark |
| Whisper via `transformers` | main | 3.11 | 2.9.1+cu130 | No | No | ✅ Phase 2 |
| faster-whisper / CTranslate2 4.8.2 | main or separate | 3.11 | own CUDA 12 runtime | No | DLL coexistence unverified | ⏳ |
| Parakeet v3 (sherpa-onnx / onnx-asr) | main, CPU | 3.11 | CPU ONNX | No | No | ⏳ |
| llama-server | standalone binary | — | bundled CUDA 12.4 or 13.4 runtime | No | None (separate process) | ✅ Phase 1 |
| Kokoro (`kokoro`, `misaki[en]`, spaCy) + espeak-ng | main | 3.11 | any torch | No (espeak-ng installer) | Low risk; verify numpy/spaCy | ✅ Phase 2 |
| Qwen3-TTS | separate venv preferred | 3.12 recommended | torch 2.x; flash-attn optional | Only if flash-attn is used | Unknown pins | ⏳ |
| CosyVoice 3 | separate venv | 3.10 | torch 2.3.1 pinned | pynini (Conda) | **Yes** | ⏳ |
| Chatterbox | separate venv | 3.11 | pinned torch [U] | No | Likely | ⏳ |
| RAG stack (sentence-transformers, LanceDB, pypdf, python-docx) | main | 3.11 | CPU | No | No | ✅ Phase 5 |
| aiortc (WebRTC) | main | 3.11 | — (PyAV wheels) | No | No | ✅ Phase 4 |
| OpenAvatarChat | sandbox venv only | 3.11.7+ | **torch 2.8.0+cu128, numpy ≤ 1.26.4** | flash-attn (FlashHead), pynini (CosyVoice) | **Yes — would replace torch** | ⏳ sandbox only |
| MuseTalk 1.5 (OpenAvatarChat-style) | `avatar/musetalk/.venv` | 3.11 | torch 2.8.0+cu128 (or test 2.9.1+cu130) | No (ONNX DWPose) | Yes → separate | ⏳ **first avatar benchmark** |
| MuseTalk 1.5 (official) | — | 3.10 | torch 2.0.1+cu118, mmcv 2.0.1 | mmcv likely | Yes | ❌ |
| LAM audio2expression | `avatar/lam/.venv` or CPU | 3.10/3.11 | torch; transformers 4.44 | No | pins | ⏳ |
| LAM avatar generation | WSL2 or HF/ModelScope Space | 3.10 | CUDA 12.1 / 11.8 + CUDA extensions | Yes | n/a | ⏳ |
| Ditto | WSL2 | 3.10 | TensorRT 8.6.1 | TRT engines | n/a | ⏳ |
| AVTR-1 | WSL2 (Pixi) | [U] | CUDA 12 + TensorRT 10 | TRT | n/a | ⏳ license |
| SoulX-FlashHead | WSL2 | 3.10 | torch 2.7.1 cu128 + flash-attn 2.8 | **Yes on Windows** | n/a | ⏳ / ❌ on 3060 |
| LiteAvatar | separate venv | 3.10+ | CPU / onnxruntime | No | funasr/transformers pins | ⏳ optional test avatar |
| Playwright | Node dev-dependency | — | — | No | No | ✅ Phase 1 |

### 9.1 Isolation plan

```
C:\AI\-AI-Avtart-Voice-Agent\.venv        backend — EXISTING env, never recreated (torch 2.9.1+cu130)
avatar\musetalk\.venv                     MuseTalk worker (Python 3.11, torch 2.8.0+cu128)
avatar\lam\.venv                          LAM audio2expression (if server-side)
tools\llama.cpp\b11242-cuda12.4\          llama-server binaries (gitignored)
tools\oac\                                optional OpenAvatarChat sandbox (own venv)
WSL2 Ubuntu ~/avatar-bench/<engine>       Ditto / AVTR-1 / FlashHead benchmarks only
E:\AI-models\                             MODEL_HOME; HF_HOME = E:\AI-models\hf
```

### 9.2 Keeping torch safe

- `constraints/backend.txt` pins `torch==2.9.1+cu130`, `torchaudio==2.9.1+cu130`, `torchvision==0.24.1+cu130`. Every install uses `uv pip install -c constraints/backend.txt …` (or `PIP_CONSTRAINT`).
- `scripts/safe_install.ps1` runs `scripts/check_env.py` **before and after** each install (Python version, torch version, `torch.version.cuda`, `cuda.is_available()`, GPU name) and fails loudly if anything changed.
- After each approved install, commit `constraints/backend.lock.txt` (a `pip freeze` snapshot) so the env is reproducible.
- Recovery if torch is ever replaced: `uv pip install torch==2.9.1+cu130 torchaudio==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130`.

---

## 10. Windows vs WSL2 recommendation

**WSL2 facts on this machine [M]:** same RTX 3060 (no extra VRAM), RAM capped at **7.6 GiB**, ~0.7 GiB used idle, **NAT** networking (WebRTC UDP from the Windows browser needs mirrored networking or port forwarding), file access across `/mnt/c` is slow (models should live inside the Linux filesystem → duplicate downloads).

| Component | Recommendation | Reason |
|---|---|---|
| Backend (FastAPI, VAD, STT, TTS, RAG) | **Native Windows** | Existing working env; no compile needed |
| llama-server | **Native Windows** | Official prebuilt CUDA binaries |
| Frontend (Vite dev server, browser) | **Native Windows** | — |
| MuseTalk 1.5 | **Native Windows** (ONNX preprocessing), WSL2 fallback | mmcv avoided; pure torch/diffusers runtime |
| LAM runtime (browser + audio2expression) | **Native Windows** | Browser WebGL; small server model |
| LAM asset generation | WSL2 or their HF/ModelScope Space (offline, one-off) | CUDA extensions |
| Ditto | WSL2 | TensorRT 8.6.1 engines, Linux-first |
| AVTR-1 | WSL2 | Pixi + TensorRT 10, Linux-first |
| SoulX-FlashHead | WSL2 (probe only) | flash-attn build; Linux-first |
| LiteAvatar | Native Windows | `.bat` scripts, CPU mode |
| OpenAvatarChat sandbox (optional) | WSL2 or native in its own venv | Linux-first tooling |

**Rule:** WSL2 is for **benchmark spikes**. If a WSL2-only engine wins, we either port its runtime to Windows or run it as the avatar worker inside WSL2, after raising the WSL2 memory cap — which requires more system RAM.

---

## 11. VRAM / RAM budget

### 11.1 VRAM — Phase 2 (voice pipeline, no neural avatar)

| Consumer | GiB | Src |
|---|---|---|
| Windows desktop + open apps | 1.8–1.9 | M |
| Backend process CUDA context | 0.3–0.5 | E |
| Whisper large-v3-turbo fp16 | 1.8–2.2 | E |
| Kokoro-82M | 0.4–0.6 | E |
| llama-server: context + Qwen3-4B Q4_K_M (2.33 file) + 4k KV f16 (0.56) + compute buffers (0.3–0.5) | 3.4–3.9 | V file + calc + E |
| **Total** | **7.7–9.1** | |
| Free (of 12.0) | 2.9–4.3 | |

An 8B model (~5 GiB file + same KV) would push the total to ~10.5–11.6 GiB → no margin. **Stay at 4B.**

### 11.2 VRAM — Phase 4 with MuseTalk resident

| Consumer | GiB | Src |
|---|---|---|
| Phase 2 total | 7.7–9.1 | above |
| Avatar worker CUDA context | 0.3–0.5 | E |
| MuseTalk v1.5 fp16 + batch activations | 2.5–3.3 | E |
| **Total** | **10.5–12.9 — over budget at the upper end** | |

**Mitigations, in order:** STT → Parakeet on CPU or Whisper small (−1.2 to −2.2) · Kokoro on CPU (−0.5) · q8_0 KV cache (−0.25) · context 3k · reference video ≤ 512p. Realistic target after mitigation: **9.0–10.3 GiB** → must be measured.

### 11.3 VRAM — Phase 4 with LAM

Phase 2 total + audio2expression (wav2vec2-base, 0.3–0.6 GiB or CPU) + browser WebGL (0.2–0.5 GiB, if the browser runs on POOH-PC) ≈ **8.5–10 GiB** → fits.

### 11.4 System RAM (the hidden bottleneck)

| Consumer | GiB | Src |
|---|---|---|
| Windows + currently open apps | ~13.8 in use | M (15.7 − 1.9 free) |
| Backend (torch + Whisper + Kokoro + embeddings) | 2.5–4.0 | E |
| llama-server (mmap'd GGUF, fully offloaded) | 0.5–1.0 | E |
| MuseTalk worker (torch + cached frames/masks) | 2.5–4.0 | E |
| Vite dev server + browser tab with WebRTC | 0.6–1.0 | E |
| **Needed for our stack** | **6–10** | |

**We cannot run the full stack with today's free RAM without heavy paging.** Short term: close Chrome/Battle.net/Telegram/WPS during tests. Recommended: **upgrade to 32 GB RAM** (the single highest-value hardware change for this project).

### 11.5 Model scheduling — A (resident) vs B (controlled)

- **A — all resident:** lowest latency; feasible for Phase 2; tight for Phase 4.
- **B — controlled scheduling** (benchmark variants):
  - **B1 priority gating:** weights stay resident; the backend gates its own GPU work with an async priority semaphore (STT burst at END_OF_TURN > first TTS clause > avatar frames); the avatar renders at most N frames ahead. llama-server cannot be pre-empted, but its bursts are short (a sentence ≈ 0.2–0.3 s).
  - **B2 weight swapping:** move STT weights to pinned CPU RAM while the avatar renders. Costs ~0.1–0.2 s per swap at PCIe 4.0 [E] **and needs ~2 GiB of free RAM we don't reliably have**.
- **Initial position:** A with smaller models, plus B1 gating. B2 only after the RAM upgrade and only if measurements justify it.
- **Guards:** set NVIDIA *CUDA Sysmem Fallback Policy → Prefer No Sysmem Fallback* for `python.exe` and `llama-server.exe` (clean OOM instead of a silent 10× slowdown); `torch.cuda.set_per_process_memory_fraction` per process; startup VRAM guard refuses to start if projected free VRAM < 1.0 GiB; `CUDA_MODULE_LOADING=LAZY`.

---

## 12. Process architecture

```
                         ┌──────────────────────── POOH-PC (Windows) ─────────────────────────┐
 Browser (React+Vite)    │  P1 backend  (.venv, torch 2.9.1+cu130)                             │
 ┌───────────────────┐   │  ┌────────────────────────────────────────────────────────────┐    │
 │ mic AudioWorklet  │──WS/WebRTC──► Session manager + explicit state machine (turn_id)   │    │
 │ player / <video>  │◄─────────┤  VAD (CPU) → STT (GPU) → RAG (CPU) → LLM client ───────┼──┐ │
 │ state + metrics   │          │  BehaviorManager ← metadata parser                     │  │ │
 └───────────────────┘          │  TTS (GPU/CPU) → audio chunks(turn_id, seq, pts) ───┐  │  │ │
                                │  Metrics / health / VRAM guard / supervisor         │  │  │ │
                                └─────────────────────────────────────────────────────┼──┘  │ │
                                  HTTP SSE (OpenAI-compatible) 127.0.0.1:8081         │     │ │
                                ┌───────────────────────────┐                         │     │ │
                                │ P2 llama-server.exe       │◄────────────────────────┼─────┘ │
                                │ Qwen3-4B Q4_K_M, 4k ctx   │                         │       │
                                └───────────────────────────┘                         │       │
                                  local WS/TCP, binary msgpack                        ▼       │
                                ┌───────────────────────────────────────────────────────┐    │
                                │ P3 avatar worker (avatar/<engine>/.venv, own torch)    │    │
                                │ audio chunks + behavior + cancel(turn_id) → frames+pts │    │
                                └───────────────────────────────────────────────────────┘    │
                         └──────────────────────────────────────────────────────────────────┘
```

- **P1 backend** supervises P2 and P3 (spawn, health check, restart), owns all state and the only client-facing connection.
- **P2** is isolated because it has its own CUDA runtime and is the largest VRAM consumer; restartable independently.
- **P3** exists because avatar engines need conflicting dependencies; it also gives crash isolation and clean per-process VRAM measurement. Cost: one extra CUDA context (~0.3–0.5 GiB [E]) and ~1–3 ms per frame of localhost IPC [E] (512×512 RGB ≈ 0.75 MB/frame → ~19 MB/s at 25 fps, trivial locally).
- No other processes in production. The Vite dev server exists only during development.

---

## 13. Streaming architecture

| Phase | Uplink (mic) | Downlink | Control/events |
|---|---|---|---|
| 1 (text) | — | LLM token deltas (JSON over WebSocket) | WebSocket JSON |
| 2–3 (voice) | AudioWorklet → PCM16 16 kHz, 20 ms binary frames over WebSocket | PCM16 24 kHz TTS chunks (binary header: `turn_id`, `seq`, `sample_offset`) → AudioWorklet ring buffer with instant flush | WebSocket JSON: `state`, `transcript`, `llm_delta`, `behavior`, `interrupt`, `metrics`, `playback_progress` |
| 4 (neural avatar) | WebRTC audio track (browser AEC/NS/AGC) | **WebRTC audio + video tracks via aiortc** (VP8/H.264 encoded on CPU; 512×512 @ 20–25 fps) | WebRTC data channel (+ WebSocket for non-realtime) |

**Why WebRTC for Phase 4:** RTP timestamps give real A/V sync; a jitter buffer; congestion control; the remote audio track gives the browser's echo canceller a correct reference; far less bandwidth than raw frames. The WebSocket-JPEG alternative is simpler but has weaker sync and echo handling → keep it as a fallback in the benchmark. (OpenAvatarChat also uses aiortc.)

**A/V sync rule:** the avatar worker stamps each frame with the audio sample PTS it corresponds to; the backend releases audio only when the matching frames exist (hold ≤ one render batch). Consequence: avatar first-frame latency adds directly to "first avatar audio".

**Perceived-latency tricks:** the avatar switches to a *listening → thinking* visual state within ~100 ms of end-of-turn; the first spoken clause is short.

**Network:** localhost needs nothing extra. LAN access needs HTTPS (browser microphone permission). Internet access needs HTTPS + TURN.

---

## 14. Interruption architecture

### 14.1 State machine (single owner: `SessionStateMachine`, explicit transition table)

| From | Event | To | Actions |
|---|---|---|---|
| IDLE | session start / mic open | LISTENING | start VAD |
| LISTENING | end-of-speech (silence ≥ endpoint, or smart-turn "complete") | END_OF_TURN | freeze utterance, allocate `turn_id` |
| END_OF_TURN | transcript non-empty | THINKING | RAG + LLM request |
| END_OF_TURN | empty/noise, or user resumes speaking | LISTENING | discard turn |
| THINKING | first TTS audio (Phase 4: first synced frame) sent | SPEAKING | — |
| THINKING / SPEAKING | confirmed barge-in, or manual stop | INTERRUPTED | cancel fan-out (below) |
| INTERRUPTED | all cancellations acknowledged | LISTENING | keep the barge-in audio pre-roll as the start of the next utterance |
| SPEAKING | client reports playback finished | LISTENING | commit assistant message |
| any | component failure | ERROR | degrade (e.g. TTS failure → text only), then recover to LISTENING / IDLE |

### 14.2 Turn IDs and stale-packet rejection

- `turn_id` = monotonically increasing integer per session. Every LLM delta, TTS chunk, behaviour event, avatar frame and client ack carries it.
- Each producer is an asyncio task bound to a `TurnContext(turn_id, cancel_event, timings)`.
- Every sink (TTS queue, avatar queue, transport sender, browser player) **drops any packet whose `turn_id` ≠ current**.

### 14.3 Cancellation fan-out (target ≤ 200–300 ms after confident detection)

1. Detection: Silero speech probability above a **playback-aware threshold** (higher while SPEAKING) for ≥ 200–300 ms, plus an energy gate. Optional confirmation: fast local transcription of the first ~0.5 s must contain ≥ 1 word. **No cloud interrupt judge.**
2. Server → browser `interrupt(turn_id)`: the browser flushes the AudioWorklet buffer / mutes the remote audio track (< 20 ms) and shows idle frames.
3. Close the LLM HTTP stream (llama-server should stop generating when the client disconnects — **verify in Phase 1**).
4. TTS worker checks the cancel event between clauses (≤ 1 clause ≈ 50–150 ms of work wasted).
5. Avatar worker `cancel(turn_id)` → drop queued audio and frames, switch to idle/listening loop.
6. Clear all queues; state → LISTENING.

### 14.4 What the user actually heard

The browser sends `playback_progress(turn_id, samples_played)` every 100–200 ms and on interrupt. The server maps it to clauses (and word timestamps when Kokoro is used) and stores only the heard text plus an `[interrupted]` marker in history.

### 14.5 Echo handling

- `getUserMedia({echoCancellation: true, noiseSuppression: true, autoGainControl: true})`.
- Headphones for development.
- Playback-aware VAD threshold; ignore the first ~150 ms after each playback start.
- Test matrix: headset / desktop speakers + USB mic / speakers + webcam mic.
- Metric: self-interruptions per 10 minutes — target 0 with headset, ≤ 1 on speakers.
- Whether Chrome's echo canceller treats WebAudio playback (Phases 2–3) as well as a WebRTC track is **[U] — test in Phase 3**. This is one reason WebRTC is planned for Phase 4.

### 14.6 Conversation memory

Last 6–8 turns verbatim + a rolling summary (updated by the same local LLM *after* the turn, never in the latency path) + retrieved knowledge + optional session facts.

---

## 15. Repository structure (proposed)

```
frontend/                  React + Vite + JavaScript
  src/audio/               capture + playback AudioWorklets
  src/net/                 WebSocket / WebRTC clients, protocol
  src/ui/                  conversation, state, metrics panels
backend/
  app/
    api/                   ws, http, health
    core/                  config (profiles), logging, metrics, gpu guard
    session/               state_machine.py, turn.py, conversation.py, memory.py
    audio/                 vad.py, turn_detector.py, codecs.py
    stt/  llm/  tts/       engine interfaces + implementations
    rag/                   ingest, chunk, embed, store, retrieve
    behavior/              behavior_manager.py, metadata parser
    avatar/                avatar-worker client + IPC protocol
    transport/             websocket.py, webrtc.py
  tests/
avatar/
  common/                  IPC protocol shared by all workers
  musetalk/                worker + requirements + constraints (own .venv)
  lam/                     (if benchmarked)
shared/                    JSON Schemas for events/packets (backend, workers, frontend)
scripts/                   check_env.py, bench_gpu.py, bench_latency.py, safe_install.ps1,
                           download_models.py, ingest.py
constraints/               backend.txt, backend.lock.txt, avatar-musetalk.txt
tools/                     llama.cpp binaries (gitignored)
docs/                      ARCHITECTURE.md, this report, decisions/ (ADRs), benchmarks/, THIRD_PARTY_NOTICES.md
tests/e2e/                 Playwright tests (fake-microphone scenarios)
data/                      knowledge/, index/, avatars/ (all gitignored — may contain personal likeness)
```

- The existing root `.venv` **stays where it is** as the backend env (moving a venv breaks it).
- No Python package is named `models` (use `schemas/`, `engines/`), so `.gitignore` can ignore `models/` everywhere, including inside vendored avatar code.
- Reference photos/videos of real people live in `data/avatars/` (gitignored) — privacy and likeness rights.

---

## 16. Licensing table

| Component | Role | Code license | Weights / data license | Commercial | Obligations / notes |
|---|---|---|---|---|---|
| OpenAvatarChat | reference / reused code | Apache-2.0 | — | ✓ | Keep NOTICE + attribution for reused files |
| MuseTalk 1.5 | avatar (lip) | MIT | MIT | ✓ | Dependency weights: sd-vae-ft-mse MIT, whisper-tiny Apache-2.0, DWPose Apache-2.0, face-parse-bisent WTFPL, S3FD BSD-3 [verify] |
| LAM / LAM_Audio2Expression | avatar (3D GS) | Apache-2.0 | audio2exp Apache-2.0; LAM-20K [U] | likely ✓ | Confirm LAM-20K terms |
| Ditto | avatar | Apache-2.0 | [U] | confirm | Check whether InsightFace is used |
| AVTR-1 | avatar | AVTR-1 Community | same | ⚠ < USD 10M revenue; InsightFace non-commercial | Legal review required |
| SoulX-FlashHead | avatar | Apache-2.0 | Apache-2.0 | ✓ | — |
| LiteAvatar | test avatar | MIT | gallery [U] | likely ✓ | — |
| LivePortrait (+InsightFace) | offline motion | MIT | InsightFace models non-commercial | ⚠ | Replace the detector or license it |
| Wav2Lip | — | custom | non-commercial | ✗ | Not used |
| Silero VAD | VAD | MIT | MIT | ✓ | — |
| smart-turn v3 | end-of-turn | BSD-2-Clause | BSD-2-Clause | ✓ | Attribution |
| Whisper large-v3-turbo / small | STT | MIT | MIT / Apache-2.0 | ✓ | — |
| Parakeet-TDT-0.6B-v3 | STT | Apache-2.0 (sherpa-onnx) | CC-BY-4.0 | ✓ | Attribution |
| SenseVoiceSmall | STT | MIT (FunASR) | FunASR Model License | ✓ with attribution | Keep model names/attribution; termination clause |
| faster-whisper / CTranslate2 | STT runtime | MIT | — | ✓ | — |
| llama.cpp | LLM runtime | MIT | — | ✓ | — |
| Qwen3-4B-Instruct-2507 / Qwen3.5 | LLM | — | Apache-2.0 | ✓ | — |
| Llama 3.2 / Gemma | LLM | — | Llama 3.2 Community / Gemma Terms | ⚠ | Extra terms; not default |
| Kokoro-82M | TTS | Apache-2.0 | Apache-2.0 | ✓ | — |
| espeak-ng | G2P fallback for Kokoro | GPL-3.0 | — | ✓ as a separate program | Don't statically bundle into closed binaries without GPL review |
| Qwen3-TTS | TTS | Apache-2.0 | Apache-2.0 | ✓ | Voice cloning of real people needs consent |
| CosyVoice 3 | TTS | Apache-2.0 | Apache-2.0 | ✓ | — |
| Chatterbox / Turbo | TTS | MIT | MIT | ✓ | Built-in watermark |
| Fish Speech / OpenAudio | TTS | Fish Audio Research License | CC-BY-NC-SA-4.0 | ✗ | Not used |
| F5-TTS | TTS | MIT | CC-BY-NC-4.0 | ✗ | Not used |
| XTTS-v2 | TTS | MPL (code) | Coqui Public Model License | ✗ | Not used |
| IndexTTS-2 | TTS | bilibili Model Use License | same | ⚠ | Legal review |
| sentence-transformers | embeddings runtime | Apache-2.0 | — | ✓ | — |
| bge-small-en-v1.5 / bge-m3 / multilingual-e5-small | embeddings | — | MIT | ✓ | — |
| Qwen3-Embedding-0.6B | embeddings | — | Apache-2.0 | ✓ | — |
| bge-reranker-v2-m3 / ms-marco-MiniLM-L6-v2 | rerank | — | Apache-2.0 | ✓ | — |
| LanceDB / Chroma / FAISS / pgvector | vector store | Apache-2.0 / Apache-2.0 / MIT / PostgreSQL | — | ✓ | — |
| pypdf / pdfplumber / python-docx | parsing | BSD-3 / MIT / MIT | — | ✓ | — |
| PyMuPDF | parsing | **AGPL-3.0** | — | ⚠ | Avoid unless AGPL is acceptable |
| FastAPI / uvicorn / aiortc | backend | MIT / BSD-3 / BSD-3 | — | ✓ | — |
| React / Vite | frontend | MIT | — | ✓ | — |
| Playwright | testing | Apache-2.0 | — | ✓ | — |

Separately from software licenses: **the likeness and voice of a real person require that person's consent** (images, reference videos, voice clones).

---

## 17. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | System RAM exhausted (only ~1.9 GiB free) → paging, stalls | **High** | High | Close apps for tests; upgrade to 32 GB; cap frame caches; lazy loading |
| 2 | VRAM oversubscription silently spills to system RAM (WDDM sysmem fallback) | Medium | High | NVIDIA "Prefer No Sysmem Fallback"; VRAM guard; per-process memory fraction |
| 3 | MuseTalk < 20 fps on a 3060 alongside the LLM/TTS | **High** | High | Benchmark early; fps 20; batch tuning; CPU STT; LAM as alternative |
| 4 | MuseTalk can't express emotion / gaze on demand | Certain | Medium | Recorded state loops + cross-fade (H1); or LAM (H3) |
| 5 | Single-image requirement conflicts with MuseTalk | Certain | Medium | Reference video now; offline motion generator (H2) later |
| 6 | Licenses: InsightFace, AVTR-1, non-commercial TTS | Medium | High (commercial) | License table; exclude NC components; legal review before launch |
| 7 | Self-echo causes self-interruption | Medium | High | Headset dev; AEC; playback-aware thresholds; WebRTC in Phase 4 |
| 8 | pip replaces CUDA torch | Medium | High | Constraints + `safe_install.ps1` + lock snapshots |
| 9 | Windows-only build failures (flash-attn, mmcv, pynini, TensorRT) | High | Medium | No compilation on Windows; WSL2 for spikes only |
| 10 | GPU compute contention (display + browser + 3 GPU processes) raises first-audio latency | High | Medium | Priority gating (B1), short first clause, render-ahead limit |
| 11 | Windows TDR (2 s GPU timeout on the display GPU) during long offline kernels | Low | Medium | Real-time kernels are short; do offline generation in WSL2/Space; do not edit TDR registry without approval |
| 12 | Remote testing via AnyDesk doesn't carry the tester's mic | High | Low | Test at the PC, or serve over HTTPS on the LAN |
| 13 | MuseTalk code unmaintained since 2025-09 | Medium | Low | We own the forked worker code (from OAC) |
| 14 | 1.2 s target missed with a neural avatar | Medium | Medium | Tuning list (§18 / Appendix B); report measured numbers honestly |
| 15 | Antivirus scanning slows model loading | Medium | Low | Optional exclusion for `E:\AI-models` (your decision) |

---

## 18. Phase 0 implementation plan (after approval — no large downloads)

| Step | Work | Output |
|---|---|---|
| 0.1 | Update `.gitignore` to your minimum list: unanchored `models/`, `weights/`, `checkpoints/`, `cache/`, `dist/`, `logs/`, `*.log`, media (`*.mp4 *.mov *.webm *.wav *.flac *.mp3` with `!tests/fixtures/**` exceptions), `data/avatars/` | `.gitignore` |
| 0.2 | Convert `README.md` UTF-16 → UTF-8; add `.gitattributes` (LF for code, binary for media) | README readable in git |
| 0.3 | `docs/ARCHITECTURE.md` (condensed decisions from this report), `docs/decisions/ADR-0001…`, `docs/THIRD_PARTY_NOTICES.md` | Documented decisions |
| 0.4 | `constraints/backend.txt` + `scripts/safe_install.ps1` + `constraints/backend.lock.txt` | Torch protected |
| 0.5 | `scripts/check_env.py` — Python/torch/CUDA/GPU/driver/VRAM/RAM/disk/ffmpeg/node → JSON in `logs/` | Environment report |
| 0.6 | `scripts/bench_gpu.py` — NVML per-process VRAM, CUDA context cost, matmul TFLOPS, pinned vs pageable H2D/D2H bandwidth (decides B2 feasibility), safe allocation ceiling | Measured baseline |
| 0.7 | Latency-event schema (`shared/schemas/metrics.json`) + `scripts/bench_latency.py` skeleton (used from Phase 1) | Measurement tooling |
| 0.8 | Backend scaffold: FastAPI `/health`, `/ws` echo, `SessionStateMachine` + transition-table unit tests (no models) | Runs + tests pass |
| 0.9 | Frontend scaffold: Vite + React (JavaScript), WS connect, state panel | Runs in browser |
| 0.10 | `.env.example` with `MODEL_HOME=E:\AI-models`, `HF_HOME=E:\AI-models\hf`; create the empty folder | Models outside git |
| 0.11 | **Manual (you):** NVIDIA Control Panel → Manage 3D settings → Program settings → `python.exe` (venv) and later `llama-server.exe` → *CUDA - Sysmem Fallback Policy* = *Prefer No Sysmem Fallback* | Clean OOM behaviour |
| 0.12 | Commit (only when you ask) | Clean git |

**Phase 0 PASS:** `check_env.py` shows torch 2.9.1+cu130 with CUDA available and the RTX 3060 detected; the scaffolds run; state-machine tests pass; this report and the ADRs are committed; `git status` is clean.

**Proposed Phase 0.5 (needs download approval):** avatar feasibility spikes in parallel with Phase 1 — MuseTalk (~4 GB) and LAM (~3 GB) — because the avatar choice determines how much VRAM is left for STT/LLM/TTS.

---

## Appendix A — Playwright and MCP

- **Playwright (Apache-2.0)** from Phase 1: `@playwright/test` + Chromium (~150–200 MB). Chromium's fake-media flags (`--use-fake-ui-for-media-stream`, `--use-fake-device-for-media-stream`, `--use-file-for-fake-audio-capture=<wav>`) let us run **automated spoken conversations**: end-to-end latency (browser-side timestamps), barge-in tests (inject speech mid-answer), stale-packet checks, reconnection, and the 30-minute soak test.
- **Playwright MCP** (`@playwright/mcp`, Apache-2.0) would let Claude Code drive and inspect the running UI while debugging. It requires adding an MCP server to Claude Code's config — a configuration change I will only make with your approval, at the start of Phase 1. Not needed for Phase 0.
- No other MCP servers are required. GitHub operations use `git`/`gh`.

## Appendix B — Benchmark protocol

**Pipeline latency (every phase):** 20 scripted utterances played through the fake microphone; timestamps for end-of-speech, VAD end, STT done, RAG done, LLM first token, first clause, TTS first audio, avatar first frame, browser first audio. Report p50/p90.

| Stage | Target | Estimate on this PC |
|---|---|---|
| VAD end-of-speech window | 300–500 ms | 300–600 ms |
| STT (≤ 5 s utterance) | ≤ 250 ms | 150–300 ms |
| RAG | ≤ 30 ms | 15–40 ms |
| LLM TTFT (prefix-cache hit) | ≤ 300 ms | 150–400 ms |
| First clause complete | ≤ 200 ms | 120–200 ms |
| TTS first audio | ≤ 150 ms | 60–150 ms |
| Avatar first frame (Phase 4) | ≤ 200 ms | 100–250 ms |
| Network + jitter buffer | ≤ 100 ms | 20–60 ms WS · 60–150 ms WebRTC |
| **End-of-speech → first audio** | **≤ 1.2 s** | **~0.8–1.8 s audio-only; ~1.0–2.0 s with MuseTalk** |

Tuning levers to reach ≤ 1.2 s: smart-turn (shorter silence window when the sentence is complete), prompt prefix caching, a short first clause, TTS starting at the first comma, a minimal WebRTC jitter buffer, render batch 2.

**Avatar benchmark (per engine, same inputs):** one reference image or 20 s reference video; one 10 s TTS clip; measure preparation time, cold start, warm first-frame latency, sustained fps over 5 minutes (p50/p95 frame time), VRAM idle/peak (NVML), system RAM, CPU %. Run **alone** and **with llama-server generating concurrently**. Also: lip-sync quality (SyncNet confidence + blind side-by-side), identity stability, and a 30-minute soak (memory growth, drift).

## Appendix C — Decisions needed from you

> **Answered 2026-09-29:** personal / non-commercial use · English only · start from **one photo** ·
> stock **and** cloned voice · browser on the owner's own PC (remote → HTTPS needed) · no RAM upgrade ·
> everything approved. See [ARCHITECTURE.md](ARCHITECTURE.md) for the resulting design changes.

1. **Commercial use?** If the product will be sold or used commercially, AVTR-1, InsightFace-based tools, IndexTTS-2 and all non-commercial TTS models are excluded.
2. **Languages** the avatar must understand and speak (decides STT, TTS, embeddings).
3. **Reference input:** can you record a 20–30 s video of the person (recommended for MuseTalk), or must it start from a single photo?
4. **Voice:** stock voice acceptable, or a cloned voice (with consent)?
5. **Where the browser runs:** on POOH-PC (localhost) or another device (HTTPS needed)?
6. **Hardware:** is a RAM upgrade to 32 GB possible?
7. **Approval** for Phase 0 steps and, separately, for the Phase 0.5 / Phase 1 downloads.

## Appendix D — Primary sources

- OpenAvatarChat — https://github.com/HumanAIGC-Engineering/OpenAvatarChat (docs: https://humanaigc-engineering.github.io/OpenAvatarChat/en/)
- OpenAvatarChat-WebUI — https://github.com/HumanAIGC-Engineering/OpenAvatarChat-WebUI
- MuseTalk — https://github.com/TMElyralab/MuseTalk · weights https://huggingface.co/TMElyralab/MuseTalk
- LiveTalking (MuseTalk fps table) — https://github.com/lipku/LiveTalking
- SoulX-FlashHead — https://github.com/Soul-AILab/SoulX-FlashHead · https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B
- LAM — https://github.com/aigc3d/LAM · https://github.com/aigc3d/LAM_Audio2Expression
- LiteAvatar — https://github.com/HumanAIGC/lite-avatar
- Ditto — https://github.com/antgroup/ditto-talkinghead
- AVTR-1 — https://huggingface.co/avaturn-live/avtr-1 · https://github.com/avaturn-live/avtr-1
- LivePortrait — https://github.com/KlingAIResearch/LivePortrait · FasterLivePortrait https://github.com/warmshao/FasterLivePortrait · JoyVASA https://github.com/jdh-algo/JoyVASA
- EchoMimic V3 — https://github.com/antgroup/echomimic_v3 · LiveAvatar — https://github.com/Alibaba-Quark/LiveAvatar
- Kokoro — https://github.com/hexgrad/kokoro · https://huggingface.co/hexgrad/Kokoro-82M
- Qwen3-TTS — https://github.com/QwenLM/Qwen3-TTS · CosyVoice — https://github.com/QwenAudio/CosyVoice
- Chatterbox — https://github.com/resemble-ai/chatterbox
- Fish Speech license — https://github.com/fishaudio/fish-speech/blob/main/LICENSE
- IndexTTS license — https://github.com/index-tts/index-tts/blob/main/LICENSE
- FunASR model license — https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- faster-whisper CUDA 13 issue — https://github.com/SYSTRAN/faster-whisper/issues/1431 · CTranslate2 releases — https://github.com/OpenNMT/CTranslate2/releases
- llama.cpp releases — https://github.com/ggml-org/llama.cpp/releases
- Silero VAD — https://github.com/snakers4/silero-vad · smart-turn — https://github.com/pipecat-ai/smart-turn
- aiortc — https://github.com/aiortc/aiortc
- License/size metadata for all listed models: Hugging Face model API (`/api/models/<id>`), queried 2026-09-29.

---

## Installation classification

### SAFE TO INSTALL NOW (Phase 0; small; main venv under the torch constraint, or tooling)
- Python (main `.venv`, with `constraints/backend.txt`): `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `httpx`, `websockets`, `orjson`, `msgpack`, `loguru`, `psutil`, `nvidia-ml-py`, `python-dotenv`, `pytest`, `pytest-asyncio`, `ruff`
- Frontend: Vite + React (JavaScript) scaffold via npm
- Configuration only (no install): NVIDIA *Sysmem Fallback Policy* setting; `MODEL_HOME` / `HF_HOME` on `E:\AI-models`

**Phase 1–2 items, known-good, pending your go-ahead for downloads:** llama.cpp b11242 Windows CUDA 12.4 binaries (~0.6 GB) · Qwen3-4B-Instruct-2507 Q4_K_M (2.33 GiB) · `silero-vad` + `onnxruntime` (CPU) · `transformers` + Whisper large-v3-turbo (1.5 GiB) · `kokoro` + `misaki[en]` + espeak-ng installer + Kokoro-82M (0.33 GiB) · Playwright + Chromium (~0.2 GB) · later: `sentence-transformers`, `lancedb`, `pypdf`, `python-docx`, `aiortc`

### WAIT UNTIL BENCHMARK
- **MuseTalk 1.5** (own venv, ~4 GB weights) — first avatar benchmark
- **LAM** + LAM_Audio2Expression + browser renderer (~3 GB)
- **Ditto** (WSL2 + TensorRT), **AVTR-1** (WSL2; license review first)
- SoulX-FlashHead Lite (WSL2 probe; expected below real time on a 3060)
- LiteAvatar (optional interim test avatar, CPU)
- OpenAvatarChat full install (sandbox venv only, as a reference)
- Qwen3.5-4B / Qwen3.5-2B vs Qwen3-4B-2507; any 7–9B LLM
- Qwen3-TTS-0.6B, CosyVoice 3, Chatterbox-Turbo, VibeVoice-Realtime
- Parakeet v3 (sherpa-onnx), faster-whisper (CUDA 12 DLL coexistence test), SenseVoiceSmall, smart-turn v3
- Rerankers; Qwen3-Embedding-0.6B

### NOT RECOMMENDED
- Installing OpenAvatarChat (or any avatar engine) **into the main `.venv`** — would replace torch 2.9.1+cu130 / numpy 2.x
- MuseTalk's official `mim` / mmcv / mmpose install path on Windows (use the ONNX DWPose path)
- Building flash-attn, mmcv or other CUDA extensions on Windows; installing MSVC / CUDA toolkit for that purpose
- Fish Speech / OpenAudio, F5-TTS, XTTS-v2, Wav2Lip (non-commercial licenses)
- InsightFace-dependent pipelines (LivePortrait, AVTR-1, possibly Ditto) for **commercial** use without a license or detector replacement
- EchoMimic V3, Hallo 2/3, LiveAvatar 14B, SoulX-FlashTalk 14B (offline or far beyond 12 GB)
- Paid/cloud services in the main path (DashScope/Bailian presets, Edge TTS); cloud LLMs only as an optional, off-by-default fallback
- Llama 3.2 / Gemma as defaults (extra license terms); PyMuPDF (AGPL) unless accepted
- An 8B+ LLM together with a neural avatar on 12 GB
- Running the main backend inside WSL2 (7.6 GiB RAM cap, NAT/UDP issues, no VRAM gain)
