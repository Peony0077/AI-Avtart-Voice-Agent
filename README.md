# AI Avatar Voice Agent

A fully self-hosted, real-time conversational avatar: talk to a realistic face in your
browser, interrupt it at any time, and get answers grounded in your own documents.
Everything runs locally on one PC with an NVIDIA GPU — built on Windows with an RTX 3060 (12 GB), also
running on Ubuntu 24.04 ([Linux](#linux)) — no paid APIs.

Pipeline: browser mic → Silero VAD → Whisper STT → RAG (LanceDB) → Qwen LLM (llama.cpp)
→ behaviour metadata → Kokoro / cloned-voice TTS → MuseTalk lip-sync on a LivePortrait-animated
photo → audio + video back to the browser.

> Status, measurements and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
> research: [docs/PHASE0_TECHNICAL_REPORT.md](docs/PHASE0_TECHNICAL_REPORT.md) ·
> licenses: [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)

> **Not cleared for commercial use as set up here.** Lead generation and the customer link are business
> features, but two components of the standard setup are licensed for **non-commercial use only**:
> the InsightFace face detector used in every avatar build, and XTTS-v2. See
> [Licences and commercial use](#licences-and-commercial-use) before using this for a business.

| Measured on the RTX 3060 | |
|---|---|
| End of speech → first audio (median) | 0.92 s voice only · 1.16 s with the avatar |
| Barge-in detected | ~350 ms after you start talking; audio flushed in 8 ms |
| Lip sync in the browser | median 27 ms between sound and frame |
| Document answers | 8/8 correct with citations, retrieval ~26 ms |

## Layout

| Path | What |
|---|---|
| `backend/` | FastAPI app (state machine, VAD, STT, LLM client, RAG, TTS, avatar client) |
| `frontend/` | React + Vite (JavaScript) UI |
| `avatar/musetalk/` | Avatar worker process (own venv): LivePortrait loops + MuseTalk lip sync |
| `voice/xtts/` | XTTS-v2 cloned-voice worker process (own venv) |
| `scripts/` | run helpers, setup, benchmarks, model downloads, ingestion |
| `config/profiles/` | configuration profiles (`default`, `avatar`, `avatar_clone`, `clone`, `lite`, `hpn`) |
| `knowledge/hpn/` | Houston Party Network's Event-helper knowledge, prices and schedule (the `hpn` profile) |
| `constraints/` | pins that protect the CUDA PyTorch build |
| `tests/e2e/` | Playwright end-to-end tests (fake microphone) |

## Environment rules

- The backend venv (`.venv`, Python 3.11, **torch 2.9.1+cu130**) must never be replaced.
  Install packages only with `scripts\safe_install.ps1` (checks CUDA before/after).
- Model weights live outside git in `MODEL_HOME` = `AVATAR_MODEL_HOME` in `.env` (default `E:\AI-models`
  on Windows, `~/AI-models` on Linux); the Hugging Face cache is `HF_HOME` (default `MODEL_HOME\hf`).
- The avatar and XTTS workers use their own venvs, so their dependencies never touch the backend.
- Every download is pinned: Hugging Face repos to a commit, release files to a SHA-256.

## First-time setup (Windows)

Needs Python 3.11, [uv](https://docs.astral.sh/uv/), git, Node.js 20+, an NVIDIA driver for CUDA 13, and
`ffmpeg` on the PATH for the avatar's H.264 video.

```powershell
copy .env.example .env                                         # then set AVATAR_MODEL_HOME / HF_HOME in it
py -3.11 -m venv .venv                                         # the backend venv (or: uv venv --python 3.11 .venv)
uv pip install --python .venv\Scripts\python.exe -c constraints\backend.txt `
  --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1+cu130 torchaudio==2.9.1+cu130 torchvision==0.24.1+cu130
scripts\safe_install.ps1 -r backend\requirements.txt           # backend packages (checks that CUDA torch stays)
scripts\get_llama_cpp.ps1                                      # llama-server (CUDA build) -> tools\llama.cpp
.venv\Scripts\python scripts\download_models.py llm voice rag voice_convert   # + stt_small for the lite profile
cd frontend; npm ci; cd ..
powershell -File avatar\musetalk\setup.ps1                     # avatar venv + MuseTalk/LivePortrait (optional)
avatar\musetalk\.venv\Scripts\python avatar\musetalk\download_weights.py   # avatar weights (with the line above)
powershell -File voice\xtts\setup.ps1                          # XTTS-v2 venv + weights (optional, non-commercial)
```

`safe_install.ps1` refuses to run until `.venv` has a working CUDA PyTorch: that is what the two lines
before it set up.

## Run

```powershell
scripts\run.ps1 -Build                     # voice + text, stock voice: http://127.0.0.1:8000
scripts\run.ps1 -Profile avatar            # + realistic avatar
scripts\run.ps1 -Profile avatar_clone      # avatar with a cloned voice
scripts\run.ps1 -Profile clone             # voice only, XTTS-v2 cloned voice (closest to the reference)
scripts\run.ps1 -Profile lite              # smallest: whisper-small.en (download_models.py stt_small), no avatar
scripts\run.ps1 -Stop
```

Settings come from `.env` (`AVATAR_PROFILE`, `AVATAR_MODEL_HOME`, `HF_HOME`); `-Profile` overrides the
profile for this start only.

**From another PC** (the microphone needs HTTPS):

```powershell
scripts\tunnel.ps1 -Profile avatar         # internet: prints https://<random>.trycloudflare.com/?token=...
scripts\tunnel.ps1 -Profile avatar -Detach # same, keeps running in the background (link in logs\tunnel_url.txt)
scripts\run.ps1 -Profile avatar -Https     # same network: prints https://<LAN IP>:8000/?token=...
```

Whenever the backend listens on more than this PC (`-Https`, or `-BindHost` other than 127.0.0.1) it
gets random owner and customer tokens: without a token everyone who reaches the page would be the owner.

Two links are printed (and saved in `logs\tunnel_url.txt` and `logs\customer_url.txt`):

- **Owner link**: everything, including **Settings** (persona, voice with ▶ Listen, business and lead
  generation, microphone) and **Leads** (what customers left, with a CSV download).
- **Customer link**: only the conversation. Each customer has their own memory; one conversation runs
  at a time and the next customer waits in line.

Type a message — the avatar answers aloud with lip sync (**Mute** for text-only replies) — or tap
**🎤 Tap to talk** on the avatar. Use headphones, with the computer's own microphone: a Bluetooth
headset's microphone turns the headset into phone-call quality.

The avatar's video goes out as H.264 High profile — NVENC on an NVIDIA GPU, else x264 — at ~2.5 Mbit/s
for the 1024x1024, 25 fps avatar (0.7 Mbit/s for a 512x512 one), lower on a slow connection; it needs
`ffmpeg` on the PATH. The voice goes as Opus (~56 kbit/s) on https links and on `localhost`. On a plain `http://` LAN
address the browser cannot decode the video, so it falls back to JPEG frames (~7 Mbit/s); add `&jpeg` to
a link to force JPEG, `&pcm` for uncompressed voice. Testing on this PC itself, open
`http://localhost:8000/?token=<the owner link's token>`: the tunnel sends everything to Cloudflare and back.

For a truly lifelike avatar, build it from a short video of a real person:
[docs/RECORDING_A_VIDEO_AVATAR.md](docs/RECORDING_A_VIDEO_AVATAR.md).

## Linux

The same app on Ubuntu 24.04, also under WSL2 (tested with an RTX 4070 Ti SUPER). Needs uv, git, curl,
Node.js 20.19+ or 22.12+ (Vite 8), and an NVIDIA driver for CUDA 13. No sudo needed: uv and Node.js install
into `~/.local`. For the avatar's video an `ffmpeg` on the PATH with NVENC (GPU) is best (`sudo apt install
ffmpeg`, or without sudo `conda create -p ~/.local/conda-ffmpeg -c conda-forge ffmpeg` and a link in
`~/.local/bin`); without one the avatar worker's bundled ffmpeg encodes on the CPU (libx264).

```bash
scripts/setup_linux.sh --avatar          # first time (idempotent): .env, backend venv + CUDA torch, llama-server,
                                         #   models, frontend; --avatar adds the avatar worker, its weights and
                                         #   the development photo (data/avatars/dev_girl/photo.png)
avatar/musetalk/.venv/bin/python avatar/musetalk/build_avatar.py --photo data/avatars/dev_girl/photo.png --id dev_girl_hd
                                         # once, ~1.5-2 h: the avatar profile's avatar (with the backend stopped)
scripts/run.sh                           # http://127.0.0.1:8000, profile from .env (AVATAR_PROFILE)
scripts/run.sh --profile avatar --build  # avatar profile, frontend rebuilt first
scripts/run.sh --host 0.0.0.0            # LAN over plain http (text only on other devices): tokens are printed
scripts/tunnel.sh --profile avatar       # internet + microphone: owner and customer links (logs/*_url.txt)
scripts/tunnel.sh --profile avatar --detach
scripts/run.sh --stop                    # backend, its helpers and our quick tunnel
```

Paths in the Windows commands below become `.venv/bin/python` and `avatar/musetalk/.venv/bin/python`.
Not ported yet: LAN HTTPS (`run.ps1 -Https`; use the tunnel) and the XTTS and LiteAvatar setup scripts (so the
`clone` profile falls back to the stock voice, and `pick_clone_voice.py` does not run). Under WSL2 `--host 0.0.0.0`
prints WSL's internal address, which other devices cannot reach: use the tunnel.

## Houston Party Network (profile `hpn`)

The avatar as the **Event helper** of h-townparty.com for HPN's customers: every answer is the website helper's
own (Craig's 111 topics, live prices, the venue schedule), word for word; the page looks like HPN's site and its
buttons open h-townparty.com. The LLM writes no answers and keeps no memory or leads: it only picks which of
HPN's questions an unclear (often misheard) question meant. Details and what differs from the website:
[`knowledge/hpn/README.md`](knowledge/hpn/README.md).

```bash
.venv/bin/python scripts/hpn_refresh.py      # the knowledge from the HPN project + today's prices and nights
scripts/tunnel.sh --profile hpn --detach     # owner and customer links (logs/tunnel_url.txt, customer_url.txt)
```

**Several customers at once:** `conversations: 3` in the profile. Each conversation has its own avatar stream
(its own idle video, reply, face and video encoder) from one avatar process that loads the models and the
avatar's frames once (`avatar/musetalk/worker.py --streams`). Listening and idle cost almost nothing; lip sync
runs on the GPU only while she speaks, so when two replies overlap they take turns batch by batch and each keeps
pace with its voice by showing fewer frames for a moment. More customers wait in line as before (a conversation
inactive for 90 s gives its place up). Measured on this PC: upload ~180 Mbit/s (a customer needs ~2.5), 12 NVENC
encoders at once without error; the limit is the lip sync (~24 ms of GPU per 1024x1024 frame, 40 ms per frame
at 25 fps). Load test: `LOAD_CUSTOMERS=3 npx playwright test --project hpn load_hpn` (tests/e2e). More
customers need a second GPU machine.

The avatar's photo had flyaway hair that the animation made stand out; `avatar/musetalk/clean_hair.py` removes
it before the build (`photo_clean.png`, avatar `dev_girl_hd_clean`).

## Make it yours

```powershell
# your avatar from one photo, then set avatar.avatar_id: me in the profile. The full build (six 30-s
# expression loops, head gestures, talking and body motion) is 1024x1024 at 25 fps with Real-ESRGAN
# sharpening by default (~1.5-2 h on an RTX 4070 Ti SUPER). On the RTX 3060 build 512x512 at 20 fps
# (--size 512 --fps 20, ~80 min) and use avatar.decoder: taesd. A sharp photo of 1500 px or more gives the
# best result: the sharpening cannot add detail the photo does not have.
avatar\musetalk\.venv\Scripts\python avatar\musetalk\build_avatar.py --photo me.jpg --id me
# (an avatar built before head gestures existed: add them with --id me --gestures-only)
# your voice: record 10-15 s of clean speech as data\voices\me.wav and set tts.clone_reference_wav;
# for the avatar_clone profile also pick the matching base voice (uses XTTS-v2: non-commercial):
.venv\Scripts\python scripts\pick_clone_voice.py --ref data\voices\me.wav
# your documents (PDF, DOCX, Markdown, TXT, HTML): copy them to data\knowledge\, then
.venv\Scripts\python scripts\ingest.py
# web pages: add --url https://... (repeatable); the page is saved to data\knowledge\web\ and indexed
```

Use only photos and voices of people who have agreed to it. XTTS-v2 and InsightFace (used when
building an avatar) are licensed for non-commercial use only - see below.

## Licences and commercial use

The code of this repository and most models are permissively licensed (MIT, Apache-2.0, BSD). These are
**not**, and the standard setup uses them:

| Component | Licence | Used by | Before commercial use |
|---|---|---|---|
| **InsightFace buffalo_l** face detector (`det_10g.onnx`, `2d106det.onnx`) | non-commercial research only | LivePortrait's face cropper in **every avatar build** (`build_avatar.py`, from a photo or a video), so every avatar the `avatar` and `avatar_clone` profiles show was made with it | replace the detector in the build with a commercially usable one and rebuild the avatars, or obtain a commercial licence from InsightFace |
| **XTTS-v2** (Coqui Public Model License) | non-commercial only | the `clone` profile's voice; `scripts/pick_clone_voice.py` (scores the voices with XTTS's speaker encoder: a step of the `avatar_clone` setup); `voice/xtts/bench_clone.py` | do not use the `clone` profile; for `avatar_clone` choose the Kokoro base voice by ear instead of `pick_clone_voice.py` (Coqui closed in 2024, so a commercial licence may not be obtainable) |

Also before a business use:

- the development avatar (`dev_girl*`, from an OpenAvatarChat sample photo) and the development voice are
  for development - use a person's likeness and voice only with their written consent;
- LiteAvatar's weights (renderer comparison only, not part of the app) state no licence;
- Cloudflare quick tunnels (`tunnel.ps1` / `tunnel.sh`) are meant for testing; serve customers through a
  tunnel of your own Cloudflare account or your own HTTPS server;
- espeak-ng (GPL-3.0, Kokoro's fallback phonemiser) and ffmpeg/x264 (LGPL/GPL, an external program) are
  used unmodified and not shipped; their terms apply mainly when you distribute them.

This summary is not legal advice. Every component and its licence:
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

## Tests

```powershell
.venv\Scripts\python -m pytest backend\tests
.venv\Scripts\python scripts\make_fixtures.py     # once: the fake-microphone recordings (not in git)
cd tests\e2e; npm ci; npx playwright install chromium; npx playwright test --project=text
```

The voice tests play `tests/e2e/fixtures/*.wav` as the microphone; `scripts/make_fixtures.py` speaks them
with Kokoro and writes their timings to `fixtures/manifest.json` (run it again after changing the test
sentences). The documents for the RAG tests come from `scripts/make_knowledge_fixtures.py`. When the
backend requires a token, the tests take it from `AVATAR_TOKEN` or `AVATAR_ACCESS_TOKEN`, else from the
links a running tunnel saved in `logs\` (customer tests: `AVATAR_CUSTOMER_TOKEN` or
`logs\customer_url.txt`). The full test list (voice, barge-in, avatar, RAG, crash recovery, 30-minute
soak) is in the "Running" section of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
