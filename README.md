# AI Avatar Voice Agent

A fully self-hosted, real-time conversational avatar: talk to a realistic face in your
browser, interrupt it at any time, and get answers grounded in your own documents.
Everything runs locally on one Windows PC with an RTX 3060 (12 GB) — no paid APIs.

Pipeline: browser mic → Silero VAD → Whisper STT → RAG (LanceDB) → Qwen LLM (llama.cpp)
→ behaviour metadata → Kokoro / cloned-voice TTS → MuseTalk lip-sync on a LivePortrait-animated
photo → audio + video back to the browser.

> Status, measurements and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
> research: [docs/PHASE0_TECHNICAL_REPORT.md](docs/PHASE0_TECHNICAL_REPORT.md) ·
> licenses: [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)

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
| `config/profiles/` | configuration profiles (`default`, `avatar`, `avatar_clone`, `clone`, `lite`) |
| `constraints/` | pins that protect the CUDA PyTorch build |
| `tests/e2e/` | Playwright end-to-end tests (fake microphone) |

## Environment rules

- The backend venv (`.venv`, Python 3.11, **torch 2.9.1+cu130**) must never be replaced.
  Install packages only with `scripts\safe_install.ps1` (checks CUDA before/after).
- Model weights live outside git in `MODEL_HOME` (default `E:\AI-models`).
- The avatar and XTTS workers use their own venvs, so their dependencies never touch the backend.

## First-time setup

```powershell
scripts\safe_install.ps1 -r backend\requirements.txt           # backend packages (torch stays untouched)
scripts\get_llama_cpp.ps1                                      # llama-server (CUDA build) -> tools\llama.cpp
.venv\Scripts\python scripts\download_models.py llm voice rag voice_convert
cd frontend; npm ci; cd ..
powershell -File avatar\musetalk\setup.ps1                     # avatar venv + MuseTalk/LivePortrait (optional)
powershell -File voice\xtts\setup.ps1                          # XTTS-v2 venv + weights (optional)
```

## Run

```powershell
scripts\run.ps1 -Build                     # voice + text, stock voice: http://127.0.0.1:8000
scripts\run.ps1 -Profile avatar            # + realistic avatar
scripts\run.ps1 -Profile avatar_clone      # avatar with a cloned voice
scripts\run.ps1 -Profile clone             # voice only, XTTS-v2 cloned voice (closest to the reference)
scripts\run.ps1 -Stop
```

**From another PC** (the microphone needs HTTPS):

```powershell
scripts\tunnel.ps1 -Profile avatar         # internet: prints https://<random>.trycloudflare.com/?token=...
scripts\tunnel.ps1 -Profile avatar -Detach # same, keeps running in the background (link in logs\tunnel_url.txt)
scripts\run.ps1 -Profile avatar -Https     # same network: prints https://<LAN IP>:8000/?token=...
```

Two links are printed (and saved in `logs\tunnel_url.txt` and `logs\customer_url.txt`):

- **Owner link**: everything, including **Settings** (persona, voice with ▶ Listen, business and lead
  generation, microphone) and **Leads** (what customers left, with a CSV download).
- **Customer link**: only the conversation. Each customer has their own memory; one conversation runs
  at a time and the next customer waits in line.

Type a message — the avatar answers aloud with lip sync (**Mute** for text-only replies) — or tap
**🎤 Tap to talk** on the avatar. Use headphones, with the computer's own microphone: a Bluetooth
headset's microphone turns the headset into phone-call quality.

The avatar's video goes out as H.264 (~1 Mbit/s; needs `ffmpeg` on the PATH) on https links and on
`localhost`. On a plain `http://` LAN address the browser cannot decode it, so it falls back to JPEG
frames (~7 Mbit/s); add `&jpeg` to a link to force JPEG.

For a truly lifelike avatar, build it from a short video of a real person:
[docs/RECORDING_A_VIDEO_AVATAR.md](docs/RECORDING_A_VIDEO_AVATAR.md).

## Make it yours

```powershell
# your avatar from one photo (~15 min incl. head gestures), then set avatar.avatar_id: me in the profile
avatar\musetalk\.venv\Scripts\python avatar\musetalk\build_avatar.py --photo me.jpg --id me
# (an avatar built before head gestures existed: add them with --id me --gestures-only)
# your voice: record 10-15 s of clean speech as data\voices\me.wav and set tts.clone_reference_wav;
# for the avatar_clone profile also pick the matching base voice:
.venv\Scripts\python scripts\pick_clone_voice.py --ref data\voices\me.wav
# your documents (PDF, DOCX, Markdown, TXT, HTML): copy them to data\knowledge\, then
.venv\Scripts\python scripts\ingest.py
# web pages: add --url https://... (repeatable); the page is saved to data\knowledge\web\ and indexed
```

Use only photos and voices of people who have agreed to it. XTTS-v2 and InsightFace (used when
building an avatar) are licensed for non-commercial use only.

## Tests

```powershell
.venv\Scripts\python -m pytest backend\tests
cd tests\e2e; npx playwright test --project=text
```

The full test list (voice, barge-in, avatar, RAG, crash recovery, 30-minute soak) is in the
"Running" section of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
