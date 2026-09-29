# Architecture

Living document — updated at the end of every phase. The research behind these decisions is in
[PHASE0_TECHNICAL_REPORT.md](PHASE0_TECHNICAL_REPORT.md).

## Project decisions (owner answers, 2026-09-29)

| Question | Answer | Consequence |
|---|---|---|
| Commercial use? | No — personal project | Non-commercial licenses (InsightFace, XTTS, F5, Fish) are acceptable; permissive ones still preferred |
| Languages | English | Kokoro stock voice, English STT setting, English embeddings |
| Avatar source | **One photo** | LivePortrait animates the photo offline into short loops (blink, gaze, head motion, emotions); MuseTalk lip-syncs them in real time |
| Voice | Stock **and** cloned | Kokoro by default; a cloning TTS as a switchable profile (one TTS resident at a time) |
| Where the browser runs | Owner's own PC (remote) | HTTPS required for the microphone → HTTPS tunnel or LAN TLS; WebSocket transport (tunnel-friendly) |
| RAM upgrade | No | Keep processes and caches small; load weights straight to GPU; measure RAM every phase |

## Processes

```
Browser (owner's PC) ──HTTPS/WSS──► P1 backend (FastAPI, .venv torch 2.9.1+cu130)
                                        │  VAD (CPU) · STT (GPU) · RAG (CPU) · TTS (GPU) · state machine
                                        ├──HTTP SSE──► P2 llama-server.exe (Qwen3-4B Q4_K_M)
                                        └──WS msgpack─► P3 avatar worker (avatar/musetalk/.venv)
```

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 Safety + research | done | see "Phase 0 results" |
| 1 Text conversation | done | see "Phase 1 results" |
| 2 Voice pipeline | done | see "Phase 2 results" |
| 3 Interruption | done | see "Phase 3 results" |
| 4 Realistic avatar | done | see "Phase 4 results" |
| 5 RAG | done | see "Phase 5 results" |
| 6 Hardening | pending | |

## Phase 0 results (measured on POOH-PC, 2026-09-29)

| Metric | Value |
|---|---|
| torch / CUDA | 2.9.1+cu130 / 13.0, RTX 3060 detected, `check_env.py` passes |
| VRAM used by desktop before start | ~2.0 GiB |
| CUDA context cost (lazy loading) | 111 MiB |
| fp16 / bf16 matmul | 25.6 / 25.9 TFLOPS |
| Host→GPU copy | 12.3 GB/s pinned, 3.5 GB/s pageable |
| Safe VRAM allocation ceiling (0.5 GiB reserve) | 10.25 GiB |
| System RAM available | 1.3–1.9 GiB of 15.7 GiB |

## Phase 1 results (text conversation)

Measured with `scripts/bench_text.py` (14 turns + cancel) and Playwright `tests/e2e/phase1_text.spec.js` (3/3 pass).
Raw data: [benchmarks/phase1_text.json](benchmarks/phase1_text.json).

| Metric | Value |
|---|---|
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M on llama-server b11242 (CUDA 12.4 build), ctx 4096, q8_0 KV |
| llama-server start → ready | 4.2 s (warm disk cache) |
| VRAM used by LLM | ~2.9 GiB |
| llama-server resident RAM | 725 MB with `-lm none` (2,492 MB with default mmap) |
| First token (warm, prefix cached) | 35–50 ms |
| First visible text (after behaviour tag) | **p50 176 ms**, max 278 ms |
| Decode speed | 87 tokens/s |
| Behaviour-tag compliance | 100 % of normal turns |
| Cancel → acknowledged | 5.8 ms; llama-server log confirms `stop: cancel task` |
| History trim (every ~4 turns) | 262 tokens re-processed (~150 ms) thanks to `--cache-reuse 256` |
| First turn after idle | ~265 ms first token (GPU clock ramp-up from P8) |

Design notes:
- The compact tag `[emotion|gesture|intensity]` replaced a JSON header: the JSON version delayed the
  first spoken word by ~230 ms, the compact one by ~130 ms.
- Every outbound packet carries `turn_id`; the single-writer outbox drops data of non-current turns.
- One active session: a new connection takes over (the old one gets close code 4001).

## Phase 2 results (voice pipeline)

Measured with `scripts/bench_voice.py latency --rounds 2` (16 spoken questions streamed in real time over the
real protocol) and Playwright `tests/e2e/phase2_voice.spec.js` (Chromium fake microphone).
Raw data: [phase2_voice.json](benchmarks/phase2_voice.json), [phase2_models.json](benchmarks/phase2_models.json).

| Metric (ms after the user's last voiced audio) | p50 | p90 | max |
|---|---|---|---|
| VAD endpoint (550 ms silence rule) | 572 | 593 | 608 |
| STT done (speculative, started after 120 ms silence) | 619 | 683 | 683 |
| LLM first token | 681 | 735 | 846 |
| LLM first spoken text | 793 | 868 | 980 |
| First clause ready | 924 | 1021 | 1082 |
| TTS first audio ready | 1099 | 1241 | 1400 |
| **End of speech → first audio (client)** | **1109** | **1254** | **1422** |

- Browser (Playwright) run: 1,052 ms end of speech → first audio played.
- Transcripts: 16/16 exact. Speculative STT reused in 16/16 turns.
- Models: Whisper large-v3-turbo fp16 (1.75 GiB VRAM, 190 ms warm), Kokoro-82M (0.55 GiB, ~120 ms/clause, RTF 0.05).
- VRAM with LLM + STT + TTS resident: 7.2 GiB of 12 GiB (≈ 2 GiB of it is the Windows desktop).
- Backend process RAM: ~2.5 GB.

Findings:
- After ~2 s of GPU idle (while the user talks) Whisper takes ~390 ms instead of ~185 ms, even though the SM
  clock reads 1.9 GHz; keeping CPU or GPU busy barely helps (`scripts/bench_gpu_idle.py`). Hiding STT behind
  the endpoint wait (speculative STT on a 120 ms pause) removes it from the critical path.
- Recommended manual setting: NVIDIA Control Panel → Manage 3D settings → Program settings → `python.exe`
  (repo `.venv`) and `llama-server.exe` → *Power management mode: Prefer maximum performance*.

## Phase 3 results (interruption / barge-in)

Measured with `scripts/bench_voice.py bargein --rounds 4` (real-time protocol client) and Playwright
`tests/e2e/phase3_bargein.spec.js` (fake mic says "Wait, stop please. What is two plus two?" while the
assistant tells a story). Raw data: [phase3_bargein.json](benchmarks/phase3_bargein.json),
[phase3_bargein_browser.json](benchmarks/phase3_bargein_browser.json).

| Metric | Value |
|---|---|
| Barge-ins detected | 4/4 (bench) + 1/1 (browser) |
| Detection after the user's speech onset | p50 348 ms, max 362 ms (bench); ~414 ms (browser, approx.) |
| — of which the confidence rule | 260 ms of speech with p ≥ 0.75 while the assistant is active |
| Server cancel fan-out (LLM stream, TTS, queues, state) | ~0.1 ms |
| Browser: interrupt message → audio flushed | 7.7 ms |
| Stale audio packets after the interrupt | 0 (server), 0 blocked needed (client) |
| History records what was actually heard | yes, e.g. "Well, there once was a tiny fox named Flick [interrupted by the user]" |
| Follow-up question answered after barge-in | 4/4 ("Two plus two is four.") |
| Manual Stop button (voice) | passes |

How it works: the VAD keeps running while the assistant speaks, with a stricter threshold; a confirmed
barge-in fires `BARGE_IN` → the turn's asyncio tasks are cancelled (closing the llama-server stream stops
generation), the outbox drops the turn's queued packets, the browser flushes its AudioWorklet buffer and
rejects the turn id, the avatar worker gets `cancel`, and the words heard so far (from Kokoro word
timestamps and the browser's playback position) go into the history. The barge-in audio itself becomes
the start of the next user utterance.

Echo: browser echo cancellation / noise suppression are enabled; the stricter threshold applies while
the assistant is active; and a transcript that mostly repeats the assistant's last words is dropped as echo
(`echo_ignored`). Acoustic echo cannot be tested with a fake microphone — use headphones for development
and try speakers once (see "Manual checks").

## 30-minute stability test (Phases 2 + 3)

Playwright `tests/e2e/soak.spec.js` with a looping fake microphone (a question every ~19 s; long answers
get barged into). Raw data: [phase3_soak_30min.json](benchmarks/phase3_soak_30min.json).

| Metric | Value |
|---|---|
| Duration / turns | 30 min / 95 turns (94 completed, 1 barge-in) |
| Server errors / browser errors | 0 / 0 |
| llama-server restarts | 0 |
| End of speech → first audio | p50 1,210 ms · p90 1,470 ms · max 1,612 ms |
| VRAM used (whole GPU) | 7,459 → 7,791 MiB (max 7,796) — no leak |
| Backend resident RAM | 2,602 → 714 MiB (Windows trims the working set under memory pressure) |

(Package installs and model downloads for Phase 4 were running in parallel during this test.)

## Phase 4 results (realistic avatar from one photo)

![Expressions generated from one photo](images/avatar_expressions.jpg)

*Left to right: the input photo, then the neutral / friendly / happy / concerned / thinking loops rendered
by the face rig.* While speaking, MuseTalk replaces the mouth region:

![Speaking frame](images/avatar_speaking.jpg)

### How it works

1. **Offline, once per photo** (`avatar/musetalk/build_avatar.py`, ~6 min): a procedural *face rig* on
   LivePortrait renders five 7-second loops at 20 fps. All loops share one seeded motion curve
   (head drift ±2°, ~17 blinks/min, micro-saccades with eye contact) and differ only in expression
   (smile, eyebrows, gaze), so the worker can switch loops at the same frame index. MuseTalk then
   pre-computes face boxes (DWPose via ONNX, S3FD), VAE latents and blend masks for each loop.
2. **Live** (`avatar/musetalk/worker.py`, own venv/process): idle frames stream at 20 fps; the loop
   follows the conversation state (listening → friendly, thinking → gaze away) and the LLM's emotion
   tag (empathetic/concerned → concerned, happy → happy, ...). Each TTS clause is lip-synced by MuseTalk
   and every frame carries a pts on the turn's audio timeline; the browser draws frames by the
   *audible* audio position (`AudioContext.getOutputTimestamp`), so sync survives network jitter.
3. **A/V gate**: a turn's first audio is released only when its first 6 lip-synced frames exist
   (batches ramp 2 → 4 → 8), so video never lags at the start of a reply.

### Measurements (RTX 3060, whole stack resident)

| Metric | Value |
|---|---|
| Photo → avatar build | LivePortrait 213 ms/frame (5×140 frames, 2.5 min) + MuseTalk prep ~35 s/loop |
| MuseTalk throughput, alone | 13.2 fps baseline → **24.3 fps** after optimisation (1.21× real time at 20 fps) |
| — optimisation 1 | decode only latent rows ≥ 12 (the blend mask covers the lower half): VAE 44.7 → 27.7 ms/frame, mean abs diff 1.9/255 |
| — optimisation 2 | two-stage pipeline: GPU batch k+1 overlaps CPU composite/JPEG of batch k |
| MuseTalk under continuous LLM load | 9.3 fps (real turns overlap the LLM only briefly) |
| Live render speed (26 clauses) | 18–24 fps, first 6 frames in 363–745 ms |
| A/V sync in the browser (3 runs) | median frame offset 27–30 ms, p90 45–126 ms, 0–11 % frames late |
| End of speech → first audio, **avatar on** (16 turns) | **p50 1,750 · p90 1,914 ms** (gate 6) |
| — with `gate_frames: 2` | p50 1,513 · p90 1,690 ms; 12–14 % frames late, up to ~250 ms at reply start |
| VRAM peak (LLM + Whisper + Kokoro + MuseTalk + desktop) | 9.5 GiB of 12 GiB |
| Emotion → loop | verified: sad → concerned, good news → happy, "let me think" → thinking |

Raw data: [phase4_musetalk_engine.json](benchmarks/phase4_musetalk_engine.json),
[phase4_voice_with_avatar.json](benchmarks/phase4_voice_with_avatar.json),
[phase4_voice_with_avatar_gate2.json](benchmarks/phase4_voice_with_avatar_gate2.json),
[phase4_avatar_browser.json](benchmarks/phase4_avatar_browser.json).

### Lessons / fixes found while measuring

- A force-stopped backend used to leave the avatar worker running; the next backend then connected to
  the stale worker (old code, VRAM held). All helper processes now live in a Windows Job Object with
  kill-on-close, and stale listeners on our ports are cleared before spawning (`app/core/procjob.py`).
- Running a second MuseTalk process next to the full stack exceeded 12 GB: Windows silently spilled
  VRAM to system RAM and throughput fell to 1–6 fps. Keep one avatar engine, and set NVIDIA
  *Sysmem Fallback Policy → Prefer No Sysmem Fallback* so this becomes a clear error instead.
- The GPU is slower for the first batches after it sat idle; the worker runs a 150 ms warm-up burst
  when the conversation enters THINKING (after STT, before the first clause).

### Limitations

- Upper body / hand gestures are not animated (only head, eyes, brows, mouth). The LLM's `gesture`
  field is sent to the browser but not rendered yet.
- MuseTalk renders the mouth at 256×256 inside a 512×512 frame; larger output needs a stronger GPU.
- The development photo is AI-generated (from OpenAvatarChat's resources). Build your own avatar with
  `build_avatar.py --photo <your photo> --id <name>` and set `avatar.avatar_id`.

## Phase 5 results (document-grounded answers)

Pipeline: `scripts/ingest.py` parses PDF (pypdf), DOCX (python-docx), Markdown/TXT and HTML → cleans
(PDF hyphenation, whitespace) → heading-aware chunks (~220 words, 30-word overlap, source/section/page
metadata) → bge-small-en-v1.5 embeddings on CPU → LanceDB. Per turn: embed the question → top-k (4)
cosine search → passages with similarity ≥ 0.62 go into the latest user message as numbered KNOWLEDGE
(keeps the cached prompt prefix intact); if the best match is only 0.50–0.62 the model is told the
documents do not contain the answer; below 0.50 the question is treated as general conversation.
Citations (source, section, page, score) are sent to the UI and shown under the answer — not spoken.

Test set: a deliberately **fictional** knowledge base (bakery, company policy, lamp manual, hiking trail)
in all four formats, so correct answers can only come from retrieval
(`scripts/make_knowledge_fixtures.py`; the PDF is printed by Playwright's Chromium).

| Metric | Value |
|---|---|
| Questions answered correctly with the right source cited | 6/6 |
| Unanswerable question ("Wi-Fi password") declined without inventing | yes |
| General question ("capital of France") answered, no citation | yes |
| Total | **8/8** (`scripts/bench_rag.py`) |
| Retrieval latency (CPU) | embed p50 20 ms + search ~6 ms (max 67 ms incl. first call) |
| Spoken question in the browser (Playwright, `mic_rag.wav`) | exact transcript, correct answer, citation shown, first audio 1,290 ms |
| Index build (10 chunks, 4 files) | 18.8 s (mostly loading the embedding model) |

Raw data: [phase5_rag.json](benchmarks/phase5_rag.json), [phase5_rag_browser.json](benchmarks/phase5_rag_browser.json).

Add your own documents: put files in `data/knowledge/` and run `.venv\Scripts\python scripts\ingest.py`
(the backend loads the index at start-up).

## Running

```powershell
scripts\run.ps1 -Build        # starts backend + llama-server, serves the UI at http://127.0.0.1:8000
scripts\run.ps1 -Stop
cd tests\e2e; npx playwright test --project text
```
