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
| 4 Realistic avatar | pending | |
| 5 RAG | pending | |
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

## Running

```powershell
scripts\run.ps1 -Build        # starts backend + llama-server, serves the UI at http://127.0.0.1:8000
scripts\run.ps1 -Stop
cd tests\e2e; npx playwright test --project text
```
