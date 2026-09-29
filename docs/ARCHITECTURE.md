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
| 2 Voice pipeline | pending | |
| 3 Interruption | pending | |
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

## Running

```powershell
scripts\run.ps1 -Build        # starts backend + llama-server, serves the UI at http://127.0.0.1:8000
scripts\run.ps1 -Stop
cd tests\e2e; npx playwright test --project text
```
