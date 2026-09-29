# Architecture

Living document — updated at the end of every phase. The research behind these decisions is in
[PHASE0_TECHNICAL_REPORT.md](PHASE0_TECHNICAL_REPORT.md).

## Project decisions (owner answers, 2026-09-29)

| Question | Answer | Consequence |
|---|---|---|
| Commercial use? | No — personal project | Non-commercial licenses (InsightFace, XTTS, F5, Fish) are acceptable; permissive ones still preferred |
| Languages | English | Kokoro stock voice, English STT setting, English embeddings |
| Avatar source | **One photo** | LivePortrait animates the photo offline into short loops (blink, gaze, head motion, emotions); MuseTalk lip-syncs them in real time |
| Voice | Stock **and** cloned | Kokoro by default; XTTS-v2 clone for voice-only use; a "light" clone (Kokoro + OpenVoice V2 converter) next to the avatar — one TTS resident at a time, chosen by profile |
| Where the browser runs | Owner's own PC (remote) | HTTPS required for the microphone → HTTPS tunnel or LAN TLS; WebSocket transport (tunnel-friendly) |
| RAM upgrade | No | Keep processes and caches small; load weights straight to GPU; measure RAM every phase |

## Processes

```
Browser (owner's PC) ──HTTPS/WSS──► P1 backend (FastAPI, .venv torch 2.9.1+cu130)
   (tunnel or LAN TLS, token)           │  VAD (CPU) · STT (GPU) · RAG (CPU) · TTS (GPU) · state machine
                                        ├──HTTP SSE──► P2 llama-server.exe (Qwen3-4B Q4_K_M)
                                        ├──WS binary+JSON─► P3 avatar worker (avatar/musetalk/.venv)   [avatar profiles]
                                        └──WS binary+JSON─► P4 XTTS-v2 worker (voice/xtts/.venv)       [clone profile]
```

All helper processes are supervised by the backend (restart on crash) and live in a Windows Job
Object, so they die with it.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 Safety + research | done | see "Phase 0 results" |
| 1 Text conversation | done | see "Phase 1 results" |
| 2 Voice pipeline | done | see "Phase 2 results" |
| 3 Interruption | done | see "Phase 3 results" |
| 4 Realistic avatar | done | see "Phase 4 results" |
| 5 RAG | done | see "Phase 5 results" |
| 6 Hardening, remote access, cloned voice | done | see "Phase 6 results" |

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
   (batches ramp 2 → 4 → 8), so video never lags at the start of a reply. (Phase 6: 4 frames for the
   stock voice, 6 for the cloned voice — see "Latency work".)

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

## Phase 6 results (hardening, remote access, cloned voice)

### Profiles

One profile = one set of resident models (`config/profiles/*.yaml`, selected with `run.ps1 -Profile`).
VRAM = what the stack itself uses right after start-up (GPU memory used after loading minus before
starting); the Windows desktop and other apps come on top (1.4–3.5 GiB measured on this PC), and
activations add ~0.4 GiB during use.

| Profile | STT | Voice | Avatar | VRAM (stack) | End of speech → first audio (p50 / p90) |
|---|---|---|---|---|---|
| `default` | Whisper large-v3-turbo | Kokoro (stock) | – | 5.2 GiB | **922 / 1,048 ms** (Phase 2: 1,109 / 1,254) |
| `lite` | whisper-small.en, LLM context 3,072 | Kokoro | – | 4.1 GiB | **873 / 1,007 ms** |
| `clone` | large-v3-turbo | XTTS-v2 clone | – | 6.7 GiB | **1,271 / 1,425 ms** (before the latency work: 1,526 / 1,768) |
| `avatar` | large-v3-turbo | Kokoro | MuseTalk | 8.8 GiB | **1,357 / 1,612 ms** (Phase 4: 1,750 / 1,914) |
| `avatar_clone` | large-v3-turbo | light clone | MuseTalk | 9.0 GiB | **1,438 / 1,600 ms** (first version: 2,854 with XTTS) |

All latencies: `scripts/bench_voice.py latency --rounds 2` (16 spoken questions streamed in real time over
the real protocol, 16/16 exact transcripts in every profile), after the latency work below. Raw data:
[phase6_profile_clone.json](benchmarks/phase6_profile_clone.json), [phase6_profile_lite.json](benchmarks/phase6_profile_lite.json)
and the files linked below.

A VRAM guard checks free VRAM before each model loads (needs from measurements, plus a 0.8 GiB
reserve). If a model would not fit it is skipped instead of letting Windows spill VRAM into system RAM
(which silently makes everything ~10× slower): clone voice → stock voice, or voice/avatar disabled. The
reason appears in `/health` → `warnings` and in the UI.

### Latency work

Where the time went (default profile, before): the VAD waits 550 ms of silence before ending a turn;
the speculative STT started at 120 ms of silence took 400–500 ms, so the transcript was ready only
around the endpoint; the LLM then needed ~350 ms for the behaviour tag and the first clause; TTS ~150 ms.

| Change | Effect (measured) |
|---|---|
| **Answer prefetch** (`session/prefetch.py`): RAG + LLM start at the VAD pause, as soon as the speculative transcript exists; the deltas are buffered and used at the endpoint only if the user stayed silent and the text is identical, otherwise cancelled (llama-server stops). A new VAD `resume` event cancels it as soon as the user keeps talking. | used in 16/16 benchmark turns; on its own it gained little, because the STT was the bottleneck |
| **GPU keep-warm while the user talks**: an encoder pass on silence every 400 ms, only while speech is active (never in the silence before the speculative STT, never while the assistant or another GPU job runs). The GPU drops its clocks within ~2 s of idle (P3, 570 MHz); small matmul/memcopy keep-alives, a CPU spin or a higher process priority did not prevent it, the real encoder workload does. | Whisper encoder after 2 s idle 150–300 ms → **133 ms**; speculative STT 400–500 ms → **170–190 ms**; transcript ready at p50 326 ms (was 545) |
| **Lean Whisper decoding**: greedy loop with the same prompt and token suppression instead of `generate()` | identical transcripts on all 15 fixtures (both models); ~18 ms less per utterance |
| **Avatar: no MuseTalk warm-up burst** at THINKING (it now competed with the prefetched LLM) | first clause p50 796 → 730 ms, render time unchanged |
| **Avatar: A/V gate 4 frames** (was 6) for the stock voice | ~70 ms; browser test 0 late frames in 3 runs. `avatar_clone` keeps 6: with the converter on the GPU, 4 let ~5 % of frames arrive late |

Result, end of speech → first audio p50: voice only 1,109 → **922 ms** (the < 1 s stretch goal),
avatar 1,750 → **1,357 ms**, avatar + cloned voice → **1,438 ms**. The ≤ 1.2 s target is met for voice
only (16/16 turns under 1.2 s, 12/16 under 1 s); with the avatar 3/16 turns are under 1.2 s (cloned
voice: 1/16), so **the target is not met with the avatar on this GPU**. What is left is mostly fixed cost: the 550 ms
endpoint rule (a shorter one would cut people off in natural pauses), TTS, and rendering the first
lip-synced frames while the LLM is still generating on the same GPU.

Checked again after these changes: barge-in 4/4 (detection p50 354 ms, 0 stale packets, follow-ups
answered), browser barge-in (342 ms, flush 1.3 ms), RAG answer with citations through the prefetch
path, text/voice/avatar/recovery tests. Raw data: [phase6_voice_prefetch.json](benchmarks/phase6_voice_prefetch.json),
[phase6_avatar_prefetch.json](benchmarks/phase6_avatar_prefetch.json) (gate 6 + burst),
[phase6_avatar_gate6_noburst.json](benchmarks/phase6_avatar_gate6_noburst.json),
[phase6_avatar_gate4.json](benchmarks/phase6_avatar_gate4.json),
[phase6_bargein_prefetch.json](benchmarks/phase6_bargein_prefetch.json).

### Remote access (browser on another PC)

Browsers only allow the microphone on HTTPS (or localhost), so there are two ways in, both protected by
a random access token (`AVATAR_ACCESS_TOKEN`; the WebSocket is closed with code 4003 and the UI shows
"access denied (token)" for a wrong or missing token):

- **Internet:** `scripts\tunnel.ps1 -Profile avatar` starts the backend and a free Cloudflare quick
  tunnel (no account; `cloudflared` is downloaded to `tools/` on first use) and prints
  `https://<random>.trycloudflare.com/?token=...`. Traffic passes through Cloudflare; the link changes
  each time. Measured: the Phase 2 browser test through the tunnel gave 1,171 ms end of speech → first
  audio (1,052 ms locally).
- **LAN:** `scripts\run.ps1 -Profile avatar -Https` creates a self-signed certificate
  (`scripts/make_cert.py`, written to the git-ignored `certs/`), listens on all interfaces and prints
  `https://<LAN IP>:8000/?token=...`. Accept the certificate warning once.

The page keeps the token for the browser tab (sessionStorage), so reconnects keep working.

### Cloned voice: two engines

| | XTTS-v2 (profile `clone`) | Light clone (profile `avatar_clone`, engine `clone_light`) |
|---|---|---|
| How | zero-shot cloning TTS, streamed in ~1 s chunks, own process/venv | Kokoro speech → OpenVoice V2 tone-colour converter (timbre only), in the backend |
| Speaker similarity to the reference¹ | **0.749** | **0.654** (0.55 before conversion) |
| Unrelated voice, for scale¹ | 0.429 | 0.429 |
| Intelligibility (Whisper transcript of a 29-word sample) | 0 word errors | 0 word errors |
| Cost | real-time factor 0.54–0.57, first chunk ~465 ms | Kokoro 83 ms + converter 66 ms per clause (real-time factor 0.05) |
| VRAM | ~2.2–2.6 GiB | ~0.2 GiB |
| License | CPML (non-commercial) | Kokoro Apache-2.0, OpenVoice MIT |

¹ Cosine similarity of XTTS-v2 speaker-encoder embeddings (`voice/xtts/bench_clone.py --score`). The
development reference `data/voices/dev_emma.wav` is itself a Kokoro voice (bf_emma), because no real
recording was available; bf_emma was excluded when choosing the base voice. Results with a real
recording will differ: record 10–15 s of clean speech (with the speaker's consent).

The light clone only changes timbre; accent and speaking style come from the Kokoro base voice, and the
best base voice cannot be predicted from the converter's own distance (measured). So
`scripts/pick_clone_voice.py --ref <wav>` tries all 28 English Kokoro voices (synthesise → convert →
score) and prints the one to put in `tts.voice` (for the dev reference: bf_isabella 0.655, bf_alice
0.644, bm_lewis 0.614, ..., af_heart 0.487). Kokoro now uses British G2P for `b*` voices.

**Why XTTS is not used next to the avatar** (measured, [phase6_gpu_contention.json](benchmarks/phase6_gpu_contention.json)):
XTTS's autoregressive GPT and MuseTalk time-slice the GPU. With both running, MuseTalk drops from 24.3
to 10.4 fps and XTTS's real-time factor rises from 0.57 to 0.75; per second of speech they need ~1.4 s
of GPU time. XTTS cannot be made cheaper here: non-streaming synthesis costs the same (the vocoder is
not the bottleneck) and fp16 autocast is slower. In the live stack the first reply audio came after
2.9 s and the lips fell behind the voice.

Fixes that came out of this work (they also protect the XTTS path):
- The avatar worker's render loop died on a segment shorter than one video frame (MuseTalk's feature
  code concatenates an empty list); the next turns then waited 1.5 s per chunk and showed no lip sync.
  Segments are now isolated (an error is logged and the segment released), empty ones are skipped, and
  contiguous segments of a turn get the previous 200 ms of audio as left context, so the lips do not
  close at every ~1 s chunk boundary.
- Streaming chunks that arrive while the avatar is rendering are merged into one segment.

### Avatar + cloned voice (profile `avatar_clone`)

| Metric | Value |
|---|---|
| End of speech → first audio (16 turns, `bench_voice.py`) | p50 1,438 · p90 1,600 · max 1,732 ms with the latency work above (1,786 before it; 2,854 with XTTS) |
| Waits for avatar frames > 1.5 s | 0 (was 32 in 8 turns with XTTS) |
| Live render speed | 20–23 fps |
| Browser A/V sync (Playwright `phase4_avatar`, 5 runs, gate 6) | 6 of 807 lip-sync frames late; frame offset p50 20–32 ms, p90 37–49 ms |
| VRAM | stack 9.0 GiB (avatar with stock voice: 8.8 GiB); XTTS instead of the converter would need ~2.2 GiB more |

Raw data: [phase6_avatar_clone.json](benchmarks/phase6_avatar_clone.json),
[phase6_clone_light.json](benchmarks/phase6_clone_light.json),
[phase6_clone_voice.json](benchmarks/phase6_clone_voice.json) (XTTS),
[phase6_voice_clone.json](benchmarks/phase6_voice_clone.json) (XTTS profile latency).

### Lip sync when the PC is busy

The first 30-minute avatar soak (morning, quiet desktop) passed with 2.5 % late frames. Later the same
day the PC was in use — Chrome with ~77 processes and ~8 GB RAM, Slack, VS Code, a screen recorder,
AnyDesk, Battle.net; 300–600 MB RAM free, up to 3.5 GiB VRAM taken by other apps — and soaks showed
12–20 % late frames (a frame > 120 ms behind the audio on screen). Measured step by step:

| Question | Measurement | Answer |
|---|---|---|
| Caused by the latency work? | A/B, 10 min each, same conditions: current code 11.9 % late vs. the settings of the morning run 12.6 % | no |
| Browser decoding? | `createImageBitmap` p50 2 ms, p90 5 ms | no |
| Network / outbox? | outbox backlog ≤ 3 while forwarding speech frames | no |
| Renderer? | frames were already late *when the backend forwarded them* (p90 up to 330 ms on long answers); MuseTalk alone measured 20.4–22.9 fps instead of 24.3 (other apps on the GPU) | **yes** — the renderer has almost no margin over real time on a busy GPU and drifts behind on long answers |

Fixes in the avatar worker:
- **Catch-up:** the browser already reports its playback position every 100 ms; the backend forwards it
  to the worker, which skips frames that could only reach the browser more than 100 ms after their
  audio (`avatar.late_ms`). The browser keeps showing the newest frame, so under load the lips update
  a little less often but stay in sync instead of drifting. Stalled playback (no report for 250 ms)
  is not extrapolated.
- **Per-frame sending:** frames go out as soon as each one is composited instead of after the whole
  batch of 8 (~80 ms), which made the first frames of every batch the late ones.

| 10-minute soak, busy PC | late frames on screen | arrived > 100 ms late | A/V offset p90 | frames skipped |
|---|---|---|---|---|
| before | 11.9–20 % | 8.4 % | 45–441 ms | – |
| catch-up (whole-batch timing estimate) | 7.5 % | 3.4 % | 47 ms | 6 % (too eager) |
| **catch-up per frame + per-frame sending** | **3.9 %** | 2.8 % | **39 ms** | 3 % |

Raw data: [phase6_lipsync_under_load.json](benchmarks/phase6_lipsync_under_load.json).

The soak test now also checks memory growth of *our* processes (PyTorch allocators of the backend and
the avatar worker, from the 3-minute mark), because the whole-GPU number moves by ±1 GiB with the
other desktop apps; whole-GPU usage is only checked against the spill limit.

### Robustness

Playwright `tests/e2e/phase6_recovery.spec.js` kills processes in the middle of a live session
(profile `avatar_clone`, [phase6_recovery_browser.json](benchmarks/phase6_recovery_browser.json)):

| Test | Result |
|---|---|
| Kill `llama-server`, ask a question right away | answered correctly ("Two plus three is five.") 9.5–12.6 s after the kill (2 runs); watchdog restart #1 |
| Kill the avatar worker | the next reply plays audio-only; worker back after 13–26 s; lip sync resumes |
| Wrong access token | rejected (4003), UI shows "access denied (token)"; right token: 4/4 text tests pass |
| Reconnect / second tab | reconnect with backoff; a new tab takes over (4001) — Phases 1–3 |

Found and fixed by these tests:
- A turn that hit the LLM while it was restarting failed immediately. The client now waits for the
  supervisor's restart (connection refused, or HTTP 503 while the model loads) and retries once — only
  if nothing was streamed yet. `/health` has `llm_ready`.
- `logs/backend.log` stopped recording as soon as Kokoro loaded, since Phase 2: Kokoro's `__init__`
  calls `logger.remove()`. Kokoro is now imported before logging is configured.
- Unit tests for both, for the chunk merging (`backend/tests/test_recovery_units.py`), the answer
  prefetch (`test_prefetch.py`), the VAD `resume` event and the avatar catch-up rule (`test_avatar_catchup.py`) — 55 unit tests in total.

### Observability

- `GET /health`: profile, resident models, restarts (LLM, avatar), VRAM guard warnings, load times.
- `GET /health/gpu`: VRAM and RAM (whole GPU, plus the PyTorch memory of the backend and the avatar
  worker); warns below 400 MiB free VRAM. `/health` → `avatar.stats` has render speed per segment and
  the number of frames skipped to stay in sync.
- Per turn (`metrics`): `avatar_forward` — how late the speech frames already were when the backend
  forwarded them (tells renderer delays apart from network/browser delays).
- `GET /api/metrics?last=50`: p50/p90 per pipeline stage (ms after the end of the user's speech) over
  the last completed turns; one JSON line per turn in `logs/metrics.jsonl`.
- Logs: `logs/backend.log`, `llama-server.log`, `avatar-worker.log`, `xtts-worker.log`.

### 30-minute soak with avatar + cloned voice

Playwright `tests/e2e/soak.spec.js`, profile `avatar_clone`, looping fake microphone (a question every
~19 s), on the final code and on a busy PC (see "Lip sync when the PC is busy"; RAM free 3.3 GB at the
start, 235 MB at the lowest). Raw data: [phase6_soak_avatar_clone_30min.json](benchmarks/phase6_soak_avatar_clone_30min.json).

| Metric | Value |
|---|---|
| Duration / turns | 30 min / 95 turns, all completed; answers were started at the pause in 95/95 |
| Server errors / browser errors | 0 / 0 |
| llama-server / avatar worker restarts | 0 / 0 |
| End of speech → first audio | p50 1,744 ms · p90 1,872 ms · max 2,272 ms (longer questions than the benchmark, busy PC) |
| Lip sync | 17,043 speech frames; 4.0 % of screen refreshes showed a frame > 120 ms behind the audio; A/V offset p50 29 ms, p90 46 ms; 489 frames (2.8 %) skipped to stay in sync |
| GPU memory of our processes | backend 2,760 MiB + avatar worker 3,484 MiB, +48 MiB from minute 3 to the end — no leak |
| Whole GPU | 11,037 → 11,423 MiB of 12,288 (flat after minute 6) |

The first run of this test (morning, quiet desktop, before the latency work) also passed its checks
(95/95 turns, 2.5 % late frames); it was reported as failed only because stopping a 30-minute Playwright
trace exceeded the time limit, so tracing is now off for the soak
([phase6_soak_avatar_clone_run1.json](benchmarks/phase6_soak_avatar_clone_run1.json)).

### Manual checks (cannot be automated here)

- **Acoustic echo:** the fake microphone has no room echo. Use headphones; try speakers once and, if the
  avatar interrupts itself, raise `vad.bargein_threshold` or keep headphones.
- **Your real microphone and network:** open the tunnel link on your PC and hold a conversation.
- **NVIDIA Control Panel** → Manage 3D settings: *Power management mode = Prefer maximum performance*
  (for `python.exe` in the repo venvs and `llama-server.exe`) and *CUDA - Sysmem Fallback Policy =
  Prefer No Sysmem Fallback*.
- **By eye and ear:** avatar realism and how close the cloned voice sounds.

## Running

First-time setup (backend venv, models, avatar and voice workers) is described in
[PHASE0_TECHNICAL_REPORT.md](PHASE0_TECHNICAL_REPORT.md); after that:

```powershell
scripts\run.ps1 -Build                     # build the UI, start everything: http://127.0.0.1:8000
scripts\run.ps1 -Profile avatar            # realistic avatar, stock voice
scripts\run.ps1 -Profile avatar_clone      # realistic avatar, cloned voice
scripts\run.ps1 -Profile clone             # voice only, XTTS-v2 cloned voice
scripts\run.ps1 -Profile avatar -Https     # LAN: https://<LAN IP>:8000/?token=...
scripts\tunnel.ps1 -Profile avatar         # internet: https://<random>.trycloudflare.com/?token=...
scripts\run.ps1 -Stop
```

Your own avatar, voice and documents:

```powershell
# avatar from one photo (~6 min), then set avatar.avatar_id: me in the profile
avatar\musetalk\.venv\Scripts\python avatar\musetalk\build_avatar.py --photo me.jpg --id me
# cloned voice: record 10-15 s of clean speech as data\voices\me.wav, set tts.clone_reference_wav,
# and for avatar_clone pick the base voice:
.venv\Scripts\python scripts\pick_clone_voice.py --ref data\voices\me.wav
# documents: put PDF/DOCX/MD/TXT/HTML files in data\knowledge\, then
.venv\Scripts\python scripts\ingest.py
```

Tests (backend running with the matching profile):

```powershell
.venv\Scripts\python -m pytest backend\tests                 # unit tests
cd tests\e2e
npx playwright test --project=text                           # Phase 1 (+ token test with $env:AVATAR_TOKEN)
npx playwright test --project=voice phase2                    # voice (default profile)
$env:FAKE_MIC='mic_story_bargein.wav'; npx playwright test --project=voice phase3   # barge-in
$env:FAKE_MIC='mic_rag.wav'; npx playwright test --project=voice phase5            # documents
npx playwright test --project=voice phase4 phase6             # avatar, crash recovery (avatar profiles)
$env:FAKE_MIC='mic_soak.wav'; $env:FAKE_MIC_LOOP='1'; npx playwright test --project=voice soak
```
