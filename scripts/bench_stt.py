"""Speech-recognition robustness on realistic input, through the app's own VAD + Whisper code.

Kokoro voices with different accents (American, British, Indian, Japanese, Spanish, Chinese - their
voice embeddings speaking English) say questions, short answers ("Thank you.", "Okay.") and sentences
with a pause in the middle; conditions: clean, background noise (SNR 10 dB), a quiet microphone,
laptop-microphone bandwidth (300-3400 Hz) and fast speech. Each recording is streamed through
StreamingVAD in 20 ms packets like the browser sends it; every VAD utterance is transcribed. Reports
word error rate (WER), utterances that came out empty, and sentences the VAD cut into pieces.

    .venv\\Scripts\\python scripts\\bench_stt.py --json docs\\benchmarks\\phase6_stt_robustness.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", "E:/AI-models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import butter, resample_poly, sosfilt  # noqa: E402

from app.core.config import STTCfg, VADCfg  # noqa: E402

VOICES = {"am_michael": "US male", "bf_emma": "British female", "hf_alpha": "Indian female",
          "jf_alpha": "Japanese female", "ef_dora": "Spanish female", "zf_xiaobei": "Chinese female"}
# "|" = a 0.9 s pause inside the sentence (people pause to think, especially in a second language)
SENTENCES = ["What is the capital of Japan?", "Can you recommend a good book for the weekend?",
             "How many hearts does an octopus have?", "Thank you.", "Okay.", "No.", "Yes, please tell me more.",
             "I would like to know | what the weather will be like tomorrow.",
             "Could you explain | how rain forms in the clouds?"]
CONDITIONS = ["clean", "noise_10db", "quiet", "laptop_mic", "fast"]
CACHE = ROOT / "logs" / "stt_bench"


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("|", " ")).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


def synth(tts_pipe, voice: str, text: str, speed: float) -> np.ndarray:
    parts = []
    for i, piece in enumerate(text.split("|")):
        if i:
            parts.append(np.zeros(int(0.9 * 24000), np.float32))
        for r in tts_pipe(piece.strip(), voice=voice, speed=speed):
            if r.audio is not None:
                parts.append(r.audio.detach().float().cpu().numpy())
    audio = np.concatenate(parts)
    idx = np.flatnonzero(np.abs(audio) > 0.01 * np.abs(audio).max())   # trim Kokoro's padding
    return resample_poly(audio[idx[0]:idx[-1] + 1], 2, 3).astype(np.float32)


def condition(audio: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    lead, tail = np.zeros(int(0.6 * 16000), np.float32), np.zeros(int(1.6 * 16000), np.float32)
    x = np.concatenate([lead, audio, tail])
    if kind == "noise_10db":
        noise = sosfilt(butter(2, 2000, fs=16000, output="sos"), rng.standard_normal(len(x))).astype(np.float32)
        speech_rms = np.sqrt(np.mean(audio ** 2))
        x = x + noise * (speech_rms / np.sqrt(np.mean(noise ** 2))) / (10 ** (10 / 20))
    elif kind == "quiet":
        x = x * (10 ** (-42 / 20) / np.sqrt(np.mean(audio ** 2)))          # speech at -42 dBFS RMS
    elif kind == "laptop_mic":
        x = sosfilt(butter(4, [300, 3400], btype="band", fs=16000, output="sos"), x).astype(np.float32)
        x = x + rng.standard_normal(len(x)).astype(np.float32) * 0.002
    return np.clip(x, -1, 1).astype(np.float32)


def run_vad_stt(vad, stt, audio: np.ndarray) -> list[str]:
    """Stream like the browser (20 ms packets) and transcribe every VAD utterance."""
    pcm = (audio * 32767).astype(np.int16)
    vad.reset()
    texts, t = [], 100.0
    for i in range(0, len(pcm), 320):
        t += 0.02
        for ev in vad.process(pcm[i:i + 320].tobytes(), assistant_active=False, now=t):
            if ev.kind == "end":
                texts.append(stt.transcribe(ev.audio))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--label", default="current")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from kokoro import KModel, KPipeline

    from app.audio.vad import StreamingVAD
    from app.stt.whisper import WhisperSTT

    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    need = [(v, s, c) for v in VOICES for s in SENTENCES for c in CONDITIONS
            if not (CACHE / f"{v}_{SENTENCES.index(s)}_{c}.wav").exists()]
    if need:
        model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        for v, s, c in need:
            clip = condition(synth(pipe, v, s, 1.25 if c == "fast" else 1.0), c, rng)
            sf.write(CACHE / f"{v}_{SENTENCES.index(s)}_{c}.wav", clip, 16000)
        del model, pipe
    stt = WhisperSTT(STTCfg())
    stt.warmup()
    vad = StreamingVAD(VADCfg())
    rows = []
    for v in VOICES:
        for si, s in enumerate(SENTENCES):
            for c in CONDITIONS:
                audio, _ = sf.read(CACHE / f"{v}_{si}_{c}.wav", dtype="float32")
                t0 = time.perf_counter()
                texts = run_vad_stt(vad, stt, audio)
                hyp = " ".join(t for t in texts if t)
                rows.append({"voice": v, "sentence": s, "condition": c, "segments": len(texts), "texts": texts,
                             "wer": round(wer(words(s), words(hyp)), 3), "empty": not hyp.strip(),
                             "ms": round((time.perf_counter() - t0) * 1000)})
                if rows[-1]["wer"] > 0:
                    print(f"  {v:10} {c:10} wer {rows[-1]['wer']:.2f} seg {len(texts)} | {s!r} -> {texts}", flush=True)

    def agg(sel):
        return {"wer": round(statistics.mean(r["wer"] for r in sel), 3), "n": len(sel),
                "empty": sum(r["empty"] for r in sel), "split": sum(r["segments"] > 1 for r in sel)}

    out = {"label": args.label, "overall": agg(rows),
           "by_condition": {c: agg([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
           "by_voice": {f"{v} ({VOICES[v]})": agg([r for r in rows if r["voice"] == v]) for v in VOICES},
           "short_answers": agg([r for r in rows if len(words(r["sentence"])) <= 2]),
           "pause_sentences": agg([r for r in rows if "|" in r["sentence"]]), "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
