"""Generate spoken test fixtures (fake microphone input) with Kokoro.

A different voice than the assistant's plays the "user". Output: 16 kHz mono PCM16 WAVs
in tests/e2e/fixtures/ plus a JSON manifest with the exact speech timings, so tests
know when speech ends / when a barge-in starts.

Usage: .venv\\Scripts\\python scripts/make_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from model_home import hf_home  # noqa: E402  (scripts/model_home.py: the env, .env, else the platform default)

os.environ.setdefault("HF_HOME", hf_home())

from app.core.config import TTSCfg  # noqa: E402
from app.tts.kokoro_tts import KokoroTTS  # noqa: E402

OUT = ROOT / "tests" / "e2e" / "fixtures"
SR = 16000

# name -> list of ("silence", seconds) | ("say", text)
SCRIPTS = {
    "mic_question": [("silence", 1.5), ("say", "What is the capital of France?"), ("silence", 14.0)],
    "mic_story_bargein": [
        ("silence", 1.5),
        ("say", "Can you tell me a long story about a lighthouse keeper and the sea?"),
        ("silence", 7.0),
        ("say", "Wait, stop please. What is two plus two?"),
        ("silence", 14.0),
    ],
    "mic_resume": [  # user pauses mid-sentence, then continues (tests END_OF_TURN -> resumed)
        ("silence", 1.5), ("say", "I was thinking about"), ("silence", 0.8), ("say", "buying a new bicycle."),
        ("silence", 12.0),
    ],
    # a 1 s pause in the middle of a question (found in user testing: each half became its own turn)
    "mic_pause_unfinished": [("silence", 1.5), ("say", "I would like to know"), ("silence", 1.0),
                             ("say", "what the weather will be like tomorrow."), ("silence", 14.0)],
    "mic_pause_question": [("silence", 1.5), ("say", "Could you explain?"), ("silence", 1.1),
                           ("say", "how rain forms in the clouds."), ("silence", 14.0)],
    "mic_soak": [("silence", 1.0), ("say", "Tell me one short interesting fact."), ("silence", 16.0)],
    "mic_rag": [("silence", 1.5), ("say", "What are the opening hours of the Moonlight Bakery on Sundays?"),
                ("silence", 14.0)],
}


BENCH_QUESTIONS = [
    "What is the capital of Japan?",
    "How many legs does a spider have?",
    "Can you recommend a good hobby for the weekend?",
    "Why is the sky blue?",
    "What should I eat for breakfast tomorrow?",
    "Tell me something nice.",
    "How far away is the moon?",
    "What is your favorite color?",
]
for _i, _q in enumerate(BENCH_QUESTIONS, 1):
    SCRIPTS[f"bench_q{_i}"] = [("silence", 0.8), ("say", _q), ("silence", 1.5)]
SCRIPTS["bench_interrupt"] = [("silence", 0.3), ("say", "Wait, stop. What is two plus two?"), ("silence", 1.5)]
SCRIPTS["bench_long"] = [("silence", 0.8), ("say", "Please tell me a long story about a brave little fox."),
                         ("silence", 1.0)]


def _trim(audio: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """Remove TTS leading/trailing silence so manifest times match real speech."""
    loud = np.flatnonzero(np.abs(audio) > thresh)
    if not len(loud):
        return audio
    pad = int(0.02 * SR)
    return audio[max(0, loud[0] - pad): loud[-1] + pad]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="cpu: while the GPU is busy (e.g. an avatar build)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tts = KokoroTTS(TTSCfg(voice="am_michael", device=args.device))
    manifest = {}
    for name, script in SCRIPTS.items():
        parts, events, t = [], [], 0.0
        for kind, value in script:
            if kind == "silence":
                parts.append(np.zeros(int(value * SR), np.float32))
                t += value
            else:
                result = tts.synthesize(value)
                audio = AF.resample(torch.from_numpy(result.audio), 24000, SR).numpy()
                audio = _trim(0.6 * audio / (np.abs(audio).max() + 1e-6))
                events.append({"text": value, "start_s": round(t, 3), "end_s": round(t + len(audio) / SR, 3)})
                parts.append(audio.astype(np.float32))
                t += len(audio) / SR
        wav = np.concatenate(parts)
        sf.write(OUT / f"{name}.wav", wav, SR, subtype="PCM_16")
        manifest[name] = {"duration_s": round(t, 3), "speech": events}
        print(f"{name}.wav  {t:.1f}s  {events}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
