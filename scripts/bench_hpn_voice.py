"""Spoken HPN questions: does the Event helper still give the right answer after speech recognition?

The realistic visitor questions of backend/tests/test_hpn_engine.py (VISITORS: a question and the answer that must
give it) are spoken by Kokoro in voices other than the avatar's, transcribed by the profile's Whisper, and asked of
the Event helper (HPN's test schedule, Thursday 8 PM). Reported: how many transcripts get the same answer as the
typed question, which do not (and what Whisper heard), and how many fell back to "not sure" (the cases the
session's LLM re-match is for). Needs the GPU: run it with the backend stopped.

    .venv/bin/python scripts/bench_hpn_voice.py                    # af_bella and am_michael
    .venv/bin/python scripts/bench_hpn_voice.py --voices bf_emma --json logs/bench_hpn_voice.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from model_home import hf_home  # noqa: E402

os.environ.setdefault("HF_HOME", hf_home())
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from tests.hpn_fixture import THURSDAY_8PM, fixture_knowledge  # noqa: E402
from tests.test_hpn_engine import VISITORS  # noqa: E402

from app.core.config import Settings, TTSCfg  # noqa: E402
from app.hpn.engine import Engine  # noqa: E402
from app.stt.whisper import WhisperSTT  # noqa: E402
from app.tts.kokoro_tts import KokoroTTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", default="af_bella,am_michael", help="Kokoro voices that play the customer")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    settings = Settings()
    stt = WhisperSTT(settings.stt)
    stt.warmup()
    engine = Engine(fixture_knowledge(), "Emma")
    rows, ok, fallbacks = [], 0, 0
    for voice in [v.strip() for v in args.voices.split(",") if v.strip()]:
        tts = KokoroTTS(TTSCfg(engine="kokoro", voice=voice, speed=1.0))
        for question, intent in VISITORS:
            result = tts.synthesize(question)
            audio = torch.from_numpy(np.asarray(result.audio, dtype=np.float32))
            audio16 = AF.resample(audio, result.sample_rate, 16000).numpy()
            pad = np.zeros(int(0.3 * 16000), np.float32)
            heard = stt.transcribe(np.concatenate([pad, audio16, pad]))
            got = engine.ask(heard, now=THURSDAY_8PM).intent
            typed = engine.ask(question, now=THURSDAY_8PM).intent
            same = got == intent
            ok += same
            fallbacks += got == "fallback"
            rows.append({"voice": voice, "question": question, "heard": heard, "want": intent, "typed": typed,
                         "got": got, "ok": same})
            if not same:
                print(f"[{voice}] {question!r}\n    heard {heard!r}\n    -> {got} (want {intent})")
        del tts
    total = len(rows)
    print(f"\n{ok}/{total} spoken questions got the right answer ({100 * ok / max(total, 1):.1f} %); "
          f"{fallbacks} fell back to 'not sure' (the LLM re-match handles those in the session)")
    if args.json:
        args.json.write_text(json.dumps({"ok": ok, "total": total, "fallbacks": fallbacks, "rows": rows}, indent=1),
                             encoding="utf-8")


if __name__ == "__main__":
    main()
