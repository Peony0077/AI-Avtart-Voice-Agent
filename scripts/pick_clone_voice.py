"""Pick the Kokoro base voice for the light clone voice (engine ``clone_light``), once per reference.

The OpenVoice converter only changes timbre; accent, pitch range and speaking style come from the
Kokoro base voice. Which base voice ends up closest is not predictable from the converter's own
tone-colour distance (measured), so this script tries every English Kokoro voice: synthesise ->
convert -> score with XTTS-v2's speaker encoder (the same metric as the XTTS clone benchmark), and
prints the best one for ``tts.voice``. Needs the XTTS venv (voice\\xtts\\setup.ps1). ~1-2 minutes.
XTTS-v2's weights are licensed for NON-COMMERCIAL use only (CPML): for a commercial setup choose the
base voice by ear instead (docs/THIRD_PARTY_NOTICES.md).

    .venv\\Scripts\\python scripts\\pick_clone_voice.py --ref data\\voices\\me.wav
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from model_home import hf_home  # noqa: E402  (scripts/model_home.py: the env, .env, else the platform default)

os.environ.setdefault("HF_HOME", hf_home())

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from app.core.config import VENV_PYTHON  # noqa: E402
from app.tts.converted_tts import SOURCE_TEXT, ToneColorConverter, reference_clips  # noqa: E402

TEXT = ("Sure! Octopuses have three hearts, and two of them pump blood to the gills. That's a great "
        "question. The capital of France is Paris, a city famous for art, food and history.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True, help="reference recording of the target voice")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--exclude", nargs="*", default=[], help="voices to skip")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "voice_pick")
    args = ap.parse_args()
    from kokoro import KModel, KPipeline

    folder = Path(snapshot_download("hexgrad/Kokoro-82M", allow_patterns=["voices/a*.pt", "voices/b*.pt"]))
    voices = sorted(p.stem for p in (folder / "voices").glob("[ab][fm]_*.pt") if p.stem not in args.exclude)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
    pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model) for lang in "ab"}
    conv = ToneColorConverter()
    ref = conv.embed(reference_clips(args.ref))
    args.out.mkdir(parents=True, exist_ok=True)

    def say(voice: str, text: str) -> np.ndarray:
        return np.concatenate([r.audio.float().cpu().numpy() for r in pipes[voice[0]](text, voice=voice)
                               if r.audio is not None])

    wavs = []
    with torch.inference_mode():
        for v in voices:
            src = conv.embed([(say(v, t), 24000) for t in SOURCE_TEXT])
            path = args.out / f"{v}.wav"
            sf.write(path, conv.convert(say(v, TEXT), 24000, src, ref, args.tau), 24000)
            wavs.append(path)
            print(f"  converted {v}", flush=True)
    xtts_py = ROOT / "voice" / "xtts" / VENV_PYTHON
    scorer = ROOT / "voice" / "xtts" / "bench_clone.py"
    res = subprocess.run([str(xtts_py), str(scorer), "--ref", str(args.ref.resolve()), "--score", *map(str, wavs)],
                         capture_output=True, text=True, cwd=scorer.parent)
    if res.returncode != 0 or "{" not in res.stdout:
        # the scorer's own error (missing venv package, CUDA out of memory, ...) - not just "exit status 1"
        sys.stderr.write(res.stderr[-4000:] + res.stdout[-2000:])
        raise SystemExit(f"the XTTS scorer failed (exit {res.returncode}): {xtts_py} {scorer.name}")
    scores = json.loads(res.stdout[res.stdout.index("{"):])
    ranked = sorted(((re.sub(r"\.wav$", "", Path(k).name), s) for k, s in scores.items()), key=lambda x: -x[1])
    (args.out / "ranking.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    print("\nspeaker similarity to the reference after conversion (higher = closer):")
    for v, s in ranked[:8]:
        print(f"  {v:12} {s:.3f}")
    print(f"\nput this in your profile (config/profiles/avatar_clone.yaml):\n  tts:\n    voice: {ranked[0][0]}")


if __name__ == "__main__":
    main()
