"""Text-to-speech correctness: does the avatar *say* what the text says?

Each sentence goes through the app's speech path (behavior.speech_text.normalize_for_speech + Kokoro,
voice af_heart) and back through Whisper; words that do not come back are listed. Catches
mispronounced numbers, money, times, dates, units and abbreviations (e.g. "100°C" read as
"degrees C"). Sentences: a built-in set of tricky ones, plus the replies of a conversation run
(scripts/bench_conversation.py --json) with --replies.

Usage (stack stopped, it loads its own Kokoro + Whisper):
    .venv\\Scripts\\python scripts/bench_tts.py [--replies conv.json] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", "E:/AI-models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from app.behavior.speech_text import clean_delta, normalize_for_speech  # noqa: E402
from app.core.config import STTCfg  # noqa: E402
from app.tts.chunker import ClauseChunker  # noqa: E402

TRICKY = [
    "The meeting is at 3:30 pm on March 5th.",
    "It costs $12.99, or about 11 euros.",
    "Around 37% of people agree with that.",
    "The population is about 14 million people.",
    "Water boils at 100°C and freezes at 0°C.",
    "Dr. Smith lives on Main St. near the park.",
    "She was born in 1987 and moved to Canada in 2004.",
    "That is version 2.5 of the app.",
    "I'll be there in 10-15 minutes.",
    "It's about 5 km away, roughly 3 miles.",
    "The 21st century started in 2001.",
    "Eat some fruit, e.g. apples or pears.",
    "NASA and the FBI both said OK.",
    "It's 72°F and sunny today.",
    "Take 2-3 tablets every 8 hours.",
    "The flight leaves at 9:05 and lands at 13:40.",
    "Mount Everest is 8,849 meters tall.",
    "That was about ~20 years ago.",
]
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    """Small integer to words (enough to compare Whisper's digits with spoken words)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num_words(n % 100))
    if 1000 < n < 2100 and n % 100 != 0 and n // 100 != 20:        # years: 1987 -> nineteen eighty seven
        return _num_words(n // 100) + " " + _num_words(n % 100)
    if n < 1_000_000:
        return _num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num_words(n % 1000))
    return str(n)


_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}


def _ordinal(n: int) -> str:
    words = _num_words(n).split()
    last = words[-1]
    words[-1] = _ORDINAL.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " " + " ".join(words) + " "


def canon(text: str) -> list[str]:
    """Lower-case words; digits as words, so "37" and "thirty seven" compare equal."""
    t = text.lower().replace("’", "'").replace("%", " percent").replace("&", " and ")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(int(m[1])), t)
    t = re.sub(r"\d+", lambda m: " " + _num_words(int(m.group())) + " ", t)
    t = t.replace("-", " ")
    words = re.sub(r"[^a-z' ]", " ", t).split()
    return [w for w in words if w not in ("uh", "um")]


def missing(ref: list[str], hyp: list[str]) -> list[str]:
    """Reference words that Whisper did not hear (order-insensitive, multiset)."""
    pool = list(hyp)
    out = []
    for w in ref:
        if w in pool:
            pool.remove(w)
        else:
            out.append(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", type=Path, help="bench_conversation.py --json output")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--voice", default="af_heart")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from kokoro import KModel, KPipeline

    from app.stt.whisper import WhisperSTT

    texts = list(TRICKY)
    if args.replies:
        texts += [r["reply"] for r in json.loads(args.replies.read_text(encoding="utf-8"))["rows"] if r["reply"]]
    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
    stt = WhisperSTT(STTCfg())
    stt.warmup()
    rows = []
    for text in texts:
        # clause by clause, as the app speaks (one long clip would also exceed Whisper's 30 s window)
        chunker = ClauseChunker()
        clauses = chunker.feed(clean_delta(text)) + chunker.flush()
        spoken, heard = "", ""
        for clause in clauses:
            said = normalize_for_speech(clause)
            audio = np.concatenate([r.audio.detach().float().cpu().numpy() for r in pipe(said, voice=args.voice)
                                    if r.audio is not None])
            spoken += " " + said
            heard += " " + stt.transcribe(resample_poly(audio, 2, 3).astype(np.float32))
        spoken, heard = spoken.strip(), heard.strip()
        lost = missing(canon(spoken), canon(heard))       # the words meant to be said (unit-tested separately)
        rows.append({"text": text, "spoken": spoken, "heard": heard, "missing": lost,
                     "ok": len(lost) <= max(1, len(canon(spoken)) // 12)})
        print(("ok " if rows[-1]["ok"] else "BAD") + f" {text}\n     said: {spoken}\n    heard: {heard}"
              + (f"\n  missing: {lost}" if lost else ""), flush=True)
    summary = {"sentences": len(rows), "ok": sum(r["ok"] for r in rows),
               "tricky_ok": sum(r["ok"] for r in rows[:len(TRICKY)]), "tricky": len(TRICKY)}
    print(json.dumps(summary))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
