"""Does the avatar smile at the right moments? For happy, sad and plain messages, runs the real LLM with
the real system prompt and reports the faces the avatar shows: while thinking (from the user's words)
and while replying (LLM tag refined by the reply's first words) - before and after the warmth rules.
It measures the LLM's behaviour tag, which is off in the app by default (llm.behavior_tag: false), so the
tag is turned on here for the benchmark's own prompts; otherwise every reply would count as "neutral".

    .venv\\Scripts\\python scripts\\bench_warmth.py --json docs\\benchmarks\\phase6_warmth.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.behavior.manager import from_meta, refine_behavior, smile_for_happy_user, smile_for_user_text  # noqa: E402
from app.behavior.meta import MetaStreamParser  # noqa: E402
from app.behavior.speech_text import clean_delta  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.prompt import ConversationMemory  # noqa: E402
from app.tts.chunker import ClauseChunker  # noqa: E402

CASES = {
    "happy": ["Hi! How are you today?", "Thank you so much for your help!", "I just got the job I applied for!",
              "Today is my birthday!", "Tell me a short joke.", "Good morning! Nice to meet you.",
              "I passed my driving test!", "You're really helpful, thanks."],
    "sad": ["My dog is sick and I'm worried.", "I failed my exam today.", "My grandmother passed away last week.",
            "I'm feeling really stressed about work."],
    "plain": ["What is the capital of Japan?", "How many legs does a spider have?", "Explain how rain forms in one sentence."],
}
# expressions before the warmth change (emotion -> expression; neutral replies had the plain face)
OLD = {"neutral": "neutral", "friendly": "soft_smile", "happy": "smile", "empathetic": "soft_concern",
       "concerned": "concern", "serious": "neutral_firm", "thinking": "thinking", "excited": "smile"}


async def reply(llm: LLMClient, text: str) -> tuple[dict | None, str]:
    memory = ConversationMemory(persona=get_settings().persona_name, behavior_tag=True)   # see the docstring
    parser, spoken = MetaStreamParser(), []
    async for delta in llm.stream_chat(memory.build(text)):
        spoken.append(clean_delta(parser.feed(delta)[1]))
    spoken.append(parser.flush())
    return parser.meta, "".join(spoken).strip()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    llm = LLMClient(get_settings().llm)
    rows = []
    for _ in range(args.rounds):
        for kind, prompts in CASES.items():
            for text in prompts:
                meta, said = await reply(llm, text)
                tag = from_meta(meta)
                warm = smile_for_happy_user(tag, text)          # as session.py does
                chunker = ClauseChunker()
                first = (chunker.feed(said) + chunker.flush() + [""])[0]
                final = refine_behavior(warm, first) or warm
                thinking = smile_for_user_text(text)
                row = {"kind": kind, "prompt": text, "emotion": tag.emotion, "old_reply_face": OLD[tag.emotion],
                       "thinking_face": thinking.expression if thinking else "thinking", "reply_face": final.expression,
                       "reply": said}
                row["old_smiles"] = row["old_reply_face"] == "smile"
                row["smiles"] = "smile" in (row["thinking_face"], row["reply_face"])
                rows.append(row)
                print(f"{kind:5} {tag.emotion:10} think={row['thinking_face']:8} reply={row['reply_face']:12} "
                      f"(before: {row['old_reply_face']:12}) {text[:34]:34} | {said[:40]}", flush=True)
    await llm.aclose()

    def rate(kind: str, key: str) -> float:
        sel = [r for r in rows if r["kind"] == kind]
        return round(sum(r[key] for r in sel) / len(sel), 2)

    warm = {"smile", "soft_smile"}
    out = {"happy_moments_smile_before": rate("happy", "old_smiles"), "happy_moments_smile_now": rate("happy", "smiles"),
           "sad_moments_smile_before": rate("sad", "old_smiles"), "sad_moments_smile_now": rate("sad", "smiles"),
           "plain_warm_face_before": round(sum(r["old_reply_face"] in warm for r in rows if r["kind"] == "plain")
                                           / sum(r["kind"] == "plain" for r in rows), 2),
           "plain_warm_face_now": round(sum(r["reply_face"] in warm for r in rows if r["kind"] == "plain")
                                        / sum(r["kind"] == "plain" for r in rows), 2),
           "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
