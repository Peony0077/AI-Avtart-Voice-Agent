"""Which head gesture does the LLM choose? Sends prompts where a gesture is natural (agree -> nod,
disagree -> shake, curious -> head_tilt, ...) with the real system prompt to the running llama-server
and reports the tag it writes, tag compliance, and the time to the first spoken text.
The tag is off in the app by default (llm.behavior_tag: false - the behaviour is then read from the reply's
words), so this benchmark always turns it on for its own prompts; otherwise it would measure nothing.

    .venv\\Scripts\\python scripts\\bench_gestures.py --json docs\\benchmarks\\phase6_gesture_choice.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.behavior.manager import from_meta, refine_behavior  # noqa: E402
from app.behavior.meta import MetaStreamParser  # noqa: E402
from app.behavior.speech_text import clean_delta  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.prompt import ConversationMemory  # noqa: E402
from app.tts.chunker import ClauseChunker  # noqa: E402

# (prompt, gestures that would fit)
CASES = [
    ("Exercise is good for mental health, right?", {"nod", "small_nod"}),
    ("So water boils at 100 degrees Celsius at sea level, correct?", {"nod", "small_nod"}),
    ("Drinking enough water every day is important, isn't it?", {"nod", "small_nod"}),
    ("The sun goes around the Earth, doesn't it?", {"shake"}),
    ("Can people breathe underwater without any equipment?", {"shake"}),
    ("Is it safe to look directly at the sun?", {"shake"}),
    ("I'm thinking about learning a new language.", {"head_tilt", "small_nod"}),
    ("Guess what happened to me today!", {"head_tilt"}),
    ("I wonder whether cats dream when they sleep.", {"head_tilt"}),
    ("My dog is sick and I'm really worried.", {"small_nod", "head_tilt", "none"}),
    ("I failed my exam today.", {"small_nod", "head_tilt", "none"}),
    ("I just got the job I applied for!", {"smile", "nod", "small_nod"}),
    ("What is the capital of Japan?", {"none", "small_nod"}),
    ("How many legs does a spider have?", {"none", "small_nod"}),
]


async def ask(llm: LLMClient, memory: ConversationMemory, text: str) -> dict:
    parser = MetaStreamParser()
    t0 = time.perf_counter()
    first_text = None
    spoken = []
    async for delta in llm.stream_chat(memory.build(text)):
        _meta, speech = parser.feed(delta)
        speech = clean_delta(speech)                 # as the pipeline does (drops emojis, markdown)
        if speech and first_text is None:
            first_text = (time.perf_counter() - t0) * 1000
        spoken.append(speech)
    spoken.append(parser.flush())
    behavior = from_meta(parser.meta)
    reply = "".join(spoken).strip()
    chunker = ClauseChunker()
    first_clause = (chunker.feed(reply) + chunker.flush() + [""])[0]
    final = refine_behavior(behavior, first_clause) or behavior       # what the avatar plays (session.py)
    return {"tag_ok": parser.meta is not None, "llm_gesture": behavior.gesture, "gesture": final.gesture,
            "emotion": behavior.emotion,
            "intensity": final.intensity, "first_text_ms": round(first_text or 0), "reply": reply}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = get_settings()
    llm = LLMClient(cfg.llm)
    rows = []
    for _ in range(args.rounds):
        for text, fits in CASES:
            # behavior_tag=True: the system prompt with the tag rules, whatever the profile says (see above)
            memory = ConversationMemory(persona=cfg.persona_name, max_turns=cfg.llm.history_turns, behavior_tag=True)
            r = await ask(llm, memory, text)
            r.update(prompt=text, fits=r["gesture"] in fits, llm_fits=r["llm_gesture"] in fits)
            rows.append(r)
            print(f"{r['llm_gesture']:9} -> {r['gesture']:9} {'ok ' if r['fits'] else '-- '} {text[:46]:46} | "
                  f"{r['reply'][:50]}", flush=True)
    await llm.aclose()
    out = {"cases": len(rows), "tag_compliance": sum(r["tag_ok"] for r in rows) / len(rows),
           "fitting_gesture_llm_tag": sum(r["llm_fits"] for r in rows) / len(rows),
           "fitting_gesture_final": sum(r["fits"] for r in rows) / len(rows),
           "gestures_llm_tag": dict(Counter(r["llm_gesture"] for r in rows)),
           "gestures_final": dict(Counter(r["gesture"] for r in rows)),
           "first_text_ms_p50": statistics.median(r["first_text_ms"] for r in rows), "rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
