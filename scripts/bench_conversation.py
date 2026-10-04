"""Conversation quality over the real WebSocket protocol (typed turns, one session).

Owner: "conversation is not working correctly". Checks what a user notices in a spoken chat:
remembers what they said (name, city), follows up in context ("and its population?"), answers
facts and simple arithmetic correctly, keeps replies short and spoken (no lists, markdown, emojis,
leftover tags) unless asked for detail, and how fast the first words come.

Usage: .venv\\Scripts\\python scripts/bench_conversation.py [--url ws://127.0.0.1:8000/ws] [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import websockets

# (user text, words of which at least one must be in the reply (lower case), max sentences)
SCRIPT = [
    ("Hi! My name is Sam and I live in Osaka.", ["sam"], 3),
    ("What's the capital of Japan?", ["tokyo"], 3),
    ("And roughly how many people live there?", ["million"], 3),
    ("What's my name?", ["sam"], 2),
    ("Which city do I live in?", ["osaka"], 2),
    ("Explain in detail how a rainbow forms.", ["light", "sunlight"], 12),
    ("In one word: what colour is the sky on a clear day?", ["blue"], 1),
    ("What is fifteen percent of eighty?", ["12", "twelve"], 2),
    ("In which year did people first walk on the moon?", ["1969"], 2),
    ("I'm feeling a bit nervous about a job interview tomorrow.", ["interview", "nervous", "nerves", "normal", "prepare", "natural"], 4),
    ("Can you remind me what we talked about first?", ["name", "sam", "osaka", "japan", "tokyo"], 3),
    ("Thanks, that's all for now.", ["welcome", "pleasure", "anytime", "glad", "bye", "care"], 2),
]
_SENTENCE = re.compile(r"[.!?]+(\s|$)")
_FORMAT = re.compile(r"(^|\n)\s*([-*•]|\d+\.)\s|[#*_`]|\[|\]|https?://")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


async def turn(ws, text: str, speaker: bool = False) -> dict:
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "user_text", "text": text}))
    first, tid, reply = None, None, ""
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=90)
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        if msg["type"] == "user_message":
            tid = msg["turn_id"]
        elif msg["type"] == "llm_delta" and msg.get("turn_id") == tid and first is None:
            first = time.perf_counter()
        elif speaker and msg["type"] == "tts_done" and msg.get("turn_id") == tid:
            # spoken replies (the LLM pauses after the first clause and continues): no browser plays
            # them here, so report the playback as finished
            await ws.send(json.dumps({"type": "playback", "turn_id": tid, "event": "ended", "played_ms": msg["total_ms"]}))
        elif msg["type"] == "assistant_done" and msg.get("turn_id") == tid:
            reply = msg["text"]
            break
    return {"first_ms": round((first - t0) * 1000) if first else None,
            "done_ms": round((time.perf_counter() - t0) * 1000), "reply": reply}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--speaker", action="store_true", help="spoken replies (TTS + avatar, the LLM pauses after the first clause)")
    args = ap.parse_args()
    rows = []
    async with websockets.connect(args.url, max_size=None) as ws:
        while json.loads(await ws.recv())["type"] != "session":
            pass
        if args.speaker:
            await ws.send(json.dumps({"type": "speaker", "enabled": True}))
        for text, expect, max_sentences in SCRIPT:
            r = await turn(ws, text, args.speaker)
            reply = r["reply"]
            low = reply.lower()
            sentences = max(1, len(_SENTENCE.findall(reply)))
            r.update(user=text, sentences=sentences,
                     ok_content=any(w in low for w in expect),
                     ok_length=sentences <= max_sentences,
                     ok_format=not _FORMAT.search(reply) and not _EMOJI.search(reply))
            r["ok"] = r["ok_content"] and r["ok_length"] and r["ok_format"]
            rows.append(r)
            flag = "ok " if r["ok"] else "BAD"
            print(f"{flag} {r['first_ms']:>5} ms  {text}\n      -> {reply}")
    firsts = [r["first_ms"] for r in rows if r["first_ms"]]
    summary = {"turns": len(rows), "ok": sum(r["ok"] for r in rows),
               "content_ok": sum(r["ok_content"] for r in rows), "length_ok": sum(r["ok_length"] for r in rows),
               "format_ok": sum(r["ok_format"] for r in rows),
               "first_text_ms_p50": statistics.median(firsts) if firsts else None,
               "first_text_ms_max": max(firsts) if firsts else None}
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
