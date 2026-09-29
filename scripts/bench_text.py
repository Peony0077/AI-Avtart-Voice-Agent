"""Phase 1 benchmark: text turns over the real WebSocket protocol.

Measures per turn (client side): time to first delta, time to done, reply length,
metadata compliance; and exercises cancellation (interrupt mid-reply).

Usage: .venv\\Scripts\\python scripts/bench_text.py [--url ws://127.0.0.1:8000/ws] [--turns 10] [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import websockets

PROMPTS = [
    "Hi! Who are you?",
    "What's a good way to relax after a long day?",
    "Tell me a fun fact about octopuses.",
    "I feel a bit stressed about my exams.",
    "Can you explain what a black hole is, simply?",
    "What should I cook for dinner tonight?",
    "How do I stay motivated when learning to code?",
    "Recommend a book for a rainy weekend.",
    "What's the difference between weather and climate?",
    "Thanks, that was helpful!",
]


async def run_turn(ws, text: str, interrupt_after_ms: float | None = None) -> dict:
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "user_text", "text": text}))
    first = done = None
    reply, behavior, metrics, turn_id, interrupted = "", None, None, None, False
    sent_interrupt = False
    while True:
        timeout = 60
        if interrupt_after_ms is not None and first is not None and not sent_interrupt:
            timeout = max(0.0, interrupt_after_ms / 1000 - (time.perf_counter() - first))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except TimeoutError:
            await ws.send(json.dumps({"type": "interrupt", "reason": "bench"}))
            sent_interrupt = True
            t_int = time.perf_counter()
            continue
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        kind = msg["type"]
        if kind == "user_message":
            turn_id = msg["turn_id"]
        elif kind == "behavior" and msg.get("turn_id") == turn_id:
            behavior = msg
        elif kind == "llm_delta" and msg["turn_id"] == turn_id:
            if first is None:
                first = time.perf_counter()
            reply += msg["text"]
        elif kind == "assistant_done" and msg["turn_id"] == turn_id:
            done = time.perf_counter()
            interrupted = msg["interrupted"]
        elif kind == "metrics" and msg["turn_id"] == turn_id:
            metrics = msg
            break
    out = {
        "prompt": text, "turn_id": turn_id, "reply": reply, "interrupted": interrupted,
        "first_delta_ms": round((first - t0) * 1000, 1) if first else None,
        "done_ms": round((done - t0) * 1000, 1) if done else None,
        "behavior": {k: behavior[k] for k in ("emotion", "gesture", "intensity", "source")} if behavior else None,
        "server": metrics,
    }
    if sent_interrupt:
        out["interrupt_ack_ms"] = round((done - t_int) * 1000, 1)
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "voice_out": False}))
        results = [await run_turn(ws, PROMPTS[i % len(PROMPTS)]) for i in range(args.turns)]
        cancel = await run_turn(ws, "Please describe the history of the Roman Empire in detail.", interrupt_after_ms=300)
        after_cancel = await run_turn(ws, "Okay, just say hello in one sentence.")

    firsts = [r["first_delta_ms"] for r in results if r["first_delta_ms"]]
    summary = {
        "turns": len(results),
        "first_delta_ms_p50": statistics.median(firsts),
        "first_delta_ms_max": max(firsts),
        "done_ms_p50": statistics.median(r["done_ms"] for r in results),
        "meta_compliance": sum(1 for r in results if r["behavior"] and r["behavior"]["source"] == "llm") / len(results),
        "tokens_per_s_p50": statistics.median(r["server"]["llm"]["tokens_per_s"] for r in results if r["server"]["llm"]["tokens_per_s"]),
        "cancel_interrupted": cancel["interrupted"],
        "cancel_ack_ms": cancel.get("interrupt_ack_ms"),
        "after_cancel_ok": bool(after_cancel["reply"]) and not after_cancel["interrupted"],
    }
    for r in results + [cancel, after_cancel]:
        print(f"[{r['turn_id']}] {r['first_delta_ms']} ms | {r['behavior']} | {r['reply'][:90]!r}")
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "turns": results, "cancel": cancel,
                                         "after_cancel": after_cancel}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
