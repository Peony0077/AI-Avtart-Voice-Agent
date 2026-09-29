"""RAG evaluation over the real WebSocket protocol (text turns).

For every question in tests/fixtures/rag_questions.json checks: the answer contains the
expected facts, the right source is cited, unanswerable questions are declined (no
invented facts), and general questions still get answered. Reports retrieval latency.

Usage: .venv\\Scripts\\python scripts/bench_rag.py [--json docs/benchmarks/phase5_rag.json]
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

ROOT = Path(__file__).resolve().parents[1]
DECLINE = re.compile(r"don't have|do not have|does not have|doesn't have|not (?:in|mentioned|included|available|listed|specified)"
                     r"|no information|couldn't find|can't find|isn't in|\bnot\b.{0,40}\blisted\b", re.I)


async def ask(ws, text: str) -> dict:
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "user_text", "text": text}))
    turn = reply = None
    citations, metrics = [], None
    while metrics is None:
        msg = json.loads(await asyncio.wait_for(ws.recv(), 60))
        kind = msg["type"]
        if kind == "user_message":
            turn = msg["turn_id"]
        elif kind == "citations" and msg["turn_id"] == turn:
            citations = msg["sources"]
        elif kind == "assistant_done" and msg["turn_id"] == turn:
            reply = msg["text"]
        elif kind == "metrics" and msg["turn_id"] == turn:
            metrics = msg
    return {"reply": reply, "citations": citations, "rag": metrics.get("rag"),
            "first_text_ms": metrics["ms_from_t0"].get("llm_first_text"), "total_ms": round((time.perf_counter() - t0) * 1000)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    questions = json.loads((ROOT / "tests/fixtures/rag_questions.json").read_text())
    rows = []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "voice_out": False}))
        for q in questions:
            r = await ask(ws, q["q"])
            sources = [c["source"] for c in r["citations"]]
            if q.get("unanswerable"):
                ok = bool(DECLINE.search(r["reply"] or "")) and not re.search(r"\d{3,}|password is", r["reply"] or "", re.I)
            elif q.get("general"):
                ok = all(e.lower() in (r["reply"] or "").lower() for e in q["expect"])
            else:
                ok = q["source"] in sources and all(e.lower() in (r["reply"] or "").lower() for e in q["expect"])
            rows.append({"question": q["q"], "ok": ok, "expected_source": q.get("source"), "cited": sources,
                         "top_score": (r["rag"] or {}).get("top_score"), "embed_ms": (r["rag"] or {}).get("embed_ms"),
                         "search_ms": (r["rag"] or {}).get("search_ms"), "first_text_ms": r["first_text_ms"],
                         "reply": r["reply"]})
            print(("PASS " if ok else "FAIL ") + json.dumps(rows[-1])[:330], flush=True)
    embed = [r["embed_ms"] for r in rows if r["embed_ms"] is not None]
    search = [r["search_ms"] for r in rows if r["search_ms"] is not None]
    summary = {"questions": len(rows), "passed": sum(r["ok"] for r in rows),
               "embed_ms_p50": statistics.median(embed), "search_ms_p50": statistics.median(search),
               "retrieval_ms_max": max(e + s for e, s in zip(embed, search, strict=False))}
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
