"""Voice pipeline benchmark over the real WebSocket protocol (no browser needed).

Streams fixture WAVs as a real-time microphone (20 ms packets), plays nothing but
simulates the browser player (reports playback started / progress / ended on the
audio timeline), and collects the server's per-turn metrics.

Modes:
  latency   one turn per bench_q*.wav → distribution of end-of-speech → first audio
  bargein   long answer, then an interruption spoken while the assistant talks →
            detection latency, stale-packet check, heard-text, follow-up turn

Usage:
  .venv\\Scripts\\python scripts/bench_voice.py latency --rounds 2 --json docs/benchmarks/phase2_voice.json
  .venv\\Scripts\\python scripts/bench_voice.py bargein --rounds 3 --json docs/benchmarks/phase3_bargein.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "e2e" / "fixtures"
MANIFEST = json.loads((FIX / "manifest.json").read_text())
HEADER = struct.Struct("<B3xIII")
FRAME = 320  # 20 ms at 16 kHz


class Client:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.msgs: list[dict] = []
        self.audio: list[tuple[float, int, int, int]] = []  # (t, turn, pts, n_samples)
        self.turn_audio_ms: dict[int, float] = {}
        self.tts_done: set[int] = set()
        self.play_start: dict[int, float] = {}
        self.ended: set[int] = set()
        self.interrupted: dict[int, float] = {}

    async def reader(self) -> None:
        async for raw in self.ws:
            now = time.time()
            if isinstance(raw, bytes):
                kind, turn, _seq, pts = HEADER.unpack_from(raw)
                if kind == 1:
                    n = (len(raw) - HEADER.size) // 2
                    self.audio.append((now, turn, pts, n))
                    if turn in self.interrupted:
                        continue  # a stale packet reached the client (counted by the check)
                    self.turn_audio_ms[turn] = self.turn_audio_ms.get(turn, 0) + n / 24.0
                    if turn not in self.play_start:
                        self.play_start[turn] = now
                        await self.send({"type": "playback", "turn_id": turn, "event": "started", "played_ms": 0})
                continue
            msg = json.loads(raw)
            msg["_t"] = now
            self.msgs.append(msg)
            if msg["type"] == "tts_done":
                self.tts_done.add(msg["turn_id"])
            elif msg["type"] == "interrupt":
                self.interrupted[msg["turn_id"]] = now

    async def player(self) -> None:
        """Simulated playback clock: progress every 100 ms, ended after all audio played."""
        while True:
            await asyncio.sleep(0.1)
            now = time.time()
            for turn, t0 in list(self.play_start.items()):
                if turn in self.ended or turn in self.interrupted:
                    continue
                played = min((now - t0) * 1000, self.turn_audio_ms.get(turn, 0))
                if turn in self.tts_done and played >= self.turn_audio_ms.get(turn, 0):
                    self.ended.add(turn)
                    await self.send({"type": "playback", "turn_id": turn, "event": "ended", "played_ms": played})
                else:
                    await self.send({"type": "playback", "turn_id": turn, "event": "progress", "played_ms": played})

    def played_ms(self, turn: int) -> float:
        t0 = self.play_start.get(turn)
        return 0.0 if t0 is None else min((time.time() - t0) * 1000, self.turn_audio_ms.get(turn, 0))

    async def send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj))

    async def stream(self, pcm: np.ndarray, stop_when=None) -> None:
        start = time.perf_counter()
        for i in range(0, len(pcm), FRAME):
            await self.ws.send(pcm[i:i + FRAME].tobytes())
            if stop_when and stop_when():
                return
            delay = start + (i + FRAME) / 16000 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    async def silence_until(self, cond, timeout: float) -> bool:
        zero = np.zeros(FRAME, np.int16).tobytes()
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            await self.ws.send(zero)
            await asyncio.sleep(0.02)
        return cond()

    def find(self, pred):
        return next((m for m in self.msgs if pred(m)), None)


def wav(name: str) -> np.ndarray:
    data, sr = sf.read(FIX / f"{name}.wav", dtype="int16")
    assert sr == 16000
    return data


async def connect(url: str) -> tuple[Client, list[asyncio.Task]]:
    ws = await websockets.connect(url, max_size=None)
    c = Client(ws)
    tasks = [asyncio.create_task(c.reader()), asyncio.create_task(c.player())]
    await c.send({"type": "hello", "voice_out": True})
    await c.send({"type": "voice", "enabled": True})
    await c.silence_until(lambda: c.find(lambda m: m["type"] == "voice"), 10)
    return c, tasks


async def run_latency(url: str, rounds: int) -> dict:
    c, tasks = await connect(url)
    rows = []
    names = sorted(n for n in MANIFEST if n.startswith("bench_q"))
    for r in range(rounds):
        for name in names:
            before = len(c.msgs)
            await c.stream(wav(name))
            got = await c.silence_until(lambda: any(m["type"] == "metrics" for m in c.msgs[before:]), 45)
            if not got:
                rows.append({"fixture": name, "error": "timeout"})
                continue
            m = next(m for m in c.msgs[before:] if m["type"] == "metrics")
            user = next((x for x in c.msgs[before:] if x["type"] == "user_message"), {})
            done = next((x for x in c.msgs[before:] if x["type"] == "assistant_done"), {})
            rows.append({"fixture": name, "round": r, "expected": MANIFEST[name]["speech"][0]["text"],
                         "transcript": user.get("text"), "reply": done.get("text"), "ms": m["ms_from_t0"],
                         "stt_speculative": m.get("stt_speculative"), "speech_ms": m.get("speech_ms")})
            await c.silence_until(lambda: False, 0.5)
    for t in tasks:
        t.cancel()
    await c.ws.close()
    ok = [r for r in rows if "ms" in r and "client_first_audio" in r["ms"]]

    def dist(key):
        vals = [r["ms"][key] for r in ok if key in r["ms"]]
        return {"p50": round(statistics.median(vals)), "p90": round(np.percentile(vals, 90)), "max": round(max(vals))} if vals else None

    summary = {"turns": len(rows), "ok": len(ok),
               "end_of_speech_to_first_audio": dist("client_first_audio"),
               "vad_endpoint": dist("vad_endpoint"), "stt_done": dist("stt_done"),
               "llm_first_token": dist("llm_first_token"), "llm_first_text": dist("llm_first_text"),
               "first_clause": dist("first_clause"), "tts_first_audio": dist("tts_first_audio"),
               "stt_speculative_rate": round(sum(1 for r in ok if r["stt_speculative"]) / max(1, len(ok)), 2),
               "transcripts_exact": sum(1 for r in ok if (r["transcript"] or "").lower().strip(" .?!") ==
                                        r["expected"].lower().strip(" .?!"))}
    return {"summary": summary, "rows": rows}


async def run_bargein(url: str, rounds: int, interrupt_after_s: float) -> dict:
    rows = []
    for r in range(rounds):
        c, tasks = await connect(url)
        long_q, stop_q = wav("bench_long"), wav("bench_interrupt")
        before = len(c.msgs)
        await c.stream(long_q)
        # wait until the assistant has been audible for interrupt_after_s
        await c.silence_until(lambda: any(c.played_ms(t) >= interrupt_after_s * 1000 for t in c.play_start), 40)
        first_turn = max(c.play_start) if c.play_start else None
        onset_offset = MANIFEST["bench_interrupt"]["speech"][0]["start_s"]
        t_stream = time.time()
        speech_onset = t_stream + onset_offset
        played_at_onset = None

        async def watch_onset():
            nonlocal played_at_onset
            await asyncio.sleep(max(0, speech_onset - time.time()))
            played_at_onset = c.played_ms(first_turn)

        watcher = asyncio.create_task(watch_onset())
        await c.stream(stop_q)
        await watcher
        await c.silence_until(lambda: any(m["type"] == "metrics" and m["turn_id"] != first_turn and not m.get("interrupted")
                                          for m in c.msgs[before:]), 45)
        intr = c.find(lambda m: m["type"] == "interrupt" and m["turn_id"] == first_turn)
        done_intr = c.find(lambda m: m["type"] == "assistant_done" and m["turn_id"] == first_turn)
        stale = [a for a in c.audio if a[1] == first_turn and intr and a[0] > intr["_t"]]
        follow_user = c.find(lambda m: m["type"] == "user_message" and m["turn_id"] != first_turn and m.get("source") == "voice"
                             and m["_t"] > (intr["_t"] if intr else 0))
        follow_done = c.find(lambda m: m["type"] == "assistant_done" and follow_user and m["turn_id"] == follow_user["turn_id"])
        m_intr = c.find(lambda m: m["type"] == "metrics" and m["turn_id"] == first_turn)
        rows.append({
            "round": r,
            "interrupt_received": intr is not None,
            "detect_ms_after_speech_onset": round((intr["_t"] - speech_onset) * 1000) if intr else None,
            "server_cancel_ms": m_intr.get("cancel_ms") if m_intr else None,
            "stale_audio_packets_after_interrupt": len(stale),
            "assistant_played_ms_at_onset": round(played_at_onset or 0),
            "heard_text_recorded": done_intr.get("text") if done_intr else None,
            "follow_up_transcript": follow_user.get("text") if follow_user else None,
            "follow_up_reply": follow_done.get("text") if follow_done else None,
        })
        for t in tasks:
            t.cancel()
        await c.ws.close()
        await asyncio.sleep(1.0)
    detects = [r["detect_ms_after_speech_onset"] for r in rows if r["detect_ms_after_speech_onset"] is not None]
    summary = {"rounds": rounds, "interrupted": sum(r["interrupt_received"] for r in rows),
               "detect_ms_p50": statistics.median(detects) if detects else None,
               "detect_ms_max": max(detects) if detects else None,
               "stale_audio_packets_total": sum(r["stale_audio_packets_after_interrupt"] for r in rows),
               "follow_up_ok": sum(1 for r in rows if r["follow_up_reply"])}
    return {"summary": summary, "rows": rows}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["latency", "bargein"])
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--interrupt-after", type=float, default=2.5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.mode == "latency":
        out = await run_latency(args.url, args.rounds)
    else:
        out = await run_bargein(args.url, args.rounds, args.interrupt_after)
    for row in out["rows"]:
        print(json.dumps(row)[:400])
    print(json.dumps(out["summary"], indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
