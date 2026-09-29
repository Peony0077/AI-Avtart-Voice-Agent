"""One conversation session = one browser connection.

Owns the state machine, the turn registry, conversation memory and the outbox.
Phase 1: typed input → streamed LLM reply. Voice (Phase 2+) plugs in through
``on_binary`` (microphone) and the speech pipeline created per turn.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import WebSocket
from loguru import logger

from app.behavior.manager import for_state, from_meta
from app.behavior.meta import MetaStreamParser
from app.behavior.speech_text import clean_delta
from app.core.logging import write_metrics
from app.llm.client import LLMStats
from app.llm.prompt import ConversationMemory
from app.services import Services
from app.session.outbox import Outbox
from app.session.state_machine import Event, State, StateMachine
from app.session.turn import TurnContext, TurnRegistry


class Session:
    def __init__(self, ws: WebSocket, services: Services) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.ws = ws
        self.svc = services
        self.cfg = services.settings
        self.sm = StateMachine()
        self.turns = TurnRegistry()
        self.memory = ConversationMemory(persona=self.cfg.persona_name, max_turns=self.cfg.llm.history_turns,
                                         knowledge_enabled=self.cfg.rag.enabled)
        self.outbox = Outbox(ws, self.turns.is_current)
        self.voice_out = False  # set from the client's hello
        self.sm.subscribe(self._on_state_change)
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.outbox.start()
        self.outbox.json({"type": "session", "session_id": self.id, "persona": self.cfg.persona_name,
                          "profile": self.cfg.profile}, force=True)
        self.sm.fire(Event.START)

    async def close(self, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        if self.turns.current:
            self.turns.current.cancel()
        self.sm.try_fire(Event.STOP)
        await self.outbox.stop()
        logger.info(f"session {self.id} closed ({reason})")

    # ------------------------------------------------------------------ inbound
    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.voice_out = bool(msg.get("voice_out", False))
        elif kind == "user_text":
            text = str(msg.get("text", "")).strip()
            if text:
                await self.start_turn("text", text)
        elif kind == "interrupt":
            await self.interrupt(msg.get("reason", "manual"))
        elif kind == "ping":
            self.outbox.json({"type": "pong", "t": msg.get("t"), "server_time": time.time()}, force=True)
        else:
            logger.debug(f"unknown message type {kind!r}")

    async def on_binary(self, data: bytes) -> None:  # microphone audio (Phase 2)
        return None

    # ------------------------------------------------------------------ turns
    async def start_turn(self, source: str, text: str, t0: float | None = None) -> TurnContext:
        if self.sm.state in (State.THINKING, State.SPEAKING):
            await self.interrupt("new_input")
        turn = self.turns.new_turn(source, text)
        turn.mark("t0", t0)
        if source == "text":
            self.sm.fire(Event.TEXT_INPUT)
        else:
            self.sm.fire(Event.TRANSCRIPT_READY)
        self.outbox.json({"type": "user_message", "turn_id": turn.turn_id, "text": text, "source": source},
                         turn.turn_id, force=True)
        task = asyncio.create_task(self._run_turn(turn), name=f"turn-{turn.turn_id}")
        turn.tasks.append(task)
        return turn

    async def _run_turn(self, turn: TurnContext) -> None:
        try:
            await self._respond(turn)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"turn {turn.turn_id} failed")
            self.outbox.json({"type": "error", "turn_id": turn.turn_id, "message": str(exc)}, force=True)
            if self.turns.is_current(turn.turn_id):
                self.sm.fire(Event.FAILURE)
                self.sm.fire(Event.RECOVER)

    async def _respond(self, turn: TurnContext) -> None:
        tid = turn.turn_id
        knowledge = None  # Phase 5 hook
        messages = self.memory.build(turn.user_text, knowledge)
        parser, spoken, stats = MetaStreamParser(), [], LLMStats()
        behavior_sent = False
        turn.mark("llm_request")
        async for delta in self.svc.llm.stream_chat(messages, stats):
            if turn.is_cancelled:
                return
            turn.mark("llm_first_token")
            meta, text = parser.feed(delta)
            text = clean_delta(text)
            if not behavior_sent and (parser.meta is not None or text):
                self._send_behavior(turn, from_meta(parser.meta))
                behavior_sent = True
            if text:
                self._emit_text(turn, text, spoken)
        tail = clean_delta(parser.flush())
        if tail:
            self._emit_text(turn, tail, spoken)
        turn.mark("llm_done")
        reply = "".join(spoken).strip()
        header = parser.raw_header or ""
        turn.header = header
        if not self.turns.is_current(tid):
            return
        self.memory.add(turn.user_text, header + reply)
        if self.sm.state is State.THINKING:  # nothing was produced
            self.sm.fire(Event.RESPONSE_DONE)
        elif self.sm.state is State.SPEAKING and not self.voice_out:
            self.sm.fire(Event.RESPONSE_DONE)
        self.outbox.json({"type": "assistant_done", "turn_id": tid, "text": reply, "interrupted": False},
                         tid, force=True)
        self._emit_metrics(turn, stats, interrupted=False)

    def _emit_text(self, turn: TurnContext, text: str, spoken: list[str]) -> None:
        spoken.append(text)
        turn.spoken_text = "".join(spoken)
        turn.mark("llm_first_text")
        self.outbox.json({"type": "llm_delta", "turn_id": turn.turn_id, "text": text}, turn.turn_id)
        if not self.voice_out and self.sm.state is State.THINKING:
            turn.mark("first_output")
            self.sm.fire(Event.FIRST_OUTPUT)

    # ------------------------------------------------------------------ interruption
    async def interrupt(self, reason: str) -> None:
        turn = self.turns.current
        if turn is None or self.sm.state not in (State.THINKING, State.SPEAKING):
            return
        t_start = time.perf_counter()
        self.sm.fire(Event.BARGE_IN)
        turn.cancel()
        self.outbox.json({"type": "interrupt", "turn_id": turn.turn_id, "reason": reason}, force=True)
        heard = self._heard_text(turn)
        self.memory.add(turn.user_text, (turn.header or "") + (heard + " [interrupted]" if heard else "[interrupted before replying]"))
        self.sm.fire(Event.CANCEL_DONE)
        turn.mark("interrupted")
        self.outbox.json({"type": "assistant_done", "turn_id": turn.turn_id, "text": heard, "interrupted": True},
                         force=True)
        logger.info(f"turn {turn.turn_id} interrupted ({reason}) in {(time.perf_counter() - t_start) * 1000:.1f} ms")
        self._emit_metrics(turn, None, interrupted=True)

    def _heard_text(self, turn: TurnContext) -> str:
        """Text the user actually received. Text mode: everything displayed so far."""
        return turn.spoken_text.strip()

    # ------------------------------------------------------------------ events
    def _on_state_change(self, old: State, new: State, event: Event) -> None:
        self.outbox.json({"type": "state", "state": new.value, "prev": old.value, "event": event.value,
                          "turn_id": self.turns.current_id}, force=True)
        if new in (State.LISTENING, State.THINKING):
            self._send_behavior(None, for_state(new.value))

    def _send_behavior(self, turn: TurnContext | None, behavior) -> None:
        self.outbox.json({"type": "behavior", "turn_id": turn.turn_id if turn else 0, **behavior.to_dict()},
                         force=turn is None)

    def _emit_metrics(self, turn: TurnContext, stats: LLMStats | None, interrupted: bool) -> None:
        t0 = turn.timings.get("t0", turn.started_at)
        rel = {k: round((v - t0) * 1000, 1) for k, v in turn.timings.items()}
        record = {"type": "metrics", "session": self.id, "turn_id": turn.turn_id, "source": turn.source,
                  "interrupted": interrupted, "ms_from_t0": rel, "llm": stats.__dict__ if stats else None,
                  "ts": time.time()}
        self.outbox.json(record, force=True)
        write_metrics(record)
