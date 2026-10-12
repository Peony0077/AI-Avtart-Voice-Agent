"""What a session accepts from the page (limits) and how a split utterance is transcribed (session.py)."""
import asyncio
import time
from types import SimpleNamespace

import numpy as np

from app.audio.mic_stats import MicStats
from app.audio.vad import VadEvent
from app.core.config import VADCfg
from app.llm.prompt import ConversationMemory
from app.session.session import MAX_MIC_FRAME_BYTES, MAX_USER_TEXT_CHARS, Session
from app.session.state_machine import Event, State, StateMachine


class Outbox:
    def json(self, *args, **kwargs):
        pass


def bare_session() -> Session:
    """A Session with only what the input paths use (no models, no socket)."""
    s = Session.__new__(Session)
    s.id, s.role, s.last_activity = "test", "owner", time.time()
    s.outbox, s.mic_stats, s.sm = Outbox(), MicStats(), StateMachine()
    s._bad_frames, s._stt_tasks, s._utterance = 0, set(), 0
    s._spec = s._prefetch = s._summary_task = None
    s._carry_text = ""
    s.cfg = SimpleNamespace(vad=VADCfg(continue_ms=0))
    s.memory = ConversationMemory(persona="Ava")
    s._log_utterance = lambda *a: None
    return s


async def test_only_microphone_frames_reach_the_vad():
    s = bare_session()
    seen = []
    s.vad = SimpleNamespace(process=lambda data, assistant_active: seen.append(len(data)) or [])
    await s.on_binary(b"\x00" * 1024)
    await s.on_binary(b"\x00" * 1023)                          # odd length: np.frombuffer raised in the VAD
    await s.on_binary(b"\x00" * (MAX_MIC_FRAME_BYTES + 2))     # not a microphone frame
    assert seen == [1024] and s._bad_frames == 2


async def test_typed_text_is_capped():
    s = bare_session()
    got = []

    async def start_turn(source, text, **kw):
        got.append(text)
    s.start_turn = start_turn
    await s.on_json({"type": "user_text", "text": "x" * 10_000})
    assert len(got[0]) == MAX_USER_TEXT_CHARS


async def test_an_utterance_continued_while_its_first_part_was_transcribed_is_answered_whole():
    """Pause, go on, stop again - while the first part's STT is still running: the older transcription
    used to start a turn with the first half."""
    s = bare_session()
    s.sm.fire(Event.START)
    gate, turns = asyncio.Event(), []
    first, second = np.zeros(1, np.float32), np.ones(1, np.float32)

    async def stt(audio):
        if audio is first:
            await gate.wait()                                   # slow
            return "I would like to book", time.time()
        return "a table for two", time.time()

    async def start_turn(source, text, **kw):
        turns.append(text)
        s.sm.fire(Event.TRANSCRIPT_READY)
        return SimpleNamespace(mark=lambda *a: None, turn_id=1)
    s._stt, s.start_turn = stt, start_turn

    await s._on_speech_end(VadEvent("end", time.time(), audio=first, speech_end_t=1.0))
    s.sm.fire(Event.SPEECH_RESUMED)                             # the user goes on ...
    await s._on_speech_end(VadEvent("end", time.time(), audio=second, speech_end_t=2.0))   # ... and stops
    await asyncio.sleep(0.05)
    assert turns == [] and s.sm.state is State.END_OF_TURN     # the second part waits for the first
    gate.set()
    await asyncio.gather(*list(s._stt_tasks))
    assert turns == ["I would like to book a table for two"]


async def test_typed_text_cancels_every_transcription():
    s = bare_session()
    s.sm.fire(Event.START)
    gate, turns = asyncio.Event(), []

    async def stt(audio):
        await gate.wait()
        return "spoken words", time.time()

    async def start_turn(source, text, **kw):
        turns.append(text)
    s._stt, s.start_turn = stt, start_turn
    await s._on_speech_end(VadEvent("end", time.time(), audio=np.zeros(1, np.float32), speech_end_t=1.0))
    tasks = list(s._stt_tasks)
    s._cancel_transcription()                                   # (start_turn("text") does this)
    gate.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert turns == [] and s._carry_text == ""


def test_the_vad_leaves_torch_threads_alone_and_takes_odd_frames():
    import torch

    from app.audio.vad import StreamingVAD
    threads = torch.get_num_threads()
    vad = StreamingVAD(VADCfg())
    assert torch.get_num_threads() == threads                   # (the RAG embedder runs on these threads)
    assert vad.process(b"\x00" * 1025, assistant_active=False) == []
