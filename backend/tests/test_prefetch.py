"""Answer prefetch (session/prefetch.py): started at the VAD pause, adopted only when nothing changed."""
import asyncio

import pytest

from app.session.prefetch import AnswerPrefetch


class _LLM:
    def __init__(self, deltas=("Hi", " there."), fail: bool = False) -> None:
        self.deltas, self.fail = deltas, fail
        self.requests: list[list[dict]] = []
        self.closed = False

    async def stream_chat(self, messages, stats=None):
        self.requests.append(messages)
        try:
            if self.fail:
                raise ConnectionError("llm down")
            for d in self.deltas:
                await asyncio.sleep(0.01)
                yield d
            await asyncio.sleep(10)          # keeps "generating" until cancelled or consumed
        finally:
            self.closed = True


async def _retrieve(text):
    return "KNOWLEDGE", [{"source": "doc.md"}], {"top": 0.7}


def _build(text, knowledge):
    return [{"role": "user", "content": f"{text} | {knowledge}"}]


async def _stt(text="What is two plus two?"):
    await asyncio.sleep(0.02)
    return text, 0.0


def _make(llm, stt_task, prepare=lambda t: t):
    return AnswerPrefetch(12.5, stt_task, prepare, _retrieve, _build, llm)


def test_adopted_when_same_pause_and_text_and_streams_buffered_deltas():
    async def run():
        llm = _LLM(deltas=("Two", " plus two", " is four."))
        pf = _make(llm, asyncio.create_task(_stt()))
        await asyncio.sleep(0.1)                               # the endpoint comes a bit later
        assert await pf.matches(12.5, "What is two plus two?")
        got = []
        async for d in pf.deltas():
            got.append(d)
            if len(got) == 3:
                break
        pf.cancel()
        await asyncio.sleep(0)
        return pf, llm, got

    pf, llm, got = asyncio.run(run())
    assert got == ["Two", " plus two", " is four."]
    assert pf.citations == [{"source": "doc.md"}] and pf.t_first_token is not None
    assert llm.requests[0][0]["content"] == "What is two plus two? | KNOWLEDGE"


def test_not_adopted_after_new_speech_or_other_text():
    async def run():
        pf = _make(_LLM(), asyncio.create_task(_stt()))
        other_pause = await pf.matches(13.0, "What is two plus two?")      # speech continued after the pause
        other_text = await pf.matches(12.5, "What is two plus three?")     # e.g. carried-over text changed
        pf.cancel()
        return other_pause, other_text

    assert asyncio.run(run()) == (False, False)


def test_echo_or_empty_transcript_is_not_prefetched():
    async def run():
        llm = _LLM()
        pf = _make(llm, asyncio.create_task(_stt("")), prepare=lambda t: None)
        ok = await pf.matches(12.5, "")
        return ok, llm.requests

    ok, requests = asyncio.run(run())
    assert ok is False and requests == []


def test_cancel_stops_the_llm_but_not_the_shared_stt_task():
    async def run():
        llm = _LLM()
        stt = asyncio.create_task(_stt())
        pf = _make(llm, stt)
        await asyncio.sleep(0.05)            # STT done, LLM streaming
        pf.cancel()
        await asyncio.sleep(0.02)
        return llm.closed, stt.cancelled(), await stt

    closed, stt_cancelled, stt_result = asyncio.run(run())
    assert closed and not stt_cancelled and stt_result[0] == "What is two plus two?"


def test_cancel_while_waiting_for_stt_leaves_stt_running():
    async def run():
        stt = asyncio.create_task(_stt())
        pf = _make(_LLM(), stt)
        await asyncio.sleep(0)
        pf.cancel()
        return await stt

    assert asyncio.run(run())[0] == "What is two plus two?"


def test_llm_failure_before_the_endpoint_is_not_adopted():
    async def run():
        pf = _make(_LLM(fail=True), asyncio.create_task(_stt()))
        await asyncio.sleep(0.1)
        return await pf.matches(12.5, "What is two plus two?"), pf.failed

    assert asyncio.run(run()) == (False, True)


def test_llm_failure_after_adoption_reaches_the_turn():
    async def run():
        pf = _make(_LLM(fail=True), asyncio.create_task(_stt()))
        with pytest.raises(ConnectionError):
            async for _ in pf.deltas():
                pass

    asyncio.run(run())
