"""Phase 6 regressions: LLM restart handling and the streaming-TTS -> avatar path."""
import asyncio
import time

import httpx
import numpy as np
import pytest

from app.core.config import LLMCfg
from app.llm.client import LLMClient
from app.tts.speech_output import SpeechOutput

SSE = (b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
       b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
       b"data: [DONE]\n\n")


def _client(wait_ready, fail_first: int, status: int | None = None) -> tuple[LLMClient, dict]:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            if status is not None:      # e.g. 503 while a restarted llama-server loads the model
                return httpx.Response(status, request=request)
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})

    client = LLMClient(LLMCfg(), wait_ready=wait_ready)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://llm")
    return client, calls


async def _collect(client: LLMClient) -> str:
    return "".join([d async for d in client.stream_chat([{"role": "user", "content": "hi"}])])


def test_llm_request_waits_for_restart_and_retries():
    waited = []

    async def wait_ready(timeout_s: float) -> bool:
        waited.append(timeout_s)
        return True

    client, calls = _client(wait_ready, fail_first=1)
    assert asyncio.run(_collect(client)) == "Hello"
    assert calls["n"] == 2 and len(waited) == 1


def test_llm_request_retries_on_503_while_model_loads_but_not_on_other_errors():
    async def wait_ready(timeout_s: float) -> bool:
        return True

    client, calls = _client(wait_ready, fail_first=1, status=503)
    assert asyncio.run(_collect(client)) == "Hello" and calls["n"] == 2
    client, calls = _client(wait_ready, fail_first=1, status=400)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_collect(client))
    assert calls["n"] == 1


def test_llm_request_fails_without_supervisor():
    client, _calls = _client(None, fail_first=1)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(_collect(client))


class _Turn:
    turn_id = 7
    is_cancelled = False

    def __init__(self) -> None:
        self.tasks, self.marks = [], []

    def mark(self, key: str) -> None:
        self.marks.append(key)


class _Outbox:
    def __init__(self) -> None:
        self.json_msgs, self.packets = [], []

    def json(self, obj, turn_id=None, force=False) -> None:
        self.json_msgs.append(obj)

    def binary(self, data, turn_id=None) -> None:
        self.packets.append(data)


class _StreamingTTS:
    sample_rate = 24000

    def __init__(self, chunks: list[int]) -> None:
        self.chunks = chunks

    def stream(self, text: str):
        for n in self.chunks:
            time.sleep(0.02)
            yield np.full(n, 0.1, np.float32)


def test_streaming_chunks_merge_while_avatar_is_busy_and_keep_all_audio():
    """Clone-voice chunks that queue up while the avatar renders go out as one longer segment, and a
    tiny last chunk (shorter than a video frame) is delivered too: nothing is lost or reordered."""
    chunks = [4800, 4800, 4800, 4800, 300]          # 0.2 s each, then 12.5 ms
    avatar_calls: list[int] = []

    async def slow_avatar(turn_id, seq, pcm, pts_ms) -> None:
        avatar_calls.append(len(pcm))
        await asyncio.sleep(0.12)                    # rendering the first frames takes a while

    async def run() -> tuple[_Outbox, SpeechOutput]:
        outbox = _Outbox()
        so = SpeechOutput(_Turn(), _StreamingTTS(chunks), outbox, asyncio.Lock(), lambda: None, slow_avatar)
        so.say("A clause long enough to stream.")
        so.close()
        await asyncio.wait_for(so.finished.wait(), 5)
        return outbox, so

    outbox, so = asyncio.run(run())
    assert sum(avatar_calls) == sum(chunks)                          # all audio reached the avatar
    assert len(avatar_calls) < len(chunks)                           # queued chunks were merged
    assert sum(len(p) - 16 for p in outbox.packets) == 2 * sum(chunks)   # PCM16 to the browser, same audio
    assert so.segments[0].duration_ms == pytest.approx(sum(chunks) / 24, abs=0.5)
    assert outbox.json_msgs[-1]["type"] == "tts_done"


class _FailingTTS:
    sample_rate = 24000

    def synthesize(self, text: str):
        raise RuntimeError("CUDA out of memory")


def test_a_tts_error_still_finishes_the_speech_output():
    """A failed synthesis used to leave ``finished`` unset: the turn waited for it forever."""
    async def run() -> SpeechOutput:
        so = SpeechOutput(_Turn(), _FailingTTS(), _Outbox(), asyncio.Lock(), lambda: None)
        so.say("Hello there.")
        so.close()
        await asyncio.wait_for(so.finished.wait(), 5)
        return so

    so = asyncio.run(run())
    assert isinstance(so.error, RuntimeError)


def test_a_cancelled_gpu_job_keeps_the_lock_until_its_thread_is_done():
    """Cancelling the caller (a barge-in) must not let the next GPU job overlap the running thread."""
    from app.core.gpu import locked_thread

    async def run() -> list[str]:
        lock, events = asyncio.Lock(), []

        def slow() -> None:
            time.sleep(0.2)
            events.append("first done")

        first = asyncio.create_task(locked_thread(lock, slow))
        await asyncio.sleep(0.05)
        first.cancel()
        await locked_thread(lock, lambda: events.append("second starts"))
        with pytest.raises(asyncio.CancelledError):
            await first
        return events

    assert asyncio.run(run()) == ["first done", "second starts"]


def test_an_external_llm_server_may_have_any_url():
    """The supervisor parsed the port from base_url even when it does not run the server."""
    from app.core.config import Settings
    from app.llm.server import LlamaServerSupervisor, port_of
    settings = Settings(_env_file=None)
    settings.llm.manage_server, settings.llm.base_url = False, "https://llm.example.com/v1"
    assert LlamaServerSupervisor(settings).port is None
    assert port_of("http://127.0.0.1:8081") == 8081 and port_of("https://llm.example.com") == 443


def test_sad_words_are_whole_words():
    from app.behavior.manager import is_happy_text
    assert is_happy_text("That's great! I love painting and my new scarf.")
    assert not is_happy_text("That's great, but I'm scared and crying.")


def test_watchdog_backs_off_and_notices_a_restart_slower_than_its_timeout(monkeypatch):
    import threading

    import app.llm.server as srv
    from app.core.config import Settings
    monkeypatch.setattr(srv, "RESTART_BACKOFF_S", (0.01, 0.08))
    sup = srv.LlamaServerSupervisor(Settings(_env_file=None))
    spawns = []

    class Proc:
        def __init__(self, alive):
            self.alive, self.returncode = alive, None if alive else 1

        def poll(self):
            return None if self.alive else 1

    def spawn():
        spawns.append(time.monotonic())
        sup._proc = Proc(alive=len(spawns) >= 5)          # crashes four times, then loads slowly

    def wait_ready(timeout_s):
        if not sup._proc.alive:
            raise RuntimeError("exited")
        raise TimeoutError("not ready after 180 s")          # (it comes up later)

    sup._proc = Proc(alive=False)
    monkeypatch.setattr(sup, "_spawn", spawn)
    monkeypatch.setattr(sup, "wait_ready", wait_ready)
    monkeypatch.setattr(sup, "_is_healthy", lambda: len(spawns) >= 5)
    watchdog = threading.Thread(target=sup._watch, daemon=True)
    watchdog.start()
    assert sup.ready.wait(5)                                  # ready is set again (it never was)
    sup._stop.set()
    watchdog.join(2)
    gaps = [b - a for a, b in zip(spawns, spawns[1:], strict=False)]
    assert len(spawns) == 5 and gaps[-1] > gaps[0]             # waits longer after each failed restart
