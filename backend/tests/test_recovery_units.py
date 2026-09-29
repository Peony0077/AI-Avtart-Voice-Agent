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
