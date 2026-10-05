"""LLM pause after the first clause, resume, continue the same reply (llm/client.py), against a fake server."""
import asyncio
import json

import httpx

from app.core.config import LLMCfg
from app.llm.client import LLMClient, LLMStats
from app.tts.chunker import ClauseChunker


def _sse(chunks: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode()


def _server(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content or b"{}")))
        if request.url.path == "/v1/chat/completions":
            deltas = ["The capital", " is Tokyo.", " It is", " very", " big."]
            return httpx.Response(200, content=_sse([{"choices": [{"delta": {"content": d}}]} for d in deltas]))
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<prompt>"})
        if request.url.path == "/completion":
            return httpx.Response(200, content=_sse([{"content": " very"}, {"content": " big."},
                                                     {"content": "", "stop": True, "stop_type": "eos",
                                                      "timings": {"predicted_n": 3, "prompt_n": 4, "cache_n": 30}}]))
        return httpx.Response(404)
    return handler


def _client(seen: list) -> LLMClient:
    client = LLMClient(LLMCfg())
    client._http = httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(_server(seen)))
    return client


def test_pauses_after_the_first_clause_and_continues_the_same_reply():
    seen, order = [], []

    async def run():
        client = _client(seen)
        resumed = asyncio.Event()

        async def resume():
            order.append("resume")
            resumed.set()

        stats = LLMStats()
        out = []
        clauses = 0

        def pause_after(text):                          # after each new clause, as the session does
            nonlocal clauses
            n = len(ClauseChunker().feed(text))
            if n > clauses:
                clauses = n
                return True
            return False

        async for delta in client.stream_chat([{"role": "user", "content": "q"}], stats,
                                              pause_after=pause_after, resume=resume):
            out.append(delta)
            order.append(delta)
        return "".join(out), stats

    text, stats = asyncio.run(run())
    assert text == "The capital is Tokyo. It is very big."
    # nothing more was generated between the first clause and the resume
    assert order.index("resume") == order.index(" It is") + 1
    # the continuation starts from exactly the text already produced
    completion = next(body for path, body in seen if path == "/completion")
    assert completion["prompt"] == "<prompt>The capital is Tokyo. It is" and completion["cache_prompt"] is True
    assert stats.finish_reason == "eos" and stats.extra["continued"]["cache_n"] == 30


def test_without_pause_it_streams_straight_through():
    seen = []

    async def run():
        client = _client(seen)
        return "".join([d async for d in client.stream_chat([{"role": "user", "content": "q"}])])

    assert asyncio.run(run()) == "The capital is Tokyo. It is very big."
    assert [path for path, _ in seen] == ["/v1/chat/completions"]


def test_a_short_opener_does_not_pause_the_llm():
    from app.session.session import clause_pauses

    pause = clause_pauses()
    assert not pause("Sure!")                               # nothing complete yet (no text after it)
    assert not pause("Sure! Here")                          # "Sure!" alone: write on
    assert not pause("Sure! Here are four facts")
    assert pause("Sure! Here are four facts about octopuses. They")   # the next sentence: pause as usual
    assert not pause("Sure! Here are four facts about octopuses. They have")

    pause = clause_pauses()
    assert pause("The capital of France is Paris. It")     # a full first sentence: pause right away
