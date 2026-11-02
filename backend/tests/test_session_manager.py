"""Who gets the one conversation (app/api/ws.py): reconnect grace period, promotion races, the line's
limit and what a page sends while it waits."""
import asyncio
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketState

import app.api.ws as ws_mod
from app.api.ws import SessionManager, keep_for_later


class FakeWS:
    def __init__(self):
        self.sent, self.closed = [], None
        self.client_state = self.application_state = WebSocketState.CONNECTED

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code=1000, reason=""):
        self.closed = code
        self.client_state = self.application_state = WebSocketState.DISCONNECTED


class FakeSession:
    def __init__(self, role="customer", visitor=None, close_gate: asyncio.Event | None = None):
        self.id = f"{role}:{visitor}"
        self.role, self.visitor_id = role, visitor
        self.ws, self.svc = FakeWS(), SimpleNamespace(avatar=None)
        self.started = self.ready = self.closed = False
        self.pending, self.handled = {}, []
        self.last_activity = time.time()
        self._gate = close_gate

    async def start(self):
        self.started = True

    async def close(self, reason=""):
        if self._gate is not None:
            await self._gate.wait()
        self.closed = True

    async def on_json(self, msg):
        assert self.started and not self.ready           # applied after start, before live messages
        self.handled.append(msg)

    def on_avatar_frame(self, *args):
        pass


@pytest.fixture
def grace(monkeypatch):
    monkeypatch.setattr(ws_mod, "GRACE_S", 0.05)
    monkeypatch.setattr(ws_mod, "RELOAD_GRACE_S", 0.05)


async def test_a_dropped_customer_gets_the_conversation_back_on_reconnect(grace):
    m = SessionManager()
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    assert await m.attach(a) is True and await m.attach(b) is False
    await m.detach(a, code=1006)                              # tunnel hiccup
    assert a.closed and m.active is None and not b.started    # the next in line waits a moment
    a2 = FakeSession(visitor="aaaa-1111")
    assert await m.attach(a2) is True                         # ... the reconnect continues
    assert m.active is a2 and a2.started and a2.ready and m.waiting == [b]
    await asyncio.sleep(0.1)
    assert m.active is a2 and not b.started                   # the grace timer was cancelled


async def test_without_a_reconnect_the_next_one_starts_after_the_grace_period(grace):
    m = SessionManager()
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    await m.attach(a)
    await m.attach(b)
    await m.detach(a, code=1001)
    await asyncio.sleep(0.15)
    assert m.active is b and b.started and b.ready


async def test_a_normal_close_or_a_handover_moves_the_line_on_at_once(grace):
    m = SessionManager()
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    await m.detach(a, code=1000)                              # the page closed itself
    assert m.active is b and b.started
    m._handed_over.add(b)                                     # inactive with people waiting (4002)
    await m.detach(b, code=4002)
    assert m.active is c and c.started


async def test_a_customer_who_drops_out_of_the_line_keeps_their_place(grace):
    m = SessionManager()
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    await m.detach(b, code=1006)
    assert m.waiting == [c]
    b2 = FakeSession(visitor="bbbb-2222")
    assert await m.attach(b2) is False and m.waiting == [b2, c]    # first again, not behind c


async def test_owner_and_customer_pages_of_one_browser_do_not_replace_each_other():
    m = SessionManager()
    owner, customer = FakeSession("owner", "same-browser-1"), FakeSession("customer", "same-browser-1")
    assert await m.attach(owner) is True
    assert await m.attach(customer) is False                  # waits instead of cutting off the owner
    assert m.active is owner and not owner.closed


async def test_the_line_is_limited(monkeypatch):
    monkeypatch.setattr(ws_mod, "MAX_WAITING", 2)
    m = SessionManager()
    await m.attach(FakeSession(visitor="aaaa-0000"))
    assert await m.attach(FakeSession(visitor="aaaa-1111")) is False
    assert await m.attach(FakeSession(visitor="aaaa-2222")) is False
    assert await m.attach(FakeSession(visitor="aaaa-3333")) is None      # refused (1013)


async def test_what_the_page_sent_while_waiting_is_applied_after_start_in_order():
    m = SessionManager()
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    await m.attach(a)
    await m.attach(b)
    for text in ('{"type":"greet"}', '{"type":"hello","video":["h264"]}', '{"type":"visible","visible":false}',
                 '{"type":"user_text","text":"hi"}', '{"type":"visible","visible":true}', "not json"):
        keep_for_later(b, {"type": "websocket.receive", "text": text})
    keep_for_later(b, {"type": "websocket.receive", "bytes": b"\x00\x00"})          # microphone: dropped
    await m.detach(a, code=1000)
    assert [x["type"] for x in b.handled] == ["hello", "visible", "greet"]       # hello first, the latest visible
    assert b.handled[1]["visible"] is True and b.ready and not b.pending


async def test_a_promoted_page_that_left_meanwhile_is_not_started(grace):
    """The one before closes slowly; meanwhile the promoted page leaves and the line moves on: the first
    detach must not start the page that left (it used to)."""
    m = SessionManager()
    gate = asyncio.Event()
    a = FakeSession(visitor="aaaa-1111", close_gate=gate)
    b, c = FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    first = asyncio.create_task(m.detach(a, code=1000))       # a's close() is still draining ...
    await asyncio.sleep(0.01)
    assert m.active is b and not b.started and not b.ready    # active, but its messages wait (ready)
    await b.ws.close()                                        # ... b leaves
    await m.detach(b, code=1006)
    assert m.active is c and c.started                        # (b never started: no place kept)
    gate.set()
    await first
    assert not b.started and m.active is c


async def test_a_page_whose_socket_closed_is_skipped():
    m = SessionManager()
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    b.ws.client_state = WebSocketState.DISCONNECTED           # its own loop has not detached it yet
    await m.detach(a, code=1000)
    assert m.active is c and c.started and not b.started


async def test_shutdown_ends_the_conversation_and_the_line():
    m = SessionManager()
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    await m.attach(a)
    await m.attach(b)
    await m.shutdown()
    assert a.closed and b.ws.closed == 1012 and not b.started and m.active is None and not m.waiting
    assert m._watch is None or m._watch.done() or m._watch.cancelling()


async def test_a_closed_tab_is_held_shorter_than_a_lost_connection(monkeypatch):
    """1001 (tab closed or reloaded - a reload is back in ~1-2 s) is held RELOAD_GRACE_S, a lost connection GRACE_S."""
    from app.api import ws as ws_mod

    monkeypatch.setattr(ws_mod, "GRACE_S", 0.5)
    monkeypatch.setattr(ws_mod, "RELOAD_GRACE_S", 0.05)
    m = SessionManager()
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    await m.attach(a)
    await m.attach(b)
    await m.detach(a, code=1001)
    await asyncio.sleep(0.15)
    assert m.active is b and b.started


# --- several conversations at once (settings.conversations, 3 Nov 2026) -------------------------------

class FakeLink:
    def __init__(self, slot):
        self.slot, self.frame_sink, self.on_hello, self.connected = slot, None, None, True


def multi(capacity=2):
    m = SessionManager()
    links = [FakeLink(i) for i in range(capacity)]
    m.configure(capacity, links)
    return m, links


async def test_two_customers_talk_at_once_and_the_third_waits(grace):
    m, links = multi(2)
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    assert await m.attach(a) is True and await m.attach(b) is True
    assert await m.attach(c) is False and m.waiting == [c]
    assert m.sessions == [a, b] and a.started and b.started and not c.started
    # each conversation has its own avatar stream, feeding its own page
    assert a.avatar_link is links[0] and b.avatar_link is links[1]
    assert links[0].frame_sink == a.on_avatar_frame and links[1].frame_sink == b.on_avatar_frame
    assert '"position": 1' in c.ws.sent[-1]


async def test_whoever_leaves_frees_their_stream_for_the_next_in_line(grace):
    m, links = multi(2)
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    await m.detach(b, code=1000)                          # b closed the page
    assert m.sessions == [a, c] and c.started and c.ready
    assert c.avatar_link is links[1] and links[1].frame_sink == c.on_avatar_frame
    assert links[0].frame_sink == a.on_avatar_frame       # a is not disturbed


async def test_a_reload_gets_its_own_stream_back_and_a_lost_connection_keeps_its_slot(grace):
    m, links = multi(2)
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b):
        await m.attach(s)
    b2 = FakeSession(visitor="bbbb-2222")                 # b reloads its page
    assert await m.attach(b2) is True and b.closed and b.ws.closed == 4001
    assert b2.avatar_link is links[1] and m.sessions == [a, b2]
    await m.detach(a, code=1006)                          # a's connection drops: its slot is kept
    assert await m.attach(c) is False                     # nobody takes a's slot during the grace period
    a2 = FakeSession(visitor="aaaa-1111")
    assert await m.attach(a2) is True and a2.avatar_link is links[0]
    assert m.waiting == [c]


async def test_a_held_slot_goes_to_the_line_when_nobody_comes_back(grace):
    m, links = multi(2)
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b, c):
        await m.attach(s)
    await m.detach(a, code=1006)
    assert not c.started
    await asyncio.sleep(0.15)                              # the grace period ends
    assert c.started and c.avatar_link is links[0] and m.sessions == [b, c]


async def test_when_full_the_longest_inactive_conversation_is_handed_over(monkeypatch):
    monkeypatch.setattr(ws_mod, "IDLE_HANDOVER_S", 0.0)
    m, _ = multi(2)
    a, b, c = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222"), FakeSession(visitor="cccc-3333")
    for s in (a, b):
        await m.attach(s)
    a.sm = b.sm = SimpleNamespace(state=None)
    a.last_activity, b.last_activity = time.time() - 50, time.time() - 10
    real_sleep = asyncio.sleep
    monkeypatch.setattr(ws_mod.asyncio, "sleep", lambda s: real_sleep(0.01))
    await m.attach(c)
    await real_sleep(0.1)
    assert a.ws.closed == 4002 and b.ws.closed is None      # a was idle longest
    assert a in m._handed_over


async def test_busy_when_any_conversation_is_busy():
    m, _ = multi(2)
    a, b = FakeSession(visitor="aaaa-1111"), FakeSession(visitor="bbbb-2222")
    for s in (a, b):
        await m.attach(s)
    a.sm, b.sm = SimpleNamespace(state=None), SimpleNamespace(state=ws_mod.State.THINKING)
    a.last_activity = b.last_activity = time.time() - 10
    assert m.busy()
    b.sm = SimpleNamespace(state=None)
    assert not m.busy()


async def test_customers_arriving_10_s_and_15_s_apart_all_start_at_once_and_a_fourth_waits(grace):
    """The owner's case (3 Nov 2026): A starts, B 10 s later, C 15 s later - all three talk at the same time,
    each with their own avatar stream; A and B are not disturbed by the newcomers; D waits for a free place."""
    m, links = multi(3)
    a, b, c, d = (FakeSession(visitor=f"{x * 4}-1111") for x in "abcd")
    assert await m.attach(a) is True
    a.last_activity = time.time() - 10                     # A has been talking for 10 s
    assert await m.attach(b) is True and a.started and not a.closed
    assert await m.attach(c) is True and not a.closed and not b.closed
    assert m.sessions == [a, b, c] and [s.avatar_link for s in (a, b, c)] == links
    assert all(link.frame_sink == s.on_avatar_frame for link, s in zip(links, (a, b, c), strict=True))
    assert await m.attach(d) is False and m.waiting == [d] and not d.started
    await m.detach(b, code=1000)                           # B leaves: D gets B's stream, A and C go on
    assert d.started and d.avatar_link is links[1] and m.sessions == [a, c, d]
