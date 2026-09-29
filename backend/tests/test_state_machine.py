import pytest
from app.session.state_machine import Event, InvalidTransition, State, StateMachine
from app.session.turn import TurnRegistry


def test_voice_turn_happy_path():
    sm = StateMachine()
    for event, expected in [
        (Event.START, State.LISTENING),
        (Event.SPEECH_END, State.END_OF_TURN),
        (Event.TRANSCRIPT_READY, State.THINKING),
        (Event.FIRST_OUTPUT, State.SPEAKING),
        (Event.RESPONSE_DONE, State.LISTENING),
    ]:
        assert sm.fire(event) is expected


def test_barge_in_path():
    sm = StateMachine(state=State.SPEAKING)
    assert sm.fire(Event.BARGE_IN) is State.INTERRUPTED
    assert sm.fire(Event.CANCEL_DONE) is State.LISTENING


def test_empty_transcript_returns_to_listening():
    sm = StateMachine(state=State.END_OF_TURN)
    assert sm.fire(Event.TRANSCRIPT_EMPTY) is State.LISTENING


def test_invalid_transition_raises():
    sm = StateMachine(state=State.LISTENING)
    with pytest.raises(InvalidTransition):
        sm.fire(Event.FIRST_OUTPUT)
    assert sm.state is State.LISTENING
    assert not sm.try_fire(Event.CANCEL_DONE)


@pytest.mark.parametrize("state", list(State))
def test_failure_and_stop_from_any_state(state):
    sm = StateMachine(state=state)
    assert sm.fire(Event.FAILURE) is State.ERROR
    sm = StateMachine(state=state)
    assert sm.fire(Event.STOP) is State.IDLE


def test_listeners_see_every_transition():
    sm = StateMachine()
    seen = []
    sm.subscribe(lambda old, new, ev: seen.append((old, new, ev)))
    sm.fire(Event.START)
    sm.fire(Event.TEXT_INPUT)
    assert seen == [
        (State.IDLE, State.LISTENING, Event.START),
        (State.LISTENING, State.THINKING, Event.TEXT_INPUT),
    ]


async def test_new_turn_cancels_previous_and_rejects_stale_ids():
    reg = TurnRegistry()
    t1 = reg.new_turn("text", "hi")
    assert reg.is_current(t1.turn_id)
    t2 = reg.new_turn("voice")
    assert t1.is_cancelled
    assert not reg.is_current(t1.turn_id)
    assert reg.is_current(t2.turn_id)
    assert t2.turn_id == t1.turn_id + 1
