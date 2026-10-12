"""Explicit conversation state machine.

One instance per session. All state changes go through :meth:`StateMachine.fire`
with an :class:`Event`; the allowed transitions are listed in ``TRANSITIONS``.
There are deliberately no loose boolean flags such as ``is_speaking``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    END_OF_TURN = "END_OF_TURN"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


class Event(StrEnum):
    START = "start"                        # session opened / microphone enabled
    SPEECH_END = "speech_end"              # VAD endpoint: user finished speaking
    TEXT_INPUT = "text_input"              # typed message (skips STT)
    TRANSCRIPT_READY = "transcript_ready"  # STT produced usable text
    TRANSCRIPT_EMPTY = "transcript_empty"  # STT produced nothing usable
    SPEECH_RESUMED = "speech_resumed"      # user kept talking before STT finished
    FIRST_OUTPUT = "first_output"          # first assistant audio (or text) delivered
    RESPONSE_DONE = "response_done"        # assistant finished (playback complete)
    BARGE_IN = "barge_in"                  # user interrupted the assistant
    CANCEL_DONE = "cancel_done"            # all cancellations acknowledged
    FAILURE = "failure"                    # unrecoverable error in the turn
    RECOVER = "recover"                    # error handled, resume listening
    STOP = "stop"                          # session closed / microphone disabled


S, E = State, Event

TRANSITIONS: dict[tuple[State, Event], State] = {
    (S.IDLE, E.START): S.LISTENING,
    (S.IDLE, E.TEXT_INPUT): S.THINKING,
    (S.LISTENING, E.SPEECH_END): S.END_OF_TURN,
    (S.LISTENING, E.TEXT_INPUT): S.THINKING,
    (S.END_OF_TURN, E.TRANSCRIPT_READY): S.THINKING,
    (S.END_OF_TURN, E.TRANSCRIPT_EMPTY): S.LISTENING,
    (S.END_OF_TURN, E.SPEECH_RESUMED): S.LISTENING,
    (S.END_OF_TURN, E.TEXT_INPUT): S.THINKING,   # typed while the spoken words were being transcribed
    (S.THINKING, E.FIRST_OUTPUT): S.SPEAKING,
    (S.THINKING, E.RESPONSE_DONE): S.LISTENING,
    (S.SPEAKING, E.RESPONSE_DONE): S.LISTENING,
    (S.THINKING, E.BARGE_IN): S.INTERRUPTED,
    (S.SPEAKING, E.BARGE_IN): S.INTERRUPTED,
    (S.INTERRUPTED, E.CANCEL_DONE): S.LISTENING,
    (S.ERROR, E.RECOVER): S.LISTENING,
}
# FAILURE and STOP are accepted from every state.
for _state in State:
    TRANSITIONS[(_state, E.FAILURE)] = S.ERROR
    TRANSITIONS[(_state, E.STOP)] = S.IDLE


class InvalidTransition(RuntimeError):
    def __init__(self, state: State, event: Event):
        super().__init__(f"event {event.value!r} is not allowed in state {state.value}")
        self.state = state
        self.event = event


Listener = Callable[[State, State, Event], None]


@dataclass
class StateMachine:
    state: State = State.IDLE
    history: list[tuple[float, State, Event, State]] = field(default_factory=list)
    _listeners: list[Listener] = field(default_factory=list)

    def can(self, event: Event) -> bool:
        return (self.state, event) in TRANSITIONS

    def fire(self, event: Event) -> State:
        try:
            new = TRANSITIONS[(self.state, event)]
        except KeyError:
            raise InvalidTransition(self.state, event) from None
        old, self.state = self.state, new
        self.history.append((time.time(), old, event, new))
        del self.history[:-200]
        for listener in list(self._listeners):
            listener(old, new, event)
        return new

    def try_fire(self, event: Event) -> bool:
        """Fire if allowed; return whether a transition happened."""
        if not self.can(event):
            return False
        self.fire(event)
        return True

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)
