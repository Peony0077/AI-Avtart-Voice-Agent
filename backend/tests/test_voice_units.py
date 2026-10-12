import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.config import VADCfg
from app.tts.chunker import ClauseChunker
from app.tts.speech_output import Segment, SpeechOutput

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "e2e" / "fixtures"


def _stream(chunker, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    return out + chunker.flush()


def test_first_clause_breaks_early_then_sentences():
    clauses = _stream(ClauseChunker(), "Sure, octopuses have three hearts. Two pump blood to the gills! Neat, right?")
    assert clauses == ["Sure, octopuses have three hearts.", "Two pump blood to the gills!", "Neat, right?"]
    clauses = _stream(ClauseChunker(), "Well, the capital of France, as you may know, is Paris.")
    assert clauses[0] == "Well, the capital of France," and clauses[-1].endswith("Paris.")


def test_a_line_break_ends_a_clause():
    text = "Sure! Here are four facts:\n\nOctopuses have three hearts, and blue blood.\n\n1. They are smart.\n"
    for step in (1, 3, len(text)):                        # streamed in small pieces, or all at once
        assert _stream(ClauseChunker(), text, step) == ["Sure!", "Here are four facts:",
                                                         "Octopuses have three hearts, and blue blood.", "1.",
                                                         "They are smart."]


def test_decimal_numbers_are_not_split():
    assert _stream(ClauseChunker(), "It costs 3.5 dollars today.") == ["It costs 3.5 dollars today."]


def test_long_text_without_punctuation_is_split():
    clauses = _stream(ClauseChunker(max_chars=50), "word " * 40)
    assert len(clauses) > 2 and all(len(c) <= 50 for c in clauses)


def test_heard_text_uses_word_timestamps():
    so = SpeechOutput.__new__(SpeechOutput)  # no event loop / TTS needed for this pure function
    so.segments = [
        Segment(1, "Hello there.", 0, 1000, [("Hello", 0.0, 0.4), ("there.", 0.45, 0.9)]),
        Segment(2, "How are you today?", 1000, 1500, [("How", 0.0, 0.2), ("are", 0.25, 0.4), ("you", 0.45, 0.6),
                                                        ("today?", 0.65, 1.2)]),
    ]
    assert so.heard_text(0) == ""
    assert so.heard_text(500) == "Hello"
    assert so.heard_text(1000) == "Hello there."
    assert so.heard_text(1450) == "Hello there. How are"
    assert so.heard_text(9999) == "Hello there. How are you today?"


@pytest.mark.skipif(not (FIXTURES / "mic_resume.wav").exists(), reason="run scripts/make_fixtures.py")
def test_vad_segments_real_speech_with_pause():
    from app.audio.vad import StreamingVAD

    wav, sr = sf.read(FIXTURES / "mic_resume.wav", dtype="int16")
    manifest = json.loads((FIXTURES / "manifest.json").read_text())["mic_resume"]["speech"]
    vad = StreamingVAD(VADCfg(endpoint_ms=550))
    events, t, all_kinds = [], 0.0, []
    for i in range(0, len(wav), 320):  # 20 ms packets, like the browser
        t += 0.02
        for ev in vad.process(wav[i:i + 320].tobytes(), assistant_active=False, now=t):
            all_kinds.append(ev.kind)
            if ev.kind not in ("pause", "resume"):
                events.append((ev.kind, round(ev.t, 2), round(ev.speech_end_t or 0, 2)))
    kinds = [k for k, *_ in events]
    # The 0.8 s pause is longer than the 0.55 s endpoint → two utterances, each with one early-STT pause
    # and no "resume" (a resume only follows a pause inside the same utterance).
    assert kinds == ["start", "end", "start", "end"], events
    assert all_kinds == ["start", "pause", "end", "start", "pause", "end"], all_kinds
    assert abs(events[1][2] - manifest[0]["end_s"]) < 0.25   # speech end detected close to truth
    assert abs(events[3][2] - manifest[1]["end_s"]) < 0.25


@pytest.mark.skipif(not (FIXTURES / "mic_resume.wav").exists(), reason="run scripts/make_fixtures.py")
def test_vad_resume_after_a_short_pause_inside_one_utterance():
    """A 0.3 s gap (> early-STT 120 ms, < endpoint 550 ms) gives pause → resume, then one end."""
    from app.audio.vad import StreamingVAD

    wav, sr = sf.read(FIXTURES / "mic_resume.wav", dtype="int16")
    first = json.loads((FIXTURES / "manifest.json").read_text())["mic_resume"]["speech"][0]
    a, b = int(first["start_s"] * sr), int(first["end_s"] * sr)
    speech = wav[a:b]
    gap = np.zeros(int(0.3 * sr), np.int16)
    tail = np.zeros(int(1.0 * sr), np.int16)
    audio = np.concatenate([np.zeros(int(0.5 * sr), np.int16), speech, gap, speech, tail])
    vad = StreamingVAD(VADCfg(endpoint_ms=550))
    kinds, t = [], 0.0
    for i in range(0, len(audio), 320):
        t += 0.02
        kinds += [ev.kind for ev in vad.process(audio[i:i + 320].tobytes(), assistant_active=False, now=t)]
    assert kinds == ["start", "pause", "resume", "pause", "end"], kinds


@pytest.mark.skipif(not (FIXTURES / "mic_question.wav").exists(), reason="run scripts/make_fixtures.py")
def test_vad_bargein_threshold_is_stricter_but_still_detects_speech():
    from app.audio.vad import StreamingVAD

    wav, sr = sf.read(FIXTURES / "mic_question.wav", dtype="int16")
    vad = StreamingVAD(VADCfg())
    starts, t = [], 0.0
    for i in range(0, len(wav), 320):
        t += 0.02
        starts += [ev.t for ev in vad.process(wav[i:i + 320].tobytes(), assistant_active=True, now=t) if ev.kind == "start"]
    onset = json.loads((FIXTURES / 'manifest.json').read_text())['mic_question']['speech'][0]['start_s']
    assert len(starts) == 1 and 0.2 < starts[0] - onset < 0.45  # barge-in fires 260 ms (+ chunking) after onset
    quiet = (np.random.default_rng(0).standard_normal(16000 * 2) * 300).astype(np.int16)  # noise ~ -40 dBFS
    vad.reset()
    assert not [e for e in vad.process(quiet.tobytes(), assistant_active=True, now=10.0) if e.kind == "start"]


def test_echo_filter_detects_assistant_words():
    from app.llm.prompt import ConversationMemory
    from app.session.session import Session

    s = Session.__new__(Session)  # only memory is needed
    s.memory = ConversationMemory(persona="Ava")
    s.memory.add("tell me a story", "[happy|smile|0.5]Once upon a time a brave little fox lived by the sea.")
    assert s._looks_like_echo("a brave little fox lived by the sea")
    assert not s._looks_like_echo("wait stop what is two plus two")
    assert not s._looks_like_echo("fox")  # too short to judge
