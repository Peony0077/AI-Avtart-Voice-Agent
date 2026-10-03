"""Speech output: padding trimmed to natural pauses, text spoken correctly, sentences not split at abbreviations."""
import numpy as np
import pytest

from app.behavior.speech_text import clean_delta, normalize_for_speech
from app.tts.base import TTSResult, pause_after, trim_silence
from app.tts.chunker import ClauseChunker

SR = 24000


def test_trim_replaces_engine_padding_with_a_natural_pause():
    # Kokoro-like clause: 0.3 s silence, 1 s voice, 0.5 s silence (measured padding)
    voice = (np.sin(np.arange(SR) / 10) * 0.5).astype(np.float32)
    audio = np.concatenate([np.zeros(int(0.3 * SR), np.float32), voice, np.zeros(int(0.5 * SR), np.float32)])
    r = TTSResult(audio, SR, words=[("hello", 0.35, 0.8), ("there", 0.85, 1.25)])
    out = trim_silence(r, pause_after("Hello there."))
    lead = np.flatnonzero(np.abs(out.audio) > 0.005)[0] / SR * 1000
    trail = (len(out.audio) - np.flatnonzero(np.abs(out.audio) > 0.005)[-1]) / SR * 1000
    assert lead <= 25                                   # was 300 ms
    assert 250 <= trail <= 320                          # sentence pause, was 500 ms
    assert out.words[0][1] == pytest.approx(0.35 - 0.28, abs=0.01)   # timestamps follow the trimmed audio


@pytest.mark.parametrize("clause, pause", [("Hello there.", 260), ("Really?", 260), ("Well,", 130), ("and so", 60)])
def test_pause_depends_on_punctuation(clause, pause):
    assert pause_after(clause) == pause


@pytest.mark.parametrize("text, spoken", [
    ("Water boils at 100°C.", "Water boils at 100 degrees Celsius."),
    ("It was -5°F.", "It was -5 degrees Fahrenheit."),
    ("It takes ~5 minutes.", "It takes about 5 minutes."),
    ("It is about ~20 minutes.", "It is about 20 minutes."),
    ("Use 2-3 drops.", "Use 2 to 3 drops."),
    ("Open 9:00 to 13:00.", "Open 9 o'clock to 13 hundred."),
    ("Back at 1:05, lunch 12:30.", "Back at 1 oh 5, lunch 12 30."),
    ("Fruit, e.g. apples.", "Fruit, for example apples."),
    ("Pages, i.e. five.", "Pages, that is five."),
    ("Fruits, nuts, etc.", "Fruits, nuts, et cetera."),
    ("It is 11.4 km, 80 km/h.", "It is 11.4 kilometers, 80 kilometers per hour."),
    ("Call 555-1234 on 2024-05-01.", "Call 555-1234 on 2024-05-01."),     # no false ranges
])
def test_spoken_forms(text, spoken):
    assert normalize_for_speech(text) == spoken


def test_en_dash_range_is_not_turned_into_a_comma():
    assert clean_delta("Use 2–3 drops — really.") == "Use 2-3 drops, really."


def _chunks(text):
    c = ClauseChunker()
    return c.feed(text) + c.flush()


@pytest.mark.parametrize("text, clauses", [
    ("Mr. Brown met Dr. Lee at 5 p.m. today. They talked.", ["Mr. Brown met Dr. Lee at 5 p.m. today.", "They talked."]),
    ("Fruit, e.g. apples, is healthy. Yes.", ["Fruit, e.g. apples, is healthy.", "Yes."]),
    ("J. K. Rowling wrote it. Great.", ["J. K. Rowling wrote it.", "Great."]),
    ("So did I. Then we left.", ["So did I.", "Then we left."]),
    ("No. That is not right.", ["No.", "That is not right."]),
])
def test_no_sentence_break_after_abbreviations(text, clauses):
    assert _chunks(text) == clauses
