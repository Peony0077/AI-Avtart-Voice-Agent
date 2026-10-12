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
    ("He lives on Main St. near the park.", "He lives on Main Street near the park."),   # was "Maincent"
    ("We flew to St. Louis.", "We flew to Saint Louis."),
    ("Go to Main St. Then turn left.", "Go to Main Street. Then turn left."),
    ("Turn onto Park Rd.", "Turn onto Park Road."),
    ("Mt. Fuji is tall.", "Mount Fuji is tall."),
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


def test_numbered_lists_are_spoken_without_their_numbers():
    """The chunker cut "2." into a clause of its own, spoken "Two." with pauses (890 ms gap, e2e)."""
    from app.behavior.speech_text import normalize_for_speech as speak
    nl = chr(10)
    assert speak("2.") == "" and speak(" 3) ") == ""
    assert speak("Here are four facts about octopuses:" + nl + nl + "1.") == "Here are four facts about octopuses:"
    assert speak("1. Octopuses have three hearts.") == "Octopuses have three hearts."
    assert speak("It costs 2.5 dollars.") == "It costs 2.5 dollars." and speak("Call 911.") == "Call 911."


def test_long_pauses_inside_a_sentence_are_shortened():
    """Kokoro paused up to 520 ms at a comma; people pause ~200-400 ms (cap_pauses)."""
    import numpy as np

    from app.tts.base import cap_pauses
    sr = 24000
    tone = (0.3 * np.sin(np.arange(sr // 2) / 3)).astype(np.float32)      # 0.5 s of sound
    gap = np.zeros(int(0.6 * sr), np.float32)                              # 600 ms comma pause
    audio = np.concatenate([tone, gap, tone])
    words = [("a", 0.0, 0.5), ("b", 1.1, 1.6)]
    out, moved = cap_pauses(audio, sr, words, max_ms=320)
    assert abs(len(out) / sr - (1.0 + 0.32)) < 0.02                       # the pause is now ~320 ms
    assert moved[0] == ("a", 0.0, 0.5) and abs(moved[1][1] - 0.82) < 0.02  # the next word moved up
    short = np.concatenate([tone, np.zeros(int(0.2 * sr), np.float32), tone])
    assert len(cap_pauses(short, sr, words)[0]) == len(short)              # a short pause stays


def _clean_stream(text, step):
    from app.behavior.speech_text import SpeechTextStream
    s, out = SpeechTextStream(), ""
    for i in range(0, len(text), step):
        out += s.feed(text[i:i + step])
    return out + s.flush()


@pytest.mark.parametrize("text, shown", [
    ("See https://example.com/a_b?x=1. Then go.", "See . Then go."),       # a URL split over tokens
    ("Write to john_doe@gmail.com today.", "Write to john_doe@gmail.com today."),   # not "johndoe@"
    ("It takes ~20 min, _really_ **fast**.", "It takes ~20 min, really fast."),   # "~20" stays for "about 20"
    ("It is normal — a test.", "It is normal, a test."),
    ("rain—or sun", "rain, or sun"),
])
def test_streamed_text_is_cleaned_a_whole_word_at_a_time(text, shown):
    for step in (1, 2, 3, 7, len(text)):                     # however the LLM's tokens split it
        assert _clean_stream(text, step) == shown, step


def test_what_the_cleaning_keeps_is_spoken_right():
    assert normalize_for_speech("It takes ~20 min.") == "It takes about 20 min."
    assert normalize_for_speech("Write to john_doe@gmail.com.") == "Write to john underscore doe@gmail.com."


def test_a_finished_sentence_is_released_before_the_next_word_exists():
    """The LLM pauses after a clause (session.clause_pauses): the speech must have that clause then."""
    from app.behavior.speech_text import SpeechTextStream
    s, c = SpeechTextStream(), ClauseChunker()
    clauses = []
    for delta in ("Sure", "!", " Here", " are", " facts", "."):
        clauses += c.feed(s.feed(delta))
        if delta == " Here":
            assert clauses == ["Sure!"]
    assert clauses + c.feed(s.flush()) + c.flush() == ["Sure!", "Here are facts."]


@pytest.mark.parametrize("text, clauses", [
    ("Plan B. Go now.", ["Plan B.", "Go now."]),
    ("Take vitamin C. It helps.", ["Take vitamin C.", "It helps."]),
    ("Fruits, nuts, etc. Then we left.", ["Fruits, nuts, etc.", "Then we left."]),
    ("Fruits, nuts, etc. and more.", ["Fruits, nuts, etc. and more."]),
    ("George W. Bush spoke. Yes.", ["George W. Bush spoke.", "Yes."]),
])
def test_sentence_ends_after_a_letter_name_and_etc(text, clauses):
    assert _chunks(text) == clauses


@pytest.mark.parametrize("text, spoken", [
    ("So x - 5 - 3 is it.", "So x - 5 - 3 is it."),       # arithmetic, not "5 to 3"
    ("And 5-3=2.", "And 5-3=2."),
    ("Then 7 - 2 is 5.", "Then 7 - 2 is 5."),
    ("It takes 10 - 15 minutes.", "It takes 10 to 15 minutes."),
])
def test_ranges_only_in_clear_contexts(text, spoken):
    assert normalize_for_speech(text) == spoken


def test_an_empty_clause_gets_no_sentence_pause():
    assert pause_after("") == 60 and pause_after("   ") == 60      # ("" is in ".!?")


def test_a_short_reply_starting_with_an_unclosed_bracket_is_not_lost():
    from app.behavior.meta import MetaStreamParser
    p = MetaStreamParser()
    assert p.feed("[Note") == (None, "") and p.feed(": short") == (None, "")
    assert p.flush() == "[Note: short"
    p = MetaStreamParser()
    assert p.feed("[friendly|nod|0.4]Hi [x")[1] == "Hi "
    assert p.flush() == "[x"
