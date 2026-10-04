"""What arrived from the user's microphone (app/audio/mic_stats.py): the session's log line and the
warning for a microphone that sends only silence."""
import numpy as np

from app.audio.mic_stats import MicStats


def frame(amplitude: int, n: int = 320) -> bytes:
    return (np.ones(n, np.int16) * amplitude).tobytes()


def test_summary_says_how_much_audio_arrived_how_loud_and_how_often_speech_was_heard():
    s = MicStats()
    assert s.summary() is None                                  # no microphone in this session
    for _ in range(50):                                          # 1 s
        s.add(frame(3277))                                       # -20 dBFS
    s.speech += 1
    assert s.summary() == "1.0 s received, peak -20 dBFS, speech detected 1x"


def test_a_silent_microphone_is_reported_once_after_ten_seconds():
    s = MicStats()
    warnings = [w for _ in range(50 * 12) if (w := s.add(frame(0)))]
    assert len(warnings) == 1 and "10 s of silence" in warnings[0] and "-inf dBFS" in warnings[0]
    s2 = MicStats()
    assert not [w for _ in range(50 * 12) if (w := s2.add(frame(1000)))]     # -30 dBFS: a working microphone


def test_audio_while_voice_is_off_is_counted_separately():
    s = MicStats()
    s.add(frame(1000), voice_on=False)
    assert s.seconds == 0 and "ignored (voice off)" in s.summary()
