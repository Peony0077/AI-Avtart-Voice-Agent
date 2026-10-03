from app.behavior.manager import for_state, from_meta
from app.behavior.meta import MetaStreamParser
from app.llm.prompt import ConversationMemory
from app.transport.protocol import KIND_AUDIO, pack, unpack


def _run(parser, pieces):
    metas, speech = [], []
    for piece in pieces:
        meta, text = parser.feed(piece)
        if meta:
            metas.append(meta)
        speech.append(text)
    speech.append(parser.flush())
    return metas, "".join(speech)


def test_compact_header_split_across_many_deltas():
    raw = "[happy|nod|0.5]Hello there! How are you?"
    metas, speech = _run(MetaStreamParser(), [raw[i:i + 2] for i in range(0, len(raw), 2)])
    assert metas == [{"emotion": "happy", "gesture": "nod", "intensity": 0.5}]
    assert speech == "Hello there! How are you?"


def test_bracket_that_is_not_a_header_is_spoken():
    metas, speech = _run(MetaStreamParser(), ["[laughs] That's funny."])
    assert metas == [] and speech == "[laughs] That's funny."


def test_legacy_json_header_split_across_many_deltas():
    raw = '<meta>{"emotion":"happy","gesture":"nod","intensity":0.5}</meta>Hello there! How are you?'
    metas, speech = _run(MetaStreamParser(), [raw[i:i + 3] for i in range(0, len(raw), 3)])
    assert metas == [{"emotion": "happy", "gesture": "nod", "intensity": 0.5}]
    assert speech == "Hello there! How are you?"


def test_missing_header_is_all_speech():
    metas, speech = _run(MetaStreamParser(), ["Hi", " there."])
    assert metas == [] and speech == "Hi there."


def test_malformed_header_json_still_strips_tags():
    metas, speech = _run(MetaStreamParser(), ["<meta>{oops}</meta>", " Okay."])
    assert metas == [] and speech == "Okay."


def test_unterminated_header_does_not_block_forever():
    parser = MetaStreamParser()
    _, text = parser.feed("<meta>" + "x" * 500)
    assert "x" in text


def test_speech_text_cleaning():
    from app.behavior.speech_text import clean_delta
    assert clean_delta("Have a great day! 😊") == "Have a great day! "
    assert clean_delta("*The Midnight Library* by Matt") == "The Midnight Library by Matt"
    assert clean_delta("rain—or sun") == "rain, or sun"


def test_behavior_validation_and_clamping():
    b = from_meta({"emotion": "EMPATHETIC", "gesture": "dance", "intensity": 7})
    assert (b.emotion, b.gesture, b.intensity, b.source) == ("empathetic", "none", 1.0, "llm")
    assert from_meta(None).emotion == "neutral"
    assert for_state("THINKING").gaze == "away_up"


def test_memory_window_and_knowledge_placement():
    mem = ConversationMemory(persona="Ava", max_turns=4)
    for i in range(5):
        mem.add(f"u{i}", f"a{i}")
    # 10 messages > 8 → trimmed in one block down to the newest 4 messages
    assert [t["content"] for t in mem.turns] == ["u3", "a3", "u4", "a4"]
    assert len(mem.archived) == 6
    msgs = mem.build("question", knowledge="[1] fact")
    assert msgs[0]["role"] == "system" and "Ava" in msgs[0]["content"]
    assert msgs[-1]["content"].startswith("question") and "KNOWLEDGE" in msgs[-1]["content"]


def test_binary_header_roundtrip():
    header, payload = unpack(pack(KIND_AUDIO, 7, 3, 1234, b"abc"))
    assert (header.kind, header.turn_id, header.seq, header.pts_ms, payload) == (KIND_AUDIO, 7, 3, 1234, b"abc")


def test_tags_repeated_mid_reply_are_never_spoken():
    # found in testing: longer replies repeated the tag before each sentence; it was displayed and read aloud
    text = ("[friendly|small_nod|0.4]Octopuses have three hearts.  \n[friendly|headtilt|0.5]They change color. "
            "[neutral|none|0.3]They are smart, see [1] and [the docs].")
    for step in (1, 2, 5, 13, len(text)):
        p = MetaStreamParser()
        out = "".join(p.feed(text[i:i + step])[1] for i in range(0, len(text), step)) + p.flush()
        assert out == "Octopuses have three hearts.  \nThey change color. They are smart, see [1] and [the docs]."
        assert p.meta["gesture"] == "small_nod" and p.later_tags == 2


def test_unclosed_bracket_at_the_end_is_still_spoken():
    p = MetaStreamParser()
    out = p.feed("[friendly|none|0.4]The ratio is [3")[1] + p.flush()
    assert out == "The ratio is [3"
