from __future__ import annotations

from app.core.config import Settings


def create_tts(settings: Settings):
    engine = settings.tts.engine
    if engine == "kokoro":
        from app.tts.kokoro_tts import KokoroTTS

        return KokoroTTS(settings.tts)
    if engine == "clone":
        from app.tts.clone_tts import CloneTTS

        return CloneTTS(settings)
    raise ValueError(f"unknown TTS engine {engine!r}")
