"""Application settings.

Precedence (highest first): environment variables (prefix ``AVATAR_``) â†’ ``.env`` â†’
YAML profile ``config/profiles/<AVATAR_PROFILE>.yaml`` â†’ defaults below.
Nested keys use ``__`` in env vars, e.g. ``AVATAR_LLM__MAX_TOKENS=200``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


class LLMCfg(BaseModel):
    base_url: str = "http://127.0.0.1:8081"
    model_path: str = "llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"  # relative to model_home
    server_exe: str = "tools/llama.cpp/current/llama-server.exe"  # relative to repo
    manage_server: bool = True  # backend starts/stops llama-server itself
    context: int = 4096
    gpu_layers: int = 99
    kv_cache_type: str = "q8_0"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 320
    request_timeout_s: float = 60.0
    history_turns: int = 8


class STTCfg(BaseModel):
    model: str = "openai/whisper-large-v3-turbo"
    device: str = "cuda"
    language: str = "en"
    min_utterance_ms: int = 250


class VADCfg(BaseModel):
    threshold: float = 0.5            # speech probability while listening
    bargein_threshold: float = 0.75   # stricter while the assistant is speaking
    min_speech_ms: int = 200          # speech needed to start an utterance
    bargein_min_ms: int = 260         # speech needed to interrupt the assistant
    endpoint_ms: int = 550            # trailing silence that ends an utterance
    preroll_ms: int = 300             # audio kept before speech start
    min_dbfs: float = -55.0           # ignore chunks quieter than this (noise floor)
    early_stt_ms: int = 120           # start STT speculatively after this much silence (0 = off)
    max_utterance_s: float = 30.0


class TTSCfg(BaseModel):
    engine: str = "kokoro"            # "kokoro" (stock voice) | "clone"
    voice: str = "af_heart"
    speed: float = 1.0
    device: str = "cuda"
    clone_reference_wav: str | None = None   # for engine == "clone"
    clone_reference_text: str | None = None


class RAGCfg(BaseModel):
    enabled: bool = True
    embed_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: str = "data/index"
    table: str = "knowledge"
    top_k: int = 4
    min_score: float = 0.55
    max_context_chars: int = 2400


class AvatarCfg(BaseModel):
    enabled: bool = False
    worker_url: str = "ws://127.0.0.1:8765/ws"
    manage_worker: bool = True
    worker_python: str = "avatar/musetalk/.venv/Scripts/python.exe"
    avatar_id: str = "dev_girl"
    fps: int = 20
    jpeg_quality: int = 80


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AVATAR_", env_nested_delimiter="__",
                                      env_file=REPO_ROOT / ".env", extra="ignore")

    profile: str = "default"
    model_home: Path = Path("E:/AI-models")
    log_dir: Path = REPO_ROOT / "logs"
    persona_name: str = "Ava"
    server: ServerCfg = ServerCfg()
    llm: LLMCfg = LLMCfg()
    stt: STTCfg = STTCfg()
    vad: VADCfg = VADCfg()
    tts: TTSCfg = TTSCfg()
    rag: RAGCfg = RAGCfg()
    avatar: AvatarCfg = AvatarCfg()
    voice_enabled: bool = True
    vram_min_free_gib: float = 0.8

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings, _YamlProfileSource(settings_cls),
                file_secret_settings)

    def model_file(self, rel: str) -> Path:
        return self.model_home / rel

    def repo_path(self, rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else REPO_ROOT / path


class _YamlProfileSource(PydanticBaseSettingsSource):
    def get_field_value(self, field, field_name):  # pragma: no cover - unused
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        name = os.environ.get("AVATAR_PROFILE", "default")
        path = REPO_ROOT / "config" / "profiles" / f"{name}.yaml"
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("profile", name)
        return data


@lru_cache
def get_settings() -> Settings:
    return Settings()
