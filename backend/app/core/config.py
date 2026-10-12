"""Application settings.

Precedence (highest first): environment variables (prefix ``AVATAR_``) -> ``.env`` ->
YAML profile ``config/profiles/<AVATAR_PROFILE>.yaml`` -> defaults below.
``AVATAR_PROFILE`` itself may come from the environment or from ``.env``.
Nested keys use ``__`` in env vars, e.g. ``AVATAR_LLM__MAX_TOKENS=200``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
WINDOWS = os.name == "nt"
# executables inside a venv / a llama.cpp build differ between Windows and Linux
VENV_PYTHON = ".venv/Scripts/python.exe" if WINDOWS else ".venv/bin/python"
DEFAULT_MODEL_HOME = Path("E:/AI-models") if WINDOWS else Path.home() / "AI-models"


class LLMCfg(BaseModel):
    base_url: str = "http://127.0.0.1:8081"
    model_path: str = "llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"  # relative to model_home
    server_exe: str = ("tools/llama.cpp/current/llama-server.exe" if WINDOWS
                       else "tools/llama.cpp/current/llama-server")  # relative to repo
    manage_server: bool = True  # backend starts/stops llama-server itself
    context: int = 4096
    gpu_layers: int = 99
    kv_cache_type: str = "q8_0"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 320
    request_timeout_s: float = 60.0
    history_turns: int = 16    # messages kept verbatim after a trim (up to twice that before one); older ones
    #                            are summarised. 8 lost the start of a 10-exchange chat ("what did we talk about first?")
    prefetch_on_pause: bool = True    # start the answer at the VAD pause, adopt it at the endpoint
    behavior_tag: bool = False        # LLM writes [emotion|gesture|intensity] first (+~180 ms); else read from the words
    # With speech the LLM writes just in time (llm/client.py pause/resume): the first clause, then nothing
    # until its audio is out, then one clause at a time whenever less than this much speech is queued.
    speech_ahead_ms: int = 3000


class STTCfg(BaseModel):
    model: str = "openai/whisper-large-v3-turbo"
    device: str = "cuda"
    language: str = "en"
    min_utterance_ms: int = 250
    keep_warm_ms: int = 400           # run the encoder this often while the user speaks (0 = off)
    debug_audio: bool = False         # save every utterance + transcript to logs/stt_debug (local only)


class VADCfg(BaseModel):
    threshold: float = 0.5            # speech probability while listening
    bargein_threshold: float = 0.75   # stricter while the assistant is speaking
    min_speech_ms: int = 200          # speech needed to start an utterance
    bargein_min_ms: int = 260         # speech needed to interrupt the assistant
    endpoint_ms: int = 450            # trailing silence that ends an utterance (550 before the unfinished-question
    #                                   wait and the continuation merge made a shorter one safe)
    continue_ms: int = 1000           # extra wait when the transcript looks unfinished ("I would like to")
    continuation_s: float = 3.0       # user resumes this soon, before hearing much of the reply: same question
    preroll_ms: int = 500             # audio kept before speech start (300 clipped soft first words: "Can you" -> "and you")
    min_dbfs: float = -55.0           # ignore chunks quieter than this (noise floor)
    early_stt_ms: int = 120           # start STT speculatively after this much silence (0 = off)
    max_utterance_s: float = 30.0


class TTSCfg(BaseModel):
    engine: str = "kokoro"            # "kokoro" (stock) | "clone" (XTTS-v2) | "clone_light" (Kokoro + converter)
    voice: str = "af_heart"           # Kokoro voice (for clone_light: the base voice that gets re-timbred)
    speed: float = 1.0
    device: str = "cuda"
    clone_reference_wav: str | None = None   # for engine == "clone"
    clone_tau: float = 0.3            # clone_light: OpenVoice conversion temperature (its default)


class RAGCfg(BaseModel):
    enabled: bool = True
    embed_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: str = "data/index"
    table: str = "knowledge"
    top_k: int = 4
    min_score: float = 0.62          # cosine similarity needed to use a passage (bge-small-en-v1.5)
    weak_score: float = 0.50         # below min but above this: tell the model the docs lack the answer
    max_context_chars: int = 2400


class AvatarCfg(BaseModel):
    enabled: bool = False
    worker_url: str = "ws://127.0.0.1:8765/ws"
    manage_worker: bool = True
    worker_python: str = f"avatar/musetalk/{VENV_PYTHON}"
    avatar_id: str = "dev_girl"
    fps: int = 20
    jpeg_quality: int = 90      # speech frames; idle frames are stored at 90 too (no visible change when she stops talking)
    gate_frames: int = 4        # lip-synced frames rendered before a turn's first audio is released
    decoder: str = "taesd"      # MuseTalk latent decoder: taesd (fast, MODEL_HOME/avatar/models/taesd) | vae (more
                                # faithful; the HD profile uses it). RTX 3060, GPU kept warm: gate 4 -> no late frames
    upscaler: str = "x2plus"    # Real-ESRGAN for the lower face of large (> 512 px) avatars (avatar/musetalk/upscale.py);
                                # "" = bicubic
    h264_encoder: str = "auto"  # auto: NVENC when the GPU and ffmpeg have it, else libx264 (avatar/musetalk/h264.py)
    late_ms: int = 100          # skip speech frames that would reach the browser more than this late
                                # (keeps lips in sync when the GPU is busy; 0 = render every frame)
    warm_burst_ms: int = 0      # MuseTalk warm-up when a turn enters THINKING; redundant since the STT
                                # keep-warm, and it slowed the LLM's first clause (~65 ms p50)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AVATAR_", env_nested_delimiter="__",
                                      env_file=REPO_ROOT / ".env", extra="ignore")

    profile: str = "default"
    model_home: Path = DEFAULT_MODEL_HOME
    log_dir: Path = REPO_ROOT / "logs"
    persona_name: str = "Ava"
    llm: LLMCfg = LLMCfg()
    stt: STTCfg = STTCfg()
    vad: VADCfg = VADCfg()
    tts: TTSCfg = TTSCfg()
    rag: RAGCfg = RAGCfg()
    avatar: AvatarCfg = AvatarCfg()
    voice_enabled: bool = True
    access_token: str | None = None   # required as ?token=... on /ws and /api when set (remote access): the owner's link
    customer_token: str | None = None   # the customer link: talks to her, no settings or leads (api/ws.py)
    vram_min_free_gib: float = 0.8

    @model_validator(mode="after")
    def _tokens(self) -> Settings:
        # Without an owner token everyone is the owner (local use): a customer link next to it would have
        # given every customer - and anyone else - the owner's settings and leads. Refused at start-up.
        if self.customer_token and not self.access_token:
            raise ValueError("customer_token is set but access_token is not: set AVATAR_ACCESS_TOKEN too (the "
                             "owner's link), or remove AVATAR_CUSTOMER_TOKEN for local use without links")
        return self

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
        name = profile_name()
        path = REPO_ROOT / "config" / "profiles" / f"{name}.yaml"
        if not path.exists():
            known = sorted(p.stem for p in path.parent.glob("*.yaml"))
            raise ValueError(f"unknown profile {name!r} (AVATAR_PROFILE); profiles: {', '.join(known)}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("profile", name)
        return data


def profile_name() -> str:
    """AVATAR_PROFILE from the environment, else from .env. (pydantic's .env source fills the settings only,
    not os.environ: with the profile set in .env alone, /health named it but default.yaml was loaded.)"""
    name = os.environ.get("AVATAR_PROFILE")
    env_file = REPO_ROOT / ".env"
    if not name and env_file.exists():
        from dotenv import dotenv_values

        name = dotenv_values(env_file).get("AVATAR_PROFILE")
    return name or "default"


@lru_cache
def get_settings() -> Settings:
    return Settings()
