"""Runtime configuration.

Precedence, lowest to highest: dataclass defaults -> environment / .env -> CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5-5",
    "openrouter": "anthropic/claude-sonnet-5.5",  # OpenRouter ids are "<vendor>/<model>"
}


def openrouter_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None


def anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def openrouter_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or None


def cohere_api_key() -> str | None:
    # The Cohere SDK itself reads CO_API_KEY; accept both spellings.
    return os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY") or None


PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY", openai_api_key),
    "anthropic": ("ANTHROPIC_API_KEY", anthropic_api_key),
    "openrouter": ("OPENROUTER_API_KEY", openrouter_api_key),
}


def _env_str(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM ---------------------------------------------------------------
    provider: str | None = None  # "openai" | "anthropic" | "openrouter" | None = auto-detect from API keys
    model: str | None = None  # None = DEFAULT_MODELS[provider]
    fallback_models: list[str] = field(default_factory=list)  # OpenRouter only; empty = auto for :free models
    max_answer_tokens: int = 600
    history_turns: int = 6  # previous exchanges sent back to the LLM as memory
    transcribe_only: bool = False  # live captions, no LLM

    # --- Audio -------------------------------------------------------------
    device: str | None = None  # name substring or index; None = auto-detect loopback device
    audio_file: Path | None = None  # replay a recording instead of a live device
    highpass_hz: float = 100.0  # removes hum / rumble below the speech band
    noise_gate_dbfs: float = -55.0  # frames quieter than this never reach the VAD

    # --- Voice activity detection ------------------------------------------
    vad_threshold: float = 0.5
    min_silence_ms: int = 900  # silence that ends an utterance in AUTO mode
    min_speech_ms: int = 400  # shorter blips (coughs, clicks) are discarded
    speech_pad_ms: int = 300  # audio kept from just before speech started
    max_segment_s: float = 45.0  # force-cut very long monologues

    # --- Trigger ------------------------------------------------------------
    mode: str = "auto"  # "auto" | "manual"
    min_words: int = 3  # AUTO mode ignores shorter utterances ("okay", "mm-hmm")

    # --- Speech-to-text -----------------------------------------------------
    whisper_model: str = "small.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 5
    hotwords: str | None = None  # domain terms that help Whisper spell names correctly

    # --- RAG ----------------------------------------------------------------
    context_paths: list[Path] = field(default_factory=list)
    embedding_provider: str | None = None  # "openai" | "openrouter" | "cohere" | "none" | None = auto
    openai_embedding_model: str = "text-embedding-3-small"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    cohere_embedding_model: str = "embed-v4.0"
    rerank: bool = True
    rerank_model: str = "rerank-v3.5"
    top_k: int = 4
    retrieval_candidates: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 150
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = field(default_factory=lambda: Path.home() / ".cache" / "ai-meeting-copilot" / "qdrant")

    # --- Session ------------------------------------------------------------
    export_dir: Path = Path("sessions")
    export: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            provider=_env_str("COPILOT_PROVIDER", d.provider),
            model=_env_str("COPILOT_MODEL", d.model),
            fallback_models=_env_list("COPILOT_FALLBACK_MODELS"),
            max_answer_tokens=_env_int("COPILOT_MAX_ANSWER_TOKENS", d.max_answer_tokens),
            history_turns=_env_int("COPILOT_HISTORY_TURNS", d.history_turns),
            device=_env_str("COPILOT_DEVICE", d.device),
            highpass_hz=_env_float("COPILOT_HIGHPASS_HZ", d.highpass_hz),
            noise_gate_dbfs=_env_float("COPILOT_NOISE_GATE_DBFS", d.noise_gate_dbfs),
            vad_threshold=_env_float("COPILOT_VAD_THRESHOLD", d.vad_threshold),
            min_silence_ms=_env_int("COPILOT_MIN_SILENCE_MS", d.min_silence_ms),
            min_speech_ms=_env_int("COPILOT_MIN_SPEECH_MS", d.min_speech_ms),
            mode=_env_str("COPILOT_MODE", d.mode) or d.mode,
            min_words=_env_int("COPILOT_MIN_WORDS", d.min_words),
            whisper_model=_env_str("COPILOT_WHISPER_MODEL", d.whisper_model) or d.whisper_model,
            whisper_device=_env_str("COPILOT_WHISPER_DEVICE", d.whisper_device) or d.whisper_device,
            whisper_compute_type=_env_str("COPILOT_WHISPER_COMPUTE_TYPE", d.whisper_compute_type)
            or d.whisper_compute_type,
            language=_env_str("COPILOT_LANGUAGE", d.language),
            beam_size=_env_int("COPILOT_BEAM_SIZE", d.beam_size),
            hotwords=_env_str("COPILOT_HOTWORDS", d.hotwords),
            embedding_provider=_env_str("COPILOT_EMBEDDINGS", d.embedding_provider),
            rerank=_env_bool("COPILOT_RERANK", d.rerank),
            top_k=_env_int("COPILOT_TOP_K", d.top_k),
            qdrant_url=_env_str("QDRANT_URL", d.qdrant_url),
            qdrant_api_key=_env_str("QDRANT_API_KEY", d.qdrant_api_key),
            export_dir=Path(_env_str("COPILOT_EXPORT_DIR", str(d.export_dir))),
        )

    def resolve_provider(self) -> str | None:
        """The LLM provider to use, or None if no API key is available."""
        if self.provider:
            return self.provider
        if openai_api_key():
            return "openai"
        if anthropic_api_key():
            return "anthropic"
        if openrouter_api_key():
            return "openrouter"
        return None

    def resolve_model(self, provider: str) -> str:
        return self.model or DEFAULT_MODELS[provider]

    def resolve_embedding_provider(self) -> str | None:
        choice = (self.embedding_provider or "").lower()
        if choice == "none":
            return None
        if choice in {"openai", "openrouter", "cohere"}:
            return choice
        if openai_api_key():
            return "openai"
        if openrouter_api_key():
            return "openrouter"
        if cohere_api_key():
            return "cohere"
        return None
