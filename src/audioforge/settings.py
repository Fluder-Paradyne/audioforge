"""Application settings loaded from environment (``AUDIOFORGE_*``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from audioforge.logging_config import normalize_log_level_name


class AppSettings(BaseSettings):
    """Global defaults for workspaces, backends, and the local API."""

    model_config = SettingsConfigDict(
        env_prefix="AUDIOFORGE_",
        extra="ignore",
    )

    work_dir: Path = Path("work")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ffmpeg_path: str = "ffmpeg"
    default_voice: str = "af_heart"
    default_prep_model: str = "llama3.2:3b"
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Allowlist DEBUG…CRITICAL; reject NOTSET and typos like ``DEGUG``."""
        return normalize_log_level_name(value)
