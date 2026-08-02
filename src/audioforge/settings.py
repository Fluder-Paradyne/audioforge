"""Application settings loaded from environment (``AUDIOFORGE_*``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        """Reject typos like ``DEGUG``; store canonical uppercase names."""
        name = value.strip().upper()
        if name not in logging.getLevelNamesMapping():
            raise ValueError(
                f"Invalid log level {value!r}; "
                "use DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )
        return name
