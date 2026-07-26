"""Application settings loaded from environment (``AUDIOFORGE_*``)."""

from __future__ import annotations

from pathlib import Path

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
