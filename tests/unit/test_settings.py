"""Tests for AppSettings (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.settings import AppSettings


def test_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.work_dir == Path("work")
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ffmpeg_path == "ffmpeg"
    assert settings.default_voice == "af_heart"
    assert settings.default_prep_model == "llama3.2:3b"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.log_level == "INFO"
    assert settings.log_format == "text"


def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", "/var/audioforge/work")
    monkeypatch.setenv("AUDIOFORGE_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("AUDIOFORGE_FFMPEG_PATH", "/opt/bin/ffmpeg")
    monkeypatch.setenv("AUDIOFORGE_DEFAULT_VOICE", "bf_emma")
    monkeypatch.setenv("AUDIOFORGE_DEFAULT_PREP_MODEL", "mistral:7b")
    monkeypatch.setenv("AUDIOFORGE_HOST", "0.0.0.0")
    monkeypatch.setenv("AUDIOFORGE_PORT", "9000")
    monkeypatch.setenv("AUDIOFORGE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AUDIOFORGE_LOG_FORMAT", "json")

    settings = AppSettings()
    assert settings.work_dir == Path("/var/audioforge/work")
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ffmpeg_path == "/opt/bin/ffmpeg"
    assert settings.default_voice == "bf_emma"
    assert settings.default_prep_model == "mistral:7b"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"


def test_settings_ignores_unprefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORK_DIR", "/should-not-apply")
    monkeypatch.setenv("PORT", "1111")
    settings = AppSettings()
    assert settings.work_dir == Path("work")
    assert settings.port == 8765
