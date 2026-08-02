"""Tests for default backend factory."""

from __future__ import annotations

import pytest

from audioforge.backends.alignment import ProportionalAlignmentBackend
from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import SubprocessFfmpegRunner
from audioforge.backends.fictionreaper import SubprocessFictionReaperRunner
from audioforge.backends.kokoro_tts import KokoroNotInstalledError
from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.factory import create_default_backends
from audioforge.models import BuildOptions
from audioforge.settings import AppSettings


def test_create_default_backends_rules_when_skip_prep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_ALLOW_FAKE_TTS", "1")
    settings = AppSettings()
    options = BuildOptions(source="/books", skip_prep=True)
    backends = create_default_backends(settings, options, ollama_available=True)
    assert isinstance(backends.prep, RulesTextPrep)
    assert isinstance(backends.tts, FakeTtsBackend)
    assert isinstance(backends.ffmpeg, SubprocessFfmpegRunner)
    assert isinstance(backends.fictionreaper, SubprocessFictionReaperRunner)
    assert isinstance(backends.aligner, ProportionalAlignmentBackend)


def test_create_default_backends_no_aligner_when_skip_align(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_ALLOW_FAKE_TTS", "1")
    settings = AppSettings()
    options = BuildOptions(source="/books", skip_prep=True, skip_align=True)
    backends = create_default_backends(settings, options)
    assert backends.aligner is None


def test_create_default_backends_reraises_without_fake_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUDIOFORGE_ALLOW_FAKE_TTS", raising=False)

    def boom() -> None:
        raise KokoroNotInstalledError("not installed")

    # Force Kokoro path to fail as if package missing
    monkeypatch.setattr(
        "audioforge.factory.KokoroTtsBackend",
        lambda *a, **k: (_ for _ in ()).throw(KokoroNotInstalledError("missing")),
    )
    settings = AppSettings()
    options = BuildOptions(source="/books", skip_prep=True)
    with pytest.raises(KokoroNotInstalledError, match="missing"):
        create_default_backends(settings, options)


def test_create_default_backends_fake_tts_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_ALLOW_FAKE_TTS", "1")
    monkeypatch.setattr(
        "audioforge.factory.KokoroTtsBackend",
        lambda *a, **k: (_ for _ in ()).throw(KokoroNotInstalledError("missing")),
    )
    settings = AppSettings(ffmpeg_path="/custom/ffmpeg")
    options = BuildOptions(source="/books", skip_prep=True)
    backends = create_default_backends(settings, options, ollama_available=False)
    assert isinstance(backends.tts, FakeTtsBackend)
    assert backends.ffmpeg.ffmpeg_path == "/custom/ffmpeg"  # type: ignore[attr-defined]
