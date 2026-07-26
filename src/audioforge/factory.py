"""Default backend construction for CLI / API entry points.

Unit tests should inject fakes into
:func:`~audioforge.pipeline.orchestrator.run_pipeline` directly. This factory
is for production wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import SubprocessFfmpegRunner
from audioforge.backends.fictionreaper import SubprocessFictionReaperRunner
from audioforge.backends.kokoro_tts import KokoroNotInstalledError, KokoroTtsBackend
from audioforge.backends.protocols import (
    FfmpegRunner,
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.models import BuildOptions
from audioforge.pipeline.prep import select_prep_backend
from audioforge.settings import AppSettings

# Env flag: allow silent FakeTtsBackend when Kokoro is not installed.
_ALLOW_FAKE_TTS_ENV = "AUDIOFORGE_ALLOW_FAKE_TTS"


@dataclass(frozen=True, slots=True)
class DefaultBackends:
    """Concrete backends selected for a build."""

    prep: TextPrepBackend
    tts: TtsBackend
    ffmpeg: FfmpegRunner
    fictionreaper: FictionReaperRunner


def create_default_backends(
    settings: AppSettings,
    options: BuildOptions,
    *,
    ollama_available: bool = False,
) -> DefaultBackends:
    """Select prep/TTS/FFmpeg/FictionReaper backends from settings and options.

    TTS selection:

    * Prefer :class:`~audioforge.backends.kokoro_tts.KokoroTtsBackend`.
    * On :class:`~audioforge.backends.kokoro_tts.KokoroNotInstalledError`, use
      :class:`~audioforge.backends.fake.FakeTtsBackend` only when the environment
      variable ``AUDIOFORGE_ALLOW_FAKE_TTS=1``; otherwise re-raise.
    """
    prep = select_prep_backend(
        settings,
        options,
        ollama_available=ollama_available,
    )
    tts = _select_tts_backend()
    ffmpeg: FfmpegRunner = SubprocessFfmpegRunner(ffmpeg_path=settings.ffmpeg_path)
    fictionreaper: FictionReaperRunner = SubprocessFictionReaperRunner()
    return DefaultBackends(
        prep=prep,
        tts=tts,
        ffmpeg=ffmpeg,
        fictionreaper=fictionreaper,
    )


def _select_tts_backend() -> TtsBackend:
    try:
        return KokoroTtsBackend()
    except KokoroNotInstalledError:
        if os.environ.get(_ALLOW_FAKE_TTS_ENV) == "1":
            return FakeTtsBackend()
        raise
