"""Pluggable backends for FictionReaper, text prep, TTS, and FFmpeg."""

from __future__ import annotations

from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import (
    FakeFfmpegRunner,
    FfmpegError,
    SubprocessFfmpegRunner,
)
from audioforge.backends.fictionreaper import (
    FakeFictionReaperRunner,
    FictionReaperError,
    SubprocessFictionReaperRunner,
)
from audioforge.backends.kokoro_tts import KokoroNotInstalledError, KokoroTtsBackend
from audioforge.backends.ollama_prep import OllamaPrepError, OllamaTextPrep
from audioforge.backends.protocols import (
    FfmpegRunner,
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.backends.rules_prep import RulesTextPrep

__all__ = [
    "FakeFfmpegRunner",
    "FakeFictionReaperRunner",
    "FakeTtsBackend",
    "FfmpegError",
    "FfmpegRunner",
    "FictionReaperError",
    "FictionReaperRunner",
    "KokoroNotInstalledError",
    "KokoroTtsBackend",
    "OllamaPrepError",
    "OllamaTextPrep",
    "RulesTextPrep",
    "SubprocessFfmpegRunner",
    "SubprocessFictionReaperRunner",
    "TextPrepBackend",
    "TtsBackend",
]
