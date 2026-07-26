"""Pluggable backends for FictionReaper, text prep, and TTS."""

from __future__ import annotations

from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.fictionreaper import (
    FakeFictionReaperRunner,
    FictionReaperError,
    SubprocessFictionReaperRunner,
)
from audioforge.backends.kokoro_tts import KokoroNotInstalledError, KokoroTtsBackend
from audioforge.backends.ollama_prep import OllamaPrepError, OllamaTextPrep
from audioforge.backends.protocols import (
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.backends.rules_prep import RulesTextPrep

__all__ = [
    "FakeFictionReaperRunner",
    "FakeTtsBackend",
    "FictionReaperError",
    "FictionReaperRunner",
    "KokoroNotInstalledError",
    "KokoroTtsBackend",
    "OllamaPrepError",
    "OllamaTextPrep",
    "RulesTextPrep",
    "SubprocessFictionReaperRunner",
    "TextPrepBackend",
    "TtsBackend",
]
