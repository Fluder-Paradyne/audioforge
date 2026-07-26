"""Pluggable backends for FictionReaper, text prep, and TTS."""

from __future__ import annotations

from audioforge.backends.fictionreaper import (
    FakeFictionReaperRunner,
    FictionReaperError,
    SubprocessFictionReaperRunner,
)
from audioforge.backends.ollama_prep import OllamaPrepError, OllamaTextPrep
from audioforge.backends.protocols import (
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.backends.rules_prep import RulesTextPrep

__all__ = [
    "FakeFictionReaperRunner",
    "FictionReaperError",
    "FictionReaperRunner",
    "OllamaPrepError",
    "OllamaTextPrep",
    "RulesTextPrep",
    "SubprocessFictionReaperRunner",
    "TextPrepBackend",
    "TtsBackend",
]
