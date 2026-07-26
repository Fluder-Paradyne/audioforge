"""Pluggable backends for FictionReaper, text prep, and TTS."""

from __future__ import annotations

from audioforge.backends.fictionreaper import (
    FakeFictionReaperRunner,
    FictionReaperError,
    SubprocessFictionReaperRunner,
)
from audioforge.backends.protocols import (
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)

__all__ = [
    "FakeFictionReaperRunner",
    "FictionReaperError",
    "FictionReaperRunner",
    "SubprocessFictionReaperRunner",
    "TextPrepBackend",
    "TtsBackend",
]
