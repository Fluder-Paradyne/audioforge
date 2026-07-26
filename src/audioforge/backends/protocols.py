"""Protocol interfaces for pluggable pipeline backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from audioforge.models import BuildOptions


class FictionReaperRunner(Protocol):
    """Download fiction via FictionReaper (or a test double)."""

    def run(self, url: str, output_dir: Path, *, bin_path: str) -> Path:
        """Download fiction to *output_dir*; return dir of chapter ``.md`` files."""
        ...


class TextPrepBackend(Protocol):
    """Clean chapter text for spoken narration."""

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        """Return speech-ready text derived from *text*."""
        ...


class TtsBackend(Protocol):
    """Synthesize speech audio from prepared text."""

    def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path:
        """Write audio for *text* to *out_path* and return that path."""
        ...


class FfmpegRunner(Protocol):
    """Run an FFmpeg command (or a test double)."""

    def run(self, args: list[str]) -> None:
        """Execute FFmpeg with *args* (excluding the binary name).

        Implementations prepend their configured ``ffmpeg`` path. Raise on
        non-zero exit or missing binary.
        """
        ...
