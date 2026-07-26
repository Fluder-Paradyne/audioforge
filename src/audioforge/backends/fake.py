"""Lightweight fake backends for tests and offline pipelines."""

from __future__ import annotations

import wave
from pathlib import Path

# Mono 16-bit PCM; sample rate common for speech TTS.
_DEFAULT_SAMPLE_RATE = 24_000
# Frames of silence per character of input text (short, length-scaled).
_FRAMES_PER_CHAR = 48
_MIN_FRAMES = 240  # 10 ms at 24 kHz
_MAX_FRAMES = 48_000  # 2 s cap so tests stay small


class FakeTtsBackend:
    """Write a minimal valid mono 16-bit PCM WAV file (silence).

    Duration scales with ``len(text)`` so callers can distinguish outputs by
    size if needed. Uses only the stdlib :mod:`wave` module.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        frames_per_char: int = _FRAMES_PER_CHAR,
    ) -> None:
        self._sample_rate = sample_rate
        self._frames_per_char = frames_per_char

    def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path:
        """Write silent WAV for *text* to *out_path* and return that path.

        Parent directories are created. *voice* is accepted for protocol
        compatibility and ignored.
        """
        del voice  # protocol-compatible; unused by the fake
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_frames = len(text) * self._frames_per_char
        n_frames = max(_MIN_FRAMES, min(n_frames, _MAX_FRAMES))
        # 16-bit mono silence: two zero bytes per sample
        pcm = b"\x00\x00" * n_frames

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm)

        return path
