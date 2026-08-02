"""Alignment backends: timed cues from chapter audio + prepared text.

Default production backend uses phrase-level proportional timing from WAV
duration (no extra deps). Tests inject :class:`FakeAlignmentBackend` (same
algorithm, explicit name). A true forced-alignment engine can implement
:class:`~audioforge.backends.protocols.AlignmentBackend` later.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

from audioforge.models import BuildOptions, ChapterAlignment, TimedCue

# Split on sentence end or blank-line paragraph breaks.
_PHRASE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")


class AlignmentError(Exception):
    """Raised when alignment cannot produce usable cues."""


def wav_duration_seconds(path: Path) -> float:
    """Return duration in seconds of a PCM WAV (stdlib ``wave``)."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise AlignmentError(f"Invalid WAV sample rate in {path}: {rate}")
        return frames / float(rate)


def split_phrases(text: str) -> list[str]:
    """Split *text* into non-empty phrase/sentence chunks."""
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = _PHRASE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def proportional_cues(text: str, duration_s: float) -> list[TimedCue]:
    """Build phrase cues spanning *duration_s* proportional to character length."""
    phrases = split_phrases(text)
    duration = max(0.05, float(duration_s))
    if not phrases:
        return [TimedCue(start_s=0.0, end_s=duration, text="…")]

    weights = [max(1, len(p)) for p in phrases]
    total_w = float(sum(weights))
    cues: list[TimedCue] = []
    cursor = 0.0
    for i, (phrase, weight) in enumerate(zip(phrases, weights, strict=True)):
        if i == len(phrases) - 1:
            end = duration
        else:
            end = min(duration, cursor + duration * (weight / total_w))
            end = max(cursor + 0.05, end)
        text_line = " ".join(phrase.split()) or "…"
        cues.append(TimedCue(start_s=cursor, end_s=end, text=text_line))
        cursor = end
    return cues


class ProportionalAlignmentBackend:
    """Phrase cues timed by character weight across the WAV duration."""

    def align(
        self,
        audio_path: Path,
        text: str,
        *,
        options: BuildOptions,
    ) -> list[TimedCue]:
        del options
        path = Path(audio_path)
        if not path.is_file():
            raise AlignmentError(f"Audio missing for alignment: {path}")
        duration = wav_duration_seconds(path)
        return proportional_cues(text, duration)


class FakeAlignmentBackend(ProportionalAlignmentBackend):
    """Test double alias for :class:`ProportionalAlignmentBackend`."""


def alignment_filename(chapter_index: int, slug: str) -> str:
    """Return ``NNNN-slug.json`` for a chapter alignment artifact."""
    return f"{chapter_index:04d}-{slug}.json"


def save_chapter_alignment(path: Path, alignment: ChapterAlignment) -> None:
    """Write *alignment* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        alignment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_chapter_alignment(path: Path) -> ChapterAlignment:
    """Load a chapter alignment JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ChapterAlignment.model_validate(data)
