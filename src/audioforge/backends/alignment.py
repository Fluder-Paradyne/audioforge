"""Alignment backends: timed cues from chapter audio + prepared text.

Default production backend:

1. Strip YAML frontmatter / light markdown (same idea as speech cleanup).
2. Detect non-silent speech regions via FFmpeg ``silencedetect``.
3. Distribute phrases across speech regions by character weight.

This is still approximate (not full forced alignment / ASR), but much closer
than spreading text evenly over the whole WAV (including pauses).
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

from audioforge.models import BuildOptions, ChapterAlignment, TimedCue

# Split on sentence end or blank-line paragraph breaks.
_PHRASE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


class AlignmentError(Exception):
    """Raised when alignment cannot produce usable cues."""


def strip_text_for_alignment(text: str) -> str:
    """Remove frontmatter and light markdown so cues are spoken content only."""
    cleaned = _FRONTMATTER_RE.sub("", text)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "")
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(
        r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).split("]")[0][1:], cleaned
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
    phrases = [p.strip() for p in parts if p and p.strip()]
    # Drop ultra-short heading leftovers like "1." alone if longer content exists.
    if len(phrases) > 1:
        phrases = [p for p in phrases if len(p) > 2 or not re.fullmatch(r"\d+\.?", p)]
    return phrases


def proportional_cues(
    text: str,
    duration_s: float,
    *,
    speech_regions: list[tuple[float, float]] | None = None,
) -> list[TimedCue]:
    """Build phrase cues; optionally only within *speech_regions* (start, end).

    Guarantees every cue satisfies ``0 <= start_s < end_s <= duration``.
    The per-cue minimum length is scaled so all phrases fit inside the
    speech timeline (never forces ends past *duration_s*).
    """
    phrases = split_phrases(text)
    duration = max(0.05, float(duration_s))
    if not phrases:
        return [TimedCue(start_s=0.0, end_s=duration, text="…")]

    if speech_regions:
        regions = [
            (max(0.0, s), min(duration, e)) for s, e in speech_regions if e - s > 0.05
        ]
        if not regions:
            regions = [(0.0, duration)]
    else:
        regions = [(0.0, duration)]

    # Flatten speech timeline length.
    region_lens = [e - s for s, e in regions]
    speech_total = sum(region_lens) or duration

    weights = [max(1, len(p)) for p in phrases]
    total_w = float(sum(weights))
    n = len(phrases)
    # Floor small enough that all phrases fit; last cue may use remainder.
    min_len = min(0.05, speech_total / (n + 1))

    # Map each phrase to a span on the concatenated speech timeline, then to absolute.
    cues: list[TimedCue] = []
    speech_cursor = 0.0
    for i, (phrase, weight) in enumerate(zip(phrases, weights, strict=True)):
        if i == n - 1:
            speech_end = speech_total
        else:
            ideal = speech_cursor + speech_total * (weight / total_w)
            remaining_after = n - 1 - i
            max_end = speech_total - min_len * remaining_after
            speech_end = min(max_end, max(speech_cursor + min_len, ideal))
            speech_end = max(speech_cursor, min(speech_end, speech_total))

        abs_start = _speech_time_to_absolute(speech_cursor, regions, region_lens)
        abs_end = _speech_time_to_absolute(speech_end, regions, region_lens)
        abs_start = max(0.0, min(abs_start, duration))
        abs_end = max(abs_start, min(abs_end, duration))
        if abs_end <= abs_start:
            abs_end = min(duration, abs_start + max(min_len, 1e-3))
        if abs_end <= abs_start:
            abs_start = max(0.0, duration - max(min_len, 1e-3))
            abs_end = duration
        text_line = " ".join(phrase.split()) or "…"
        cues.append(TimedCue(start_s=abs_start, end_s=abs_end, text=text_line))
        speech_cursor = speech_end

    # Snap last cue to end of last speech region, still within duration.
    last_end = min(duration, regions[-1][1])
    last = cues[-1]
    start = min(last.start_s, last_end)
    if last_end > start:
        end = last_end
    else:
        end = min(duration, start + max(min_len, 1e-3))
        if end <= start:
            start = max(0.0, duration - max(min_len, 1e-3))
            end = duration
    cues[-1] = TimedCue(start_s=start, end_s=end, text=last.text)
    return cues


def _speech_time_to_absolute(
    speech_t: float,
    regions: list[tuple[float, float]],
    region_lens: list[float],
) -> float:
    """Map a time on the concatenated speech axis to absolute audio time."""
    remaining = max(0.0, speech_t)
    for (start, end), length in zip(regions, region_lens, strict=True):
        if remaining <= length + 1e-9:
            return min(end, start + remaining)
        remaining -= length
    return regions[-1][1]


def detect_speech_regions(
    audio_path: Path,
    *,
    duration_s: float,
    noise_db: float = -35.0,
    min_silence_s: float = 0.35,
    ffmpeg_bin: str = "ffmpeg",
) -> list[tuple[float, float]]:
    """Return (start, end) speech intervals using FFmpeg silencedetect."""
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return [(0.0, duration_s)]

    log = (result.stderr or "") + (result.stdout or "")
    silences: list[tuple[float, float]] = []
    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(log)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(log)]
    # Pair silence intervals (silence_end may be fewer if trailing silence open).
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else duration_s
        silences.append((s, min(max(e, s + 0.01), duration_s)))

    if not silences:
        return [(0.0, duration_s)]

    # Invert silences → speech.
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor + 0.05:
            speech.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration_s - 0.05:
        speech.append((cursor, duration_s))
    if not speech:
        return [(0.0, duration_s)]
    return speech


class ProportionalAlignmentBackend:
    """Phrase cues timed by character weight within detected speech regions."""

    def __init__(self, *, ffmpeg_bin: str = "ffmpeg", use_silence: bool = True) -> None:
        self._ffmpeg_bin = ffmpeg_bin
        self._use_silence = use_silence

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
        cleaned = strip_text_for_alignment(text)
        regions: list[tuple[float, float]] | None = None
        if self._use_silence:
            regions = detect_speech_regions(
                path,
                duration_s=duration,
                ffmpeg_bin=self._ffmpeg_bin,
            )
        return proportional_cues(cleaned, duration, speech_regions=regions)


class FakeAlignmentBackend(ProportionalAlignmentBackend):
    """Test double: same algorithm, silence detection off for determinism."""

    def __init__(self) -> None:
        super().__init__(use_silence=False)


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
