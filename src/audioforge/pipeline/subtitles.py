"""Build book-level WebVTT from per-chapter alignments."""

from __future__ import annotations

from pathlib import Path

from audioforge.models import ChapterAlignment, ChapterRef, TimedCue


class SubtitleError(Exception):
    """Raised when subtitle construction fails."""


def format_vtt_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS.mmm`` for WebVTT."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def offset_cues(
    cues: list[TimedCue],
    *,
    chapter_start_s: float,
    chapter_end_s: float,
) -> list[TimedCue]:
    """Shift chapter-relative cues into absolute book time; clamp to chapter span."""
    if chapter_end_s <= chapter_start_s:
        raise SubtitleError(
            f"Invalid chapter span: start={chapter_start_s} end={chapter_end_s}"
        )
    out: list[TimedCue] = []
    for cue in cues:
        start = chapter_start_s + cue.start_s
        end = chapter_start_s + cue.end_s
        start = max(chapter_start_s, min(start, chapter_end_s))
        end = max(start + 0.05, min(end, chapter_end_s))
        text = " ".join(cue.text.split()) or "…"
        out.append(TimedCue(start_s=start, end_s=end, text=text))
    return out


def build_book_vtt(
    chapters: list[ChapterRef],
    alignments: list[ChapterAlignment],
    durations_s: list[float],
) -> str:
    """Return WebVTT body for the whole book (absolute cue times)."""
    if not (len(chapters) == len(alignments) == len(durations_s)):
        raise SubtitleError(
            "chapters, alignments, and durations must have the same length"
        )
    lines: list[str] = ["WEBVTT", ""]
    cursor = 0.0
    cue_index = 1
    by_index = {a.chapter_index: a for a in alignments}
    for chapter, duration in zip(chapters, durations_s, strict=True):
        alignment = by_index.get(chapter.index)
        if alignment is None:
            raise SubtitleError(
                f"Missing alignment for chapter {chapter.index} ({chapter.slug})"
            )
        chapter_end = cursor + max(0.05, float(duration))
        absolute = offset_cues(
            alignment.cues,
            chapter_start_s=cursor,
            chapter_end_s=chapter_end,
        )
        for cue in absolute:
            lines.append(str(cue_index))
            lines.append(
                f"{format_vtt_timestamp(cue.start_s)} --> "
                f"{format_vtt_timestamp(cue.end_s)}"
            )
            lines.append(cue.text)
            lines.append("")
            cue_index += 1
        cursor = chapter_end
    body = "\n".join(lines)
    return body if body.endswith("\n") else body + "\n"


def write_book_vtt(path: Path, vtt_body: str) -> Path:
    """Write WebVTT file and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        vtt_body if vtt_body.endswith("\n") else vtt_body + "\n", encoding="utf-8"
    )
    return path
