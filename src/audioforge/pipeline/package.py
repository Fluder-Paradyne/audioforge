"""Package stage: verify chapter audio and produce a chaptered M4B via FFmpeg.

FFmpeg strategy (documented, deterministic for tests):

1. Resolve each chapter WAV under ``paths.audio / NNNN-slug.wav``.
2. Measure durations (default: ``wave`` module frame/rate) for chapter markers.
3. Write a concat demuxer list (``paths.out / concat.txt``) listing each WAV.
4. Write FFMETADATA1 chapters (``paths.out / chapters.ffmetadata``) with
   ``TIMEBASE=1/1000`` and cumulative START/END times in milliseconds.
5. Invoke the injectable :class:`~audioforge.backends.protocols.FfmpegRunner`::

       -y -f concat -safe 0 -i <concat> -i <ffmetadata>
       -map 0:a -map_metadata 1 -map_chapters 1
       -c:a aac -b:a 128k -movflags +faststart
       <paths.out / {book_slug}.m4b>

Chapter WAVs stay under ``paths.audio`` and are listed on the returned
:class:`~audioforge.models.ArtifactManifest`.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

from audioforge.backends.protocols import FfmpegRunner
from audioforge.io.paths import JobPaths
from audioforge.models import ArtifactManifest, ChapterRef
from audioforge.pipeline.tts import audio_filename

# Default book stem when *book_slug* is omitted.
_DEFAULT_BOOK_SLUG = "audiobook"

CONCAT_LIST_NAME = "concat.txt"
CHAPTERS_METADATA_NAME = "chapters.ffmetadata"


class PackageError(Exception):
    """Raised when the package stage cannot produce the M4B artifact."""


def wav_duration_seconds(path: Path) -> float:
    """Return duration in seconds of a PCM WAV file via the stdlib ``wave`` module."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise PackageError(f"Invalid WAV sample rate in {path}: {rate}")
        return frames / float(rate)


def package_book(
    *,
    chapters: list[ChapterRef],
    paths: JobPaths,
    ffmpeg: FfmpegRunner,
    book_slug: str | None = None,
    duration_seconds: Callable[[Path], float] | None = None,
) -> ArtifactManifest:
    """Verify chapter audio, write concat/metadata, encode chaptered M4B.

    Parameters
    ----------
    chapters:
        Ordered chapter references (audio names derived from index + slug).
    paths:
        Job workspace; audio is read from ``paths.audio``, outputs under
        ``paths.out``.
    ffmpeg:
        Injectable FFmpeg runner (real subprocess or fake).
    book_slug:
        Stem for the output M4B (``{book_slug}.m4b``). Defaults to
        ``\"audiobook\"``.
    duration_seconds:
        Callable returning chapter duration in seconds for marker END times.
        Defaults to :func:`wav_duration_seconds`.
    """
    if not chapters:
        raise PackageError("No chapters to package")

    paths.ensure()
    slug = book_slug if book_slug is not None else _DEFAULT_BOOK_SLUG
    if not slug.strip():
        raise PackageError("book_slug must be non-empty")

    probe = duration_seconds if duration_seconds is not None else wav_duration_seconds

    chapter_audio: list[Path] = []
    durations: list[float] = []
    for chapter in chapters:
        audio_path = paths.audio / audio_filename(chapter)
        if not audio_path.is_file():
            raise PackageError(
                f"Chapter audio missing for chapter {chapter.index} "
                f"({chapter.slug}): {audio_path}"
            )
        chapter_audio.append(audio_path)
        try:
            durations.append(probe(audio_path))
        except PackageError:
            raise
        except Exception as exc:
            raise PackageError(
                f"Failed to probe duration for {audio_path}: {exc}"
            ) from exc

    concat_path = paths.out / CONCAT_LIST_NAME
    metadata_path = paths.out / CHAPTERS_METADATA_NAME
    m4b_path = paths.out / f"{slug}.m4b"

    _write_concat_list(concat_path, chapter_audio)
    _write_ffmetadata(metadata_path, chapters, durations)

    args = [
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path.resolve()),
        "-i",
        str(metadata_path.resolve()),
        "-map",
        "0:a",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(m4b_path.resolve()),
    ]
    try:
        ffmpeg.run(args)
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"FFmpeg packaging failed: {exc}") from exc

    if not m4b_path.is_file():
        raise PackageError(f"FFmpeg did not produce M4B output: {m4b_path}")

    return ArtifactManifest(chapter_audio=chapter_audio, m4b_path=m4b_path)


def _write_concat_list(path: Path, audio_files: list[Path]) -> None:
    """Write an FFmpeg concat demuxer list for *audio_files*."""
    lines: list[str] = []
    for audio in audio_files:
        escaped = _concat_path_escape(audio.resolve())
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_path_escape(path: Path) -> str:
    """Escape a path for use inside single quotes in a concat demuxer list."""
    # FFmpeg concat demuxer: escape single quotes as '\''
    return str(path).replace("'", r"'\''")


def _write_ffmetadata(
    path: Path,
    chapters: list[ChapterRef],
    durations: list[float],
) -> None:
    """Write FFMETADATA1 chapter markers from cumulative *durations* (seconds)."""
    lines: list[str] = [";FFMETADATA1"]
    cursor_ms = 0
    for chapter, duration in zip(chapters, durations, strict=True):
        start_ms = cursor_ms
        # Clamp non-positive durations to 1 ms so markers remain ordered.
        duration_ms = max(1, int(round(duration * 1000.0)))
        end_ms = start_ms + duration_ms
        title = _ffmeta_escape(chapter.title)
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={title}",
            ]
        )
        cursor_ms = end_ms
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ffmeta_escape(value: str) -> str:
    """Escape special characters for FFMETADATA field values."""
    return (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", " ")
    )
