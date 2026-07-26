"""TTS stage: synthesize prepared chapter text to audio files."""

from __future__ import annotations

from audioforge.backends.protocols import TtsBackend
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions, ChapterProgress, ChapterRef


class TtsError(Exception):
    """Raised when the TTS stage cannot synthesize a chapter."""


def prepared_filename(chapter: ChapterRef) -> str:
    """Return ``NNNN-slug.txt`` for *chapter* (prepared input name)."""
    return f"{chapter.index:04d}-{chapter.slug}.txt"


def audio_filename(chapter: ChapterRef) -> str:
    """Return ``NNNN-slug.wav`` for *chapter*."""
    return f"{chapter.index:04d}-{chapter.slug}.wav"


def synthesize_chapters(
    *,
    chapters: list[ChapterRef],
    paths: JobPaths,
    options: BuildOptions,
    backend: TtsBackend,
    progress: list[ChapterProgress] | None = None,
) -> list[ChapterProgress]:
    """Synthesize each chapter from prepared text into ``paths.audio``.

    Reads ``paths.prepared / NNNN-slug.txt`` and writes
    ``paths.audio / NNNN-slug.wav``.

    Resume: when *options.resume* is true, *options.force* is false, and the
    audio file already exists, skip synthesis and mark ``audio_done``.

    Fail-fast: on the first error, record it on that chapter's progress and raise
    :class:`TtsError`.
    """
    paths.ensure()
    by_index = _progress_map(progress, chapters)

    for chapter in chapters:
        entry = by_index[chapter.index]
        prepared_path = paths.prepared / prepared_filename(chapter)
        out = paths.audio / audio_filename(chapter)

        if options.resume and not options.force and out.is_file():
            entry.audio_done = True
            entry.error = None
            continue

        try:
            if not prepared_path.is_file():
                raise FileNotFoundError(f"Prepared text missing: {prepared_path}")
            text = prepared_path.read_text(encoding="utf-8")
            backend.synthesize(text, voice=options.voice, out_path=out)
            entry.audio_done = True
            entry.error = None
        except Exception as exc:
            entry.audio_done = False
            entry.error = str(exc)
            raise TtsError(
                f"TTS failed for chapter {chapter.index} ({chapter.slug}): {exc}"
            ) from exc

    return [by_index[c.index] for c in chapters]


def _progress_map(
    progress: list[ChapterProgress] | None,
    chapters: list[ChapterRef],
) -> dict[int, ChapterProgress]:
    result: dict[int, ChapterProgress] = {}
    if progress is not None:
        for item in progress:
            result[item.chapter_index] = item
    for chapter in chapters:
        if chapter.index not in result:
            result[chapter.index] = ChapterProgress(chapter_index=chapter.index)
    return result
