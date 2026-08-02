"""Align stage: force-align (or estimate) cues for each chapter WAV + text."""

from __future__ import annotations

from audioforge.backends.alignment import (
    alignment_filename,
    load_chapter_alignment,
    save_chapter_alignment,
)
from audioforge.backends.protocols import AlignmentBackend
from audioforge.io.paths import JobPaths
from audioforge.logging_config import get_logger
from audioforge.models import (
    BuildOptions,
    ChapterAlignment,
    ChapterProgress,
    ChapterRef,
)
from audioforge.pipeline.tts import audio_filename, prepared_filename

logger = get_logger(__name__)


class AlignStageError(Exception):
    """Raised when the align stage cannot process a chapter."""


def align_chapters(
    *,
    chapters: list[ChapterRef],
    paths: JobPaths,
    options: BuildOptions,
    backend: AlignmentBackend,
    progress: list[ChapterProgress] | None = None,
) -> list[ChapterProgress]:
    """Align each chapter; write ``paths.aligned / NNNN-slug.json``.

    Resume: when *options.resume* is true, *options.force* is false, and the
    alignment file already exists, skip and mark ``align_done``.

    Fail-fast: on the first error, record it and raise :class:`AlignStageError`.
    """
    paths.ensure()
    by_index = _progress_map(progress, chapters)

    for chapter in chapters:
        entry = by_index[chapter.index]
        prepared_path = paths.prepared / prepared_filename(chapter)
        audio_path = paths.audio / audio_filename(chapter)
        out = paths.aligned / alignment_filename(chapter.index, chapter.slug)

        if options.resume and not options.force and out.is_file():
            entry.align_done = True
            entry.error = None
            logger.info(
                "align skip chapter %s (resume)",
                chapter.index,
                extra={
                    "stage": "align",
                    "event": "chapter_skip",
                    "chapter_index": chapter.index,
                    "chapter_slug": chapter.slug,
                    "chapter_total": len(chapters),
                },
            )
            continue

        try:
            if not prepared_path.is_file():
                raise FileNotFoundError(f"Prepared text missing: {prepared_path}")
            if not audio_path.is_file():
                raise FileNotFoundError(f"Chapter audio missing: {audio_path}")
            text = prepared_path.read_text(encoding="utf-8")
            logger.info(
                "align chapter %s/%s %s",
                chapter.index,
                len(chapters),
                chapter.slug,
                extra={
                    "stage": "align",
                    "event": "chapter_start",
                    "chapter_index": chapter.index,
                    "chapter_slug": chapter.slug,
                    "chapter_total": len(chapters),
                },
            )
            cues = backend.align(audio_path, text, options=options)
            save_chapter_alignment(
                out,
                ChapterAlignment(chapter_index=chapter.index, cues=cues),
            )
            entry.align_done = True
            entry.error = None
            logger.info(
                "align done chapter %s (%s cues)",
                chapter.index,
                len(cues),
                extra={
                    "stage": "align",
                    "event": "chapter_end",
                    "chapter_index": chapter.index,
                    "chapter_slug": chapter.slug,
                    "chapter_total": len(chapters),
                },
            )
        except Exception as exc:
            entry.align_done = False
            entry.error = str(exc)
            logger.error(
                "align failed chapter %s (%s): %s",
                chapter.index,
                chapter.slug,
                exc,
                extra={
                    "stage": "align",
                    "event": "chapter_failed",
                    "chapter_index": chapter.index,
                    "chapter_slug": chapter.slug,
                    "chapter_total": len(chapters),
                },
            )
            raise AlignStageError(
                f"Align failed for chapter {chapter.index} ({chapter.slug}): {exc}"
            ) from exc

    return [by_index[c.index] for c in chapters]


def load_alignments_for_chapters(
    chapters: list[ChapterRef],
    paths: JobPaths,
) -> list[ChapterAlignment]:
    """Load alignment JSON for each chapter in order; raise if missing."""
    result: list[ChapterAlignment] = []
    for chapter in chapters:
        path = paths.aligned / alignment_filename(chapter.index, chapter.slug)
        if not path.is_file():
            raise AlignStageError(
                f"Alignment missing for chapter {chapter.index} "
                f"({chapter.slug}): {path}"
            )
        result.append(load_chapter_alignment(path))
    return result


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
