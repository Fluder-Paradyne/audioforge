"""Prep stage: clean chapter text and write prepared files."""

from __future__ import annotations

from audioforge.backends.ollama_prep import OllamaTextPrep
from audioforge.backends.protocols import TextPrepBackend
from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions, ChapterProgress, ChapterRef
from audioforge.settings import AppSettings


class PrepError(Exception):
    """Raised when the prep stage cannot process a chapter."""


def prepared_filename(chapter: ChapterRef) -> str:
    """Return ``NNNN-slug.txt`` for *chapter*."""
    return f"{chapter.index:04d}-{chapter.slug}.txt"


def prep_chapters(
    *,
    chapters: list[ChapterRef],
    paths: JobPaths,
    options: BuildOptions,
    backend: TextPrepBackend,
    progress: list[ChapterProgress] | None = None,
) -> list[ChapterProgress]:
    """Prepare each chapter, write ``paths.prepared``, update progress.

    Resume: when *options.resume* is true, *options.force* is false, and the
    prepared file already exists, skip re-prep and mark ``prep_done``.

    Fail-fast: on the first error, record it on that chapter's progress and raise
    :class:`PrepError`.
    """
    paths.ensure()
    by_index = _progress_map(progress, chapters)

    for chapter in chapters:
        entry = by_index[chapter.index]
        out = paths.prepared / prepared_filename(chapter)

        if options.resume and not options.force and out.is_file():
            entry.prep_done = True
            entry.error = None
            continue

        try:
            raw = chapter.source_path.read_text(encoding="utf-8")
            prepared = backend.prepare(raw, options=options)
            out.write_text(prepared, encoding="utf-8")
            entry.prep_done = True
            entry.error = None
        except Exception as exc:
            entry.prep_done = False
            entry.error = str(exc)
            raise PrepError(
                f"Prep failed for chapter {chapter.index} ({chapter.slug}): {exc}"
            ) from exc

    return [by_index[c.index] for c in chapters]


def select_prep_backend(
    settings: AppSettings,
    options: BuildOptions,
    *,
    ollama_available: bool,
) -> TextPrepBackend:
    """Choose rules vs Ollama based on *options.skip_prep* and availability.

    Pure selection: does not perform network health checks; callers pass
    *ollama_available* after their own probe.
    """
    if options.skip_prep or not ollama_available:
        return RulesTextPrep()
    return OllamaTextPrep(base_url=str(settings.ollama_base_url))


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
