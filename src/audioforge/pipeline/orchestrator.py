"""Orchestrator: wire ingest → prep → tts → package with JobState persistence.

Fail-fast policy
----------------
On any stage error the orchestrator:

1. Sets ``JobState.status`` to :attr:`~audioforge.models.JobStatus.FAILED`
2. Records ``JobState.error`` with the exception message
3. Persists ``job.json`` (including any in-place progress updates from stages)
4. Raises :class:`PipelineError` with the failed ``state`` attached

Callers (CLI/API) should catch :class:`PipelineError` for a non-zero exit and
may read ``exc.state`` or reload ``job.json`` for the authoritative failed state.
Successful runs return the completed :class:`~audioforge.models.JobState`
without raising.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from audioforge.backends.protocols import (
    FfmpegRunner,
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.io.paths import JobPaths
from audioforge.jobstore import load_job, save_job
from audioforge.models import (
    BuildOptions,
    ChapterProgress,
    ChapterRef,
    JobStage,
    JobState,
    JobStatus,
)
from audioforge.pipeline.ingest import ingest
from audioforge.pipeline.package import package_book
from audioforge.pipeline.prep import prep_chapters
from audioforge.pipeline.tts import synthesize_chapters
from audioforge.settings import AppSettings

# Safe path segment from a local directory/file name (fallback when empty).
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class PipelineError(Exception):
    """Raised after a pipeline failure has been persisted to ``job.json``.

    Attributes:
        state: The failed :class:`~audioforge.models.JobState` that was saved.
    """

    def __init__(self, message: str, *, state: JobState) -> None:
        super().__init__(message)
        self.state = state


def run_pipeline(
    options: BuildOptions,
    settings: AppSettings,
    *,
    prep: TextPrepBackend,
    tts: TtsBackend,
    ffmpeg: FfmpegRunner,
    fictionreaper: FictionReaperRunner | None = None,
    job_id: str | None = None,
) -> JobState:
    """Run ingest → prep → tts → package, persisting :class:`JobState` at each stage.

    Parameters
    ----------
    options:
        Per-build knobs (source, voice, resume/force, etc.).
    settings:
        App settings (workspace root under ``work_dir``).
    prep / tts / ffmpeg:
        Injectable backends (tests use fakes; CLI may use
        :func:`~audioforge.factory.create_default_backends`).
    fictionreaper:
        Required only when ``options.source`` is an ``http(s)`` URL.
    job_id:
        Explicit job id. Falls back to ``options.job_id``, then a derived id
        (source slug + short uuid, or short uuid alone).

    Returns
    -------
    JobState
        Completed state with ``status=completed`` and ``artifacts`` set.

    Raises
    ------
    PipelineError
        After persisting ``status=failed`` and ``error`` on the job. The failed
        state is available as ``exc.state``.
    """
    resolved_id = _resolve_job_id(options, job_id)
    paths = JobPaths.for_job(settings.work_dir, resolved_id).ensure()
    state = _load_or_create_state(paths, resolved_id, options)
    state.status = JobStatus.RUNNING
    state.error = None
    state.options = options
    state.source = options.source
    _touch_and_save(state, paths)

    try:
        # --- ingest ---
        state.stage = JobStage.INGEST
        _touch_and_save(state, paths)
        chapters = ingest(
            source=options.source,
            paths=paths,
            options=options,
            runner=fictionreaper,
        )
        state.chapters = chapters
        # Share progress objects with stages so fail-fast mutations persist.
        state.progress = _merge_progress(state.progress, chapters)
        _touch_and_save(state, paths)

        # --- prep ---
        state.stage = JobStage.PREP
        _touch_and_save(state, paths)
        state.progress = prep_chapters(
            chapters=chapters,
            paths=paths,
            options=options,
            backend=prep,
            progress=state.progress,
        )
        _touch_and_save(state, paths)

        # --- tts ---
        state.stage = JobStage.TTS
        _touch_and_save(state, paths)
        state.progress = synthesize_chapters(
            chapters=chapters,
            paths=paths,
            options=options,
            backend=tts,
            progress=state.progress,
        )
        _touch_and_save(state, paths)

        # --- package ---
        state.stage = JobStage.PACKAGE
        _touch_and_save(state, paths)
        manifest = package_book(
            chapters=chapters,
            paths=paths,
            ffmpeg=ffmpeg,
            book_slug=_book_slug(resolved_id, options.source),
        )
        state.artifacts = manifest
        state.status = JobStatus.COMPLETED
        state.error = None
        _touch_and_save(state, paths)
        return state
    except Exception as exc:
        state.status = JobStatus.FAILED
        state.error = str(exc)
        _touch_and_save(state, paths)
        raise PipelineError(str(exc), state=state) from exc


def _resolve_job_id(options: BuildOptions, job_id: str | None) -> str:
    """Prefer explicit *job_id*, then options.job_id, else derive a short id."""
    if job_id is not None and job_id.strip():
        return job_id.strip()
    if options.job_id is not None and options.job_id.strip():
        return options.job_id.strip()
    return _derive_job_id(options.source)


def _derive_job_id(source: str) -> str:
    """Build ``{slug}-{hex8}`` from a local path basename, else short uuid."""
    short = uuid.uuid4().hex[:8]
    if _is_http_url(source):
        return short
    name = Path(source).expanduser().name
    slug = _slugify(name)
    if slug:
        return f"{slug}-{short}"
    return short


def _slugify(value: str) -> str:
    cleaned = _SLUG_RE.sub("-", value.strip()).strip("-").lower()
    # Collapse repeated dashes
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _book_slug(job_id: str, source: str) -> str:
    """Stem for the output M4B file (prefer source basename, else job id)."""
    if not _is_http_url(source):
        name = Path(source).expanduser().name
        slug = _slugify(name)
        if slug:
            return slug
    return _slugify(job_id) or "audiobook"


def _load_or_create_state(
    paths: JobPaths,
    job_id: str,
    options: BuildOptions,
) -> JobState:
    if paths.job_json.is_file():
        existing = load_job(paths.job_json)
        # Keep job_id / created_at from disk; caller refreshes status/options.
        return existing
    return JobState(
        job_id=job_id,
        source=options.source,
        options=options,
        status=JobStatus.PENDING,
    )


def _merge_progress(
    existing: list[ChapterProgress],
    chapters: list[ChapterRef],
) -> list[ChapterProgress]:
    by_index = {p.chapter_index: p for p in existing}
    merged: list[ChapterProgress] = []
    for chapter in chapters:
        if chapter.index in by_index:
            merged.append(by_index[chapter.index])
        else:
            merged.append(ChapterProgress(chapter_index=chapter.index))
    return merged


def _touch_and_save(state: JobState, paths: JobPaths) -> None:
    state.updated_at = datetime.now(UTC)
    save_job(state, paths.job_json)
