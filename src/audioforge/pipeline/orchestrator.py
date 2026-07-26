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
    state = _begin_job(paths, resolved_id, options)

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
        return _complete_job(state, paths)
    except Exception as exc:
        raise _fail_job(state, paths, exc) from exc


def run_prepare(
    options: BuildOptions,
    settings: AppSettings,
    *,
    prep: TextPrepBackend,
    fictionreaper: FictionReaperRunner | None = None,
    job_id: str | None = None,
) -> JobState:
    """Run ingest → prep only, persisting :class:`JobState`.

    On success ``status=completed`` and ``stage=prep``. Raises
    :class:`PipelineError` after persisting failure (same policy as
    :func:`run_pipeline`).
    """
    resolved_id = _resolve_job_id(options, job_id)
    paths = JobPaths.for_job(settings.work_dir, resolved_id).ensure()
    state = _begin_job(paths, resolved_id, options)

    try:
        state.stage = JobStage.INGEST
        _touch_and_save(state, paths)
        chapters = ingest(
            source=options.source,
            paths=paths,
            options=options,
            runner=fictionreaper,
        )
        state.chapters = chapters
        state.progress = _merge_progress(state.progress, chapters)
        _touch_and_save(state, paths)

        state.stage = JobStage.PREP
        _touch_and_save(state, paths)
        state.progress = prep_chapters(
            chapters=chapters,
            paths=paths,
            options=options,
            backend=prep,
            progress=state.progress,
        )
        return _complete_job(state, paths)
    except Exception as exc:
        raise _fail_job(state, paths, exc) from exc


def run_synthesize(
    settings: AppSettings,
    *,
    job_or_path: str,
    tts: TtsBackend,
) -> JobState:
    """Run TTS for an existing job (job id under work dir or job folder path).

    Requires ``job.json`` with chapters (run prepare first). On success
    ``status=completed`` and ``stage=tts``.
    """
    paths = resolve_job_paths(job_or_path, settings.work_dir)
    state = load_job(paths.job_json)
    if not state.chapters:
        state.status = JobStatus.FAILED
        state.error = "Job has no chapters; run prepare first"
        state.stage = JobStage.TTS
        _touch_and_save(state, paths)
        raise PipelineError(state.error, state=state)

    options = state.options
    state.status = JobStatus.RUNNING
    state.error = None
    _touch_and_save(state, paths)

    try:
        state.stage = JobStage.TTS
        _touch_and_save(state, paths)
        state.progress = synthesize_chapters(
            chapters=state.chapters,
            paths=paths,
            options=options,
            backend=tts,
            progress=state.progress,
        )
        return _complete_job(state, paths)
    except Exception as exc:
        raise _fail_job(state, paths, exc) from exc


def run_package(
    settings: AppSettings,
    *,
    job_or_path: str,
    ffmpeg: FfmpegRunner,
) -> JobState:
    """Package chapter audio into M4B for an existing job.

    Requires ``job.json`` with chapters. On success ``status=completed``,
    ``stage=package``, and ``artifacts`` set.
    """
    paths = resolve_job_paths(job_or_path, settings.work_dir)
    state = load_job(paths.job_json)
    if not state.chapters:
        state.status = JobStatus.FAILED
        state.error = "Job has no chapters; run prepare first"
        state.stage = JobStage.PACKAGE
        _touch_and_save(state, paths)
        raise PipelineError(state.error, state=state)

    state.status = JobStatus.RUNNING
    state.error = None
    _touch_and_save(state, paths)

    try:
        state.stage = JobStage.PACKAGE
        _touch_and_save(state, paths)
        manifest = package_book(
            chapters=state.chapters,
            paths=paths,
            ffmpeg=ffmpeg,
            book_slug=_book_slug(state.job_id, state.source),
        )
        state.artifacts = manifest
        return _complete_job(state, paths)
    except Exception as exc:
        raise _fail_job(state, paths, exc) from exc


def resolve_job_paths(job_or_path: str, work_dir: Path) -> JobPaths:
    """Resolve a job id or filesystem path to :class:`JobPaths`.

    Accepts:

    * Absolute/relative path to a job directory (containing ``job.json``)
    * Path to a ``job.json`` file
    * Job id string under *work_dir*

    Raises:
        FileNotFoundError: If no ``job.json`` can be found.
    """
    expanded = Path(job_or_path).expanduser()
    if expanded.exists():
        if expanded.is_file():
            root = expanded.parent.resolve()
            job_json = expanded.resolve()
        else:
            root = expanded.resolve()
            job_json = root / "job.json"
        paths = JobPaths(
            root=root,
            source=root / "source",
            prepared=root / "prepared",
            audio=root / "audio",
            out=root / "out",
            job_json=job_json,
        )
        if not paths.job_json.is_file():
            raise FileNotFoundError(f"No job.json found at {paths.job_json}")
        return paths

    paths = JobPaths.for_job(work_dir, job_or_path)
    if not paths.job_json.is_file():
        raise FileNotFoundError(
            f"No job found for {job_or_path!r} (looked for {paths.job_json})"
        )
    return paths


def _begin_job(
    paths: JobPaths,
    job_id: str,
    options: BuildOptions,
) -> JobState:
    state = _load_or_create_state(paths, job_id, options)
    state.status = JobStatus.RUNNING
    state.error = None
    state.options = options
    state.source = options.source
    _touch_and_save(state, paths)
    return state


def _complete_job(state: JobState, paths: JobPaths) -> JobState:
    state.status = JobStatus.COMPLETED
    state.error = None
    _touch_and_save(state, paths)
    return state


def _fail_job(
    state: JobState,
    paths: JobPaths,
    exc: BaseException,
) -> PipelineError:
    state.status = JobStatus.FAILED
    state.error = str(exc)
    _touch_and_save(state, paths)
    return PipelineError(str(exc), state=state)


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
