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
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from audioforge.backends.protocols import (
    AlignmentBackend,
    FfmpegRunner,
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.io.paths import JobPaths
from audioforge.jobstore import load_job, save_job
from audioforge.logging_config import (
    attach_job_file_handler,
    detach_handler,
    get_logger,
    job_logging_context,
)
from audioforge.models import (
    BuildOptions,
    ChapterProgress,
    ChapterRef,
    JobStage,
    JobState,
    JobStatus,
)
from audioforge.pipeline.align import align_chapters, load_alignments_for_chapters
from audioforge.pipeline.ingest import ingest
from audioforge.pipeline.package import package_book
from audioforge.pipeline.prep import prep_chapters
from audioforge.pipeline.tts import synthesize_chapters
from audioforge.settings import AppSettings

# Safe path segment from a local directory/file name (fallback when empty).
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
logger = get_logger(__name__)


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
    aligner: AlignmentBackend | None = None,
    job_id: str | None = None,
) -> JobState:
    """Run ingest → prep → tts → [align] → package, persisting job state.

    Parameters
    ----------
    options:
        Per-build knobs (source, voice, resume/force, subtitles, etc.).
    settings:
        App settings (workspace root under ``work_dir``).
    prep / tts / ffmpeg:
        Injectable backends (tests use fakes; CLI may use
        :func:`~audioforge.factory.create_default_backends`).
    fictionreaper:
        Required only when ``options.source`` is an ``http(s)`` URL.
    aligner:
        Required when subtitles are enabled (``subtitles`` and not
        ``skip_align``).
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
    want_subs = _want_subtitles(options)
    if want_subs and aligner is None:
        raise ValueError(
            "aligner backend is required when subtitles are enabled "
            "(pass aligner= or set skip_align/subtitles=false)"
        )

    resolved_id = _resolve_job_id(options, job_id)
    paths = JobPaths.for_job(settings.work_dir, resolved_id).ensure()
    state = _begin_job(paths, resolved_id, options)

    with _job_file_logging(paths, settings, resolved_id):
        try:
            # --- ingest ---
            _enter_stage(state, paths, JobStage.INGEST, resolved_id)
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
            logger.info(
                "ingest complete",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.INGEST.value,
                    "event": "stage_end",
                    "chapter_total": len(chapters),
                },
            )

            # --- prep ---
            _enter_stage(state, paths, JobStage.PREP, resolved_id)
            state.progress = prep_chapters(
                chapters=chapters,
                paths=paths,
                options=options,
                backend=prep,
                progress=state.progress,
            )
            _touch_and_save(state, paths)
            logger.info(
                "prep complete",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.PREP.value,
                    "event": "stage_end",
                },
            )

            # --- tts ---
            _enter_stage(state, paths, JobStage.TTS, resolved_id)
            state.progress = synthesize_chapters(
                chapters=chapters,
                paths=paths,
                options=options,
                backend=tts,
                progress=state.progress,
            )
            _touch_and_save(state, paths)
            logger.info(
                "tts complete",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.TTS.value,
                    "event": "stage_end",
                },
            )

            # --- align (optional) ---
            alignments = None
            if want_subs:
                assert aligner is not None
                _enter_stage(state, paths, JobStage.ALIGN, resolved_id)
                state.progress = align_chapters(
                    chapters=chapters,
                    paths=paths,
                    options=options,
                    backend=aligner,
                    progress=state.progress,
                )
                _touch_and_save(state, paths)
                alignments = load_alignments_for_chapters(chapters, paths)
                logger.info(
                    "align complete",
                    extra={
                        "job_id": resolved_id,
                        "stage": JobStage.ALIGN.value,
                        "event": "stage_end",
                    },
                )

            # --- package ---
            _enter_stage(state, paths, JobStage.PACKAGE, resolved_id)
            manifest = package_book(
                chapters=chapters,
                paths=paths,
                ffmpeg=ffmpeg,
                book_slug=_book_slug(resolved_id, options.source),
                alignments=alignments,
                include_subtitles=want_subs,
            )
            state.artifacts = manifest
            completed = _complete_job(state, paths)
            logger.info(
                "pipeline completed",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.PACKAGE.value,
                    "event": "pipeline_end",
                },
            )
            return completed
        except Exception as exc:
            logger.exception(
                "pipeline failed: %s",
                exc,
                extra={
                    "job_id": resolved_id,
                    "stage": state.stage.value if state.stage else None,
                    "event": "pipeline_failed",
                },
            )
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

    with _job_file_logging(paths, settings, resolved_id):
        try:
            _enter_stage(state, paths, JobStage.INGEST, resolved_id)
            chapters = ingest(
                source=options.source,
                paths=paths,
                options=options,
                runner=fictionreaper,
            )
            state.chapters = chapters
            state.progress = _merge_progress(state.progress, chapters)
            _touch_and_save(state, paths)
            logger.info(
                "ingest complete",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.INGEST.value,
                    "event": "stage_end",
                    "chapter_total": len(chapters),
                },
            )

            _enter_stage(state, paths, JobStage.PREP, resolved_id)
            state.progress = prep_chapters(
                chapters=chapters,
                paths=paths,
                options=options,
                backend=prep,
                progress=state.progress,
            )
            completed = _complete_job(state, paths)
            logger.info(
                "prepare completed",
                extra={
                    "job_id": resolved_id,
                    "stage": JobStage.PREP.value,
                    "event": "pipeline_end",
                },
            )
            return completed
        except Exception as exc:
            logger.exception(
                "prepare failed: %s",
                exc,
                extra={
                    "job_id": resolved_id,
                    "stage": state.stage.value if state.stage else None,
                    "event": "pipeline_failed",
                },
            )
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

    with _job_file_logging(paths, settings, state.job_id):
        if not state.chapters:
            state.status = JobStatus.FAILED
            state.error = "Job has no chapters; run prepare first"
            state.stage = JobStage.TTS
            _touch_and_save(state, paths)
            logger.error(
                "synthesize aborted: no chapters",
                extra={
                    "job_id": state.job_id,
                    "stage": JobStage.TTS.value,
                    "event": "pipeline_failed",
                },
            )
            raise PipelineError(state.error, state=state)

        options = state.options
        state.status = JobStatus.RUNNING
        state.error = None
        _touch_and_save(state, paths)

        try:
            _enter_stage(state, paths, JobStage.TTS, state.job_id)
            state.progress = synthesize_chapters(
                chapters=state.chapters,
                paths=paths,
                options=options,
                backend=tts,
                progress=state.progress,
            )
            completed = _complete_job(state, paths)
            logger.info(
                "synthesize completed",
                extra={
                    "job_id": state.job_id,
                    "stage": JobStage.TTS.value,
                    "event": "pipeline_end",
                },
            )
            return completed
        except Exception as exc:
            logger.exception(
                "synthesize failed: %s",
                exc,
                extra={
                    "job_id": state.job_id,
                    "stage": JobStage.TTS.value,
                    "event": "pipeline_failed",
                },
            )
            raise _fail_job(state, paths, exc) from exc


def run_package(
    settings: AppSettings,
    *,
    job_or_path: str,
    ffmpeg: FfmpegRunner,
    aligner: AlignmentBackend | None = None,
) -> JobState:
    """Package chapter audio into M4B for an existing job.

    Requires ``job.json`` with chapters. When job options enable subtitles,
    runs align if needed (using *aligner*) then muxes ``mov_text``.
    On success ``status=completed``, ``stage=package``, and ``artifacts`` set.
    """
    paths = resolve_job_paths(job_or_path, settings.work_dir)
    state = load_job(paths.job_json)

    with _job_file_logging(paths, settings, state.job_id):
        if not state.chapters:
            state.status = JobStatus.FAILED
            state.error = "Job has no chapters; run prepare first"
            state.stage = JobStage.PACKAGE
            _touch_and_save(state, paths)
            logger.error(
                "package aborted: no chapters",
                extra={
                    "job_id": state.job_id,
                    "stage": JobStage.PACKAGE.value,
                    "event": "pipeline_failed",
                },
            )
            raise PipelineError(state.error, state=state)

        state.status = JobStatus.RUNNING
        state.error = None
        _touch_and_save(state, paths)

        try:
            options = state.options
            want_subs = _want_subtitles(options)
            alignments = None
            if want_subs:
                if aligner is None:
                    raise ValueError(
                        "aligner backend is required when subtitles are enabled"
                    )
                _enter_stage(state, paths, JobStage.ALIGN, state.job_id)
                state.progress = align_chapters(
                    chapters=state.chapters,
                    paths=paths,
                    options=options,
                    backend=aligner,
                    progress=state.progress,
                )
                _touch_and_save(state, paths)
                alignments = load_alignments_for_chapters(state.chapters, paths)

            _enter_stage(state, paths, JobStage.PACKAGE, state.job_id)
            manifest = package_book(
                chapters=state.chapters,
                paths=paths,
                ffmpeg=ffmpeg,
                book_slug=_book_slug(state.job_id, state.source),
                alignments=alignments,
                include_subtitles=want_subs,
            )
            state.artifacts = manifest
            completed = _complete_job(state, paths)
            logger.info(
                "package completed",
                extra={
                    "job_id": state.job_id,
                    "stage": JobStage.PACKAGE.value,
                    "event": "pipeline_end",
                },
            )
            return completed
        except Exception as exc:
            logger.exception(
                "package failed: %s",
                exc,
                extra={
                    "job_id": state.job_id,
                    "stage": state.stage.value if state.stage else None,
                    "event": "pipeline_failed",
                },
            )
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
            aligned=root / "aligned",
            out=root / "out",
            job_json=job_json,
            job_log=root / "job.log",
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


@contextmanager
def _job_file_logging(
    paths: JobPaths,
    settings: AppSettings,
    job_id: str,
) -> Iterator[None]:
    """Attach filtered ``job.log`` and bind job context for the entrypoint.

    Detach always runs in an outer ``finally`` so bookend log I/O failures
    cannot leak ``FileHandler``s on long-lived API processes.
    """
    handler = attach_job_file_handler(
        paths.job_log,
        job_id=job_id,
        fmt=settings.log_format,
        level=settings.log_level,
    )
    try:
        with job_logging_context(job_id):
            # Bookend logging must not prevent the pipeline or detach.
            with suppress(Exception):
                logger.info(
                    "job started",
                    extra={"job_id": job_id, "event": "job_start"},
                )
            try:
                yield
            finally:
                with suppress(Exception):
                    logger.info(
                        "job log closed",
                        extra={"job_id": job_id, "event": "job_log_close"},
                    )
    finally:
        detach_handler(handler)


def _want_subtitles(options: BuildOptions) -> bool:
    """True when the build should align and mux a subtitle track."""
    return bool(options.subtitles) and not bool(options.skip_align)


def _enter_stage(
    state: JobState,
    paths: JobPaths,
    stage: JobStage,
    job_id: str,
) -> None:
    """Set stage, persist, and emit a stage_start log event."""
    state.stage = stage
    _touch_and_save(state, paths)
    logger.info(
        "stage start: %s",
        stage.value,
        extra={
            "job_id": job_id,
            "stage": stage.value,
            "event": "stage_start",
        },
    )


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
