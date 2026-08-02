"""FastAPI application: jobs API and health."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from audioforge.api.schemas import CreateJobRequest, CreateJobResponse, HealthResponse
from audioforge.backends.protocols import (
    FfmpegRunner,
    FictionReaperRunner,
    TextPrepBackend,
    TtsBackend,
)
from audioforge.factory import create_default_backends
from audioforge.io.paths import JobPaths
from audioforge.jobstore import load_job
from audioforge.logging_config import configure_logging, get_logger
from audioforge.models import ArtifactManifest, BuildOptions, JobState, JobStatus
from audioforge.pipeline.orchestrator import PipelineError, run_pipeline
from audioforge.settings import AppSettings

logger = get_logger(__name__)


def create_app(
    settings: AppSettings | None = None,
    *,
    run_sync: bool = False,
    prep: TextPrepBackend | None = None,
    tts: TtsBackend | None = None,
    ffmpeg: FfmpegRunner | None = None,
    fictionreaper: FictionReaperRunner | None = None,
) -> FastAPI:
    """Build the AudioForge ASGI app.

    Parameters
    ----------
    settings:
        Application settings (workspace, backend defaults). Defaults to
        :class:`~audioforge.settings.AppSettings` from the environment.
    run_sync:
        When ``True``, ``POST /jobs`` runs the pipeline inline before returning
        (preferred for unit tests). When ``False`` (default), the pipeline runs
        in a FastAPI background task.
    prep / tts / ffmpeg / fictionreaper:
        Optional injectable backends. Any omitted backend is filled from
        :func:`~audioforge.factory.create_default_backends` at job start.
    """
    app_settings = settings if settings is not None else AppSettings()
    configure_logging(
        level=app_settings.log_level,
        fmt=app_settings.log_format,
    )
    logger.info(
        "API app created",
        extra={"event": "api_start"},
    )

    application = FastAPI(
        title="AudioForge",
        version="0.1.0",
        description="Local FictionReaper → audiobook pipeline API.",
    )

    def _resolve_backends(
        options: BuildOptions,
    ) -> tuple[
        TextPrepBackend,
        TtsBackend,
        FfmpegRunner,
        FictionReaperRunner | None,
    ]:
        if prep is not None and tts is not None and ffmpeg is not None:
            return prep, tts, ffmpeg, fictionreaper
        defaults = create_default_backends(app_settings, options)
        return (
            prep if prep is not None else defaults.prep,
            tts if tts is not None else defaults.tts,
            ffmpeg if ffmpeg is not None else defaults.ffmpeg,
            fictionreaper if fictionreaper is not None else defaults.fictionreaper,
        )

    def _options_from_request(body: CreateJobRequest) -> BuildOptions:
        job_id = body.job_id.strip() if body.job_id and body.job_id.strip() else None
        if job_id is None:
            job_id = uuid.uuid4().hex[:12]
        return BuildOptions(
            source=body.source,
            voice=body.voice or app_settings.default_voice,
            prep_model=body.prep_model or app_settings.default_prep_model,
            skip_prep=body.skip_prep,
            resume=body.resume,
            force=body.force,
            fictionreaper_bin=body.fictionreaper_bin,
            job_id=job_id,
        )

    def _execute_pipeline(options: BuildOptions) -> JobState:
        p, t, f, fr = _resolve_backends(options)
        try:
            return run_pipeline(
                options,
                app_settings,
                prep=p,
                tts=t,
                ffmpeg=f,
                fictionreaper=fr,
                job_id=options.job_id,
            )
        except PipelineError as exc:
            return exc.state

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness probe with configuration hints (no live pings)."""
        ffmpeg_path = (app_settings.ffmpeg_path or "").strip()
        return HealthResponse(
            status="ok",
            ffmpeg_configured=bool(ffmpeg_path),
            ollama_base_url=app_settings.ollama_base_url,
        )

    @application.post(
        "/jobs",
        response_model=CreateJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_job(
        body: CreateJobRequest,
        background_tasks: BackgroundTasks,
    ) -> CreateJobResponse:
        """Start a build job (sync or background depending on app config)."""
        options = _options_from_request(body)
        assert options.job_id is not None  # set in _options_from_request
        job_id = options.job_id
        logger.info(
            "job accepted",
            extra={"job_id": job_id, "event": "job_accepted"},
        )

        if run_sync:
            state = _execute_pipeline(options)
            return CreateJobResponse(job_id=state.job_id, status=state.status)

        background_tasks.add_task(_execute_pipeline, options)
        return CreateJobResponse(job_id=job_id, status=JobStatus.PENDING)

    @application.get("/jobs/{job_id}", response_model=JobState)
    def get_job(job_id: Annotated[str, "Job identifier under work_dir"]) -> JobState:
        """Return authoritative job state from ``job.json``."""
        paths = JobPaths.for_job(app_settings.work_dir, job_id)
        if not paths.job_json.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return load_job(paths.job_json)

    @application.get("/jobs/{job_id}/artifacts", response_model=ArtifactManifest)
    def get_artifacts(
        job_id: Annotated[str, "Job identifier under work_dir"],
    ) -> ArtifactManifest:
        """Return artifact paths when the job has produced them."""
        paths = JobPaths.for_job(app_settings.work_dir, job_id)
        if not paths.job_json.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        state = load_job(paths.job_json)
        if state.artifacts is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Artifacts not ready for job {job_id} "
                    f"(status={state.status.value})"
                ),
            )
        return state.artifacts

    return application
