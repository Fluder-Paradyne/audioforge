"""Command-line interface for AudioForge."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from audioforge import __version__
from audioforge.doctor import format_doctor_report, run_doctor
from audioforge.factory import create_default_backends
from audioforge.jobstore import load_job
from audioforge.logging_config import configure_logging
from audioforge.models import BuildOptions, JobState
from audioforge.pipeline.orchestrator import (
    PipelineError,
    resolve_job_paths,
    run_package,
    run_pipeline,
    run_prepare,
    run_synthesize,
)
from audioforge.settings import AppSettings

app = typer.Typer(
    name="audioforge",
    help="FictionReaper Markdown → single-voice audiobook (local pipeline).",
    invoke_without_command=True,
    add_completion=False,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """AudioForge CLI entrypoint."""
    settings = AppSettings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    if version:
        typer.echo(f"audioforge {__version__}")
        raise typer.Exit()


def _build_options(
    *,
    source: str,
    settings: AppSettings,
    output_dir: Path | None,
    voice: str | None,
    prep_model: str | None,
    skip_prep: bool,
    resume: bool,
    force: bool,
    fictionreaper_bin: str,
    job_id: str | None,
) -> BuildOptions:
    return BuildOptions(
        source=source,
        voice=voice if voice is not None else settings.default_voice,
        prep_model=(
            prep_model if prep_model is not None else settings.default_prep_model
        ),
        skip_prep=skip_prep,
        resume=resume,
        force=force,
        fictionreaper_bin=fictionreaper_bin,
        output_dir=output_dir,
        job_id=job_id,
    )


def _print_job_result(state: JobState) -> None:
    typer.echo(f"job_id: {state.job_id}")
    typer.echo(f"status: {state.status.value}")
    if state.stage is not None:
        typer.echo(f"stage: {state.stage.value}")
    if state.error:
        typer.echo(f"error: {state.error}")


def _exit_pipeline_error(exc: PipelineError) -> NoReturn:
    _print_job_result(exc.state)
    typer.echo(f"Pipeline failed: {exc}", err=True)
    raise typer.Exit(code=1)


@app.command()
def build(
    source: Annotated[
        str,
        typer.Argument(help="Local chapter directory or fiction URL."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Optional output directory hint."),
    ] = None,
    voice: Annotated[
        str | None,
        typer.Option("--voice", help="TTS voice id (default from settings)."),
    ] = None,
    prep_model: Annotated[
        str | None,
        typer.Option("--prep-model", help="Ollama prep model name."),
    ] = None,
    skip_prep: Annotated[
        bool,
        typer.Option("--skip-prep", help="Skip LLM/rules text prep."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip work already on disk."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Re-run stages even if artifacts exist.",
        ),
    ] = False,
    fictionreaper_bin: Annotated[
        str,
        typer.Option("--fictionreaper-bin", help="Path to fictionreaper binary."),
    ] = "fictionreaper",
    job_id: Annotated[
        str | None,
        typer.Option("--job-id", help="Explicit job id under the work directory."),
    ] = None,
) -> None:
    """Run the full ingest → prep → tts → package pipeline."""
    settings = AppSettings()
    options = _build_options(
        source=source,
        settings=settings,
        output_dir=output_dir,
        voice=voice,
        prep_model=prep_model,
        skip_prep=skip_prep,
        resume=resume,
        force=force,
        fictionreaper_bin=fictionreaper_bin,
        job_id=job_id,
    )
    backends = create_default_backends(settings, options)
    try:
        state = run_pipeline(
            options,
            settings,
            prep=backends.prep,
            tts=backends.tts,
            ffmpeg=backends.ffmpeg,
            fictionreaper=backends.fictionreaper,
            job_id=job_id,
        )
    except PipelineError as exc:
        _exit_pipeline_error(exc)
    _print_job_result(state)


@app.command()
def prepare(
    source: Annotated[
        str,
        typer.Argument(help="Local chapter directory or fiction URL."),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Optional output directory hint."),
    ] = None,
    voice: Annotated[
        str | None,
        typer.Option("--voice", help="TTS voice id (stored on job options)."),
    ] = None,
    prep_model: Annotated[
        str | None,
        typer.Option("--prep-model", help="Ollama prep model name."),
    ] = None,
    skip_prep: Annotated[
        bool,
        typer.Option("--skip-prep", help="Skip LLM/rules text prep."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip work already on disk."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Re-run stages even if artifacts exist.",
        ),
    ] = False,
    fictionreaper_bin: Annotated[
        str,
        typer.Option("--fictionreaper-bin", help="Path to fictionreaper binary."),
    ] = "fictionreaper",
    job_id: Annotated[
        str | None,
        typer.Option("--job-id", help="Explicit job id under the work directory."),
    ] = None,
) -> None:
    """Ingest source chapters and run text prep only."""
    settings = AppSettings()
    options = _build_options(
        source=source,
        settings=settings,
        output_dir=output_dir,
        voice=voice,
        prep_model=prep_model,
        skip_prep=skip_prep,
        resume=resume,
        force=force,
        fictionreaper_bin=fictionreaper_bin,
        job_id=job_id,
    )
    backends = create_default_backends(settings, options)
    try:
        state = run_prepare(
            options,
            settings,
            prep=backends.prep,
            fictionreaper=backends.fictionreaper,
            job_id=job_id,
        )
    except PipelineError as exc:
        _exit_pipeline_error(exc)
    _print_job_result(state)


@app.command()
def synthesize(
    job_or_path: Annotated[
        str,
        typer.Argument(help="Job id under work dir, or path to job folder/job.json."),
    ],
) -> None:
    """Synthesize prepared chapter audio for an existing job."""
    settings = AppSettings()
    # Load options from job for backend selection defaults.
    try:
        paths = resolve_job_paths(job_or_path, settings.work_dir)
        job = load_job(paths.job_json)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    backends = create_default_backends(settings, job.options)
    try:
        state = run_synthesize(
            settings,
            job_or_path=job_or_path,
            tts=backends.tts,
        )
    except PipelineError as exc:
        _exit_pipeline_error(exc)
    _print_job_result(state)


@app.command(name="package")
def package_cmd(
    job_or_path: Annotated[
        str,
        typer.Argument(help="Job id under work dir, or path to job folder/job.json."),
    ],
) -> None:
    """Package chapter audio into a chaptered M4B for an existing job."""
    settings = AppSettings()
    try:
        paths = resolve_job_paths(job_or_path, settings.work_dir)
        job = load_job(paths.job_json)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    backends = create_default_backends(settings, job.options)
    try:
        state = run_package(
            settings,
            job_or_path=job_or_path,
            ffmpeg=backends.ffmpeg,
        )
    except PipelineError as exc:
        _exit_pipeline_error(exc)
    _print_job_result(state)


@app.command()
def status(
    job_or_path: Annotated[
        str,
        typer.Argument(help="Job id under work dir, or path to job folder/job.json."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print full job.json as JSON."),
    ] = False,
) -> None:
    """Show job status (human summary or full JSON)."""
    settings = AppSettings()
    try:
        paths = resolve_job_paths(job_or_path, settings.work_dir)
        state = load_job(paths.job_json)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(state.model_dump_json(indent=2))
        return

    typer.echo(f"job_id: {state.job_id}")
    typer.echo(f"status: {state.status.value}")
    typer.echo(f"stage: {state.stage.value if state.stage else '—'}")
    typer.echo(f"error: {state.error if state.error else '—'}")
    if state.chapters:
        typer.echo(f"chapters: {len(state.chapters)}")
    if state.progress:
        done_prep = sum(1 for p in state.progress if p.prep_done)
        done_audio = sum(1 for p in state.progress if p.audio_done)
        typer.echo(f"prep_done: {done_prep}/{len(state.progress)}")
        typer.echo(f"audio_done: {done_audio}/{len(state.progress)}")


@app.command()
def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print full doctor report as JSON."),
    ] = False,
    fictionreaper_bin: Annotated[
        str,
        typer.Option(
            "--fictionreaper-bin",
            help="FictionReaper binary to look for on PATH.",
        ),
    ] = "fictionreaper",
) -> None:
    """Check local dependencies (FFmpeg, Kokoro, Ollama, work dir, …).

    May create the work directory and briefly write a probe file. Contacts
    Ollama at the configured base URL when probing optional LLM prep.
    """
    settings = AppSettings()
    report = run_doctor(settings, fictionreaper_bin=fictionreaper_bin)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(format_doctor_report(report), nl=False)
    raise typer.Exit(code=0 if report.ok else 1)


@app.command()
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind host (default from settings)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Bind port (default from settings)."),
    ] = None,
) -> None:
    """Start the local HTTP API (uvicorn)."""
    import uvicorn

    settings = AppSettings()
    bind_host = host if host is not None else settings.host
    bind_port = port if port is not None else settings.port
    uvicorn.run(
        "audioforge.api.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
    )


def run() -> None:
    """Console-script friendly entry that invokes the Typer app."""
    app()
