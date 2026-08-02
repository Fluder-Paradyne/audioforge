"""Tests for the AudioForge CLI (Typer CliRunner)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from audioforge import __version__
from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import FakeFfmpegRunner
from audioforge.backends.fictionreaper import FakeFictionReaperRunner
from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.cli import app, run
from audioforge.factory import DefaultBackends
from audioforge.io.paths import JobPaths
from audioforge.jobstore import save_job
from audioforge.models import (
    BuildOptions,
    ChapterProgress,
    ChapterRef,
    JobStage,
    JobState,
    JobStatus,
)
from audioforge.pipeline.orchestrator import PipelineError

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"


def _runner() -> CliRunner:
    return CliRunner()


def _fake_backends() -> DefaultBackends:
    return DefaultBackends(
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        fictionreaper=FakeFictionReaperRunner(),
    )


def _patch_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "audioforge.cli.create_default_backends",
        lambda settings, options, **kwargs: _fake_backends(),
    )


def _completed_state(job_id: str = "job-1") -> JobState:
    return JobState(
        job_id=job_id,
        source="/books/x",
        options=BuildOptions(source="/books/x"),
        status=JobStatus.COMPLETED,
        stage=JobStage.PACKAGE,
    )


# --- scaffolding / version ---


def test_version_flag() -> None:
    """``--version`` prints package version and exits 0."""
    result = _runner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"audioforge {__version__}" in result.stdout


def test_version_short_flag() -> None:
    """``-V`` is an alias for ``--version``."""
    result = _runner().invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help() -> None:
    """``--help`` documents the CLI and exits 0."""
    result = _runner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AudioForge" in result.stdout or "audiobook" in result.stdout.lower()
    for name in (
        "build",
        "prepare",
        "synthesize",
        "package",
        "status",
        "doctor",
        "serve",
    ):
        assert name in result.stdout


def test_no_args_invokes_callback() -> None:
    """Bare invoke runs the root callback without error."""
    result = _runner().invoke(app, [])
    assert result.exit_code == 0


def test_run_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run()`` delegates to the Typer app."""
    called: list[bool] = []

    def fake_app() -> None:
        called.append(True)

    monkeypatch.setattr("audioforge.cli.app", fake_app)
    run()
    assert called == [True]


# --- build ---


def test_build_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_backends(monkeypatch)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))

    def fake_pipeline(
        options: BuildOptions,
        settings: Any,
        **kwargs: Any,
    ) -> JobState:
        assert options.source == str(FIXTURES)
        assert options.voice == "af_heart"
        assert options.skip_prep is True
        assert options.job_id == "cli-build"
        return _completed_state("cli-build")

    monkeypatch.setattr("audioforge.cli.run_pipeline", fake_pipeline)
    result = _runner().invoke(
        app,
        [
            "build",
            str(FIXTURES),
            "--skip-prep",
            "--job-id",
            "cli-build",
            "--voice",
            "af_heart",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "job_id: cli-build" in result.stdout
    assert "status: completed" in result.stdout
    assert "stage: package" in result.stdout


def test_build_pipeline_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backends(monkeypatch)
    failed = JobState(
        job_id="bad",
        source="/x",
        options=BuildOptions(source="/x"),
        status=JobStatus.FAILED,
        stage=JobStage.PREP,
        error="boom",
    )

    def boom(*args: Any, **kwargs: Any) -> JobState:
        raise PipelineError("boom", state=failed)

    monkeypatch.setattr("audioforge.cli.run_pipeline", boom)
    result = _runner().invoke(app, ["build", "/tmp/nope"])
    assert result.exit_code == 1
    assert "status: failed" in result.stdout
    assert "error: boom" in result.stdout
    assert "boom" in result.stdout or "boom" in result.stderr


def test_print_result_without_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover _print_job_result when stage is None."""
    _patch_backends(monkeypatch)
    no_stage = JobState(
        job_id="early",
        source="/x",
        options=BuildOptions(source="/x"),
        status=JobStatus.FAILED,
        stage=None,
        error="before stage",
    )

    def boom(*args: Any, **kwargs: Any) -> JobState:
        raise PipelineError("before stage", state=no_stage)

    monkeypatch.setattr("audioforge.cli.run_pipeline", boom)
    result = _runner().invoke(app, ["build", "/x"])
    assert result.exit_code == 1
    assert "job_id: early" in result.stdout
    assert "stage:" not in result.stdout
    assert "error: before stage" in result.stdout


def test_build_passes_all_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backends(monkeypatch)
    captured: dict[str, Any] = {}

    def fake_pipeline(
        options: BuildOptions,
        settings: Any,
        **kwargs: Any,
    ) -> JobState:
        captured["options"] = options
        captured["job_id"] = kwargs.get("job_id")
        return _completed_state("opts")

    monkeypatch.setattr("audioforge.cli.run_pipeline", fake_pipeline)
    result = _runner().invoke(
        app,
        [
            "build",
            "/src",
            "--output-dir",
            "/out",
            "--voice",
            "bf_emma",
            "--prep-model",
            "llama-test",
            "--skip-prep",
            "--no-resume",
            "--force",
            "--fictionreaper-bin",
            "/bin/fr",
            "--job-id",
            "opts",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    opts: BuildOptions = captured["options"]
    assert opts.voice == "bf_emma"
    assert opts.prep_model == "llama-test"
    assert opts.skip_prep is True
    assert opts.resume is False
    assert opts.force is True
    assert opts.fictionreaper_bin == "/bin/fr"
    assert opts.output_dir == Path("/out")
    assert opts.job_id == "opts"
    assert captured["job_id"] == "opts"


# --- prepare ---


def test_prepare_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backends(monkeypatch)

    def fake_prepare(
        options: BuildOptions,
        settings: Any,
        **kwargs: Any,
    ) -> JobState:
        return JobState(
            job_id="prep-1",
            source=options.source,
            options=options,
            status=JobStatus.COMPLETED,
            stage=JobStage.PREP,
        )

    monkeypatch.setattr("audioforge.cli.run_prepare", fake_prepare)
    result = _runner().invoke(app, ["prepare", str(FIXTURES), "--job-id", "prep-1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "job_id: prep-1" in result.stdout
    assert "stage: prep" in result.stdout


def test_prepare_pipeline_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backends(monkeypatch)
    failed = JobState(
        job_id="p",
        source="/x",
        options=BuildOptions(source="/x"),
        status=JobStatus.FAILED,
        stage=JobStage.INGEST,
        error="no source",
    )

    def boom(*args: Any, **kwargs: Any) -> JobState:
        raise PipelineError("no source", state=failed)

    monkeypatch.setattr("audioforge.cli.run_prepare", boom)
    result = _runner().invoke(app, ["prepare", "/missing"])
    assert result.exit_code == 1
    assert "failed" in result.stdout


# --- synthesize / package ---


def test_synthesize_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_backends(monkeypatch)
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "syn-job").ensure()
    state = JobState(
        job_id="syn-job",
        source=str(FIXTURES),
        options=BuildOptions(source=str(FIXTURES)),
        status=JobStatus.COMPLETED,
        stage=JobStage.PREP,
        chapters=[
            ChapterRef(
                index=1,
                title="One",
                source_path=paths.source / "0001-one.md",
                slug="one",
            )
        ],
        progress=[ChapterProgress(chapter_index=1, prep_done=True)],
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    def fake_synth(settings: Any, *, job_or_path: str, tts: Any) -> JobState:
        assert job_or_path == "syn-job"
        out = state.model_copy(deep=True)
        out.status = JobStatus.COMPLETED
        out.stage = JobStage.TTS
        return out

    monkeypatch.setattr("audioforge.cli.run_synthesize", fake_synth)
    result = _runner().invoke(app, ["synthesize", "syn-job"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "job_id: syn-job" in result.stdout
    assert "stage: tts" in result.stdout


def test_synthesize_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))
    result = _runner().invoke(app, ["synthesize", "does-not-exist"])
    assert result.exit_code == 1
    assert "No job" in result.stderr or "No job" in result.stdout


def test_synthesize_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_backends(monkeypatch)
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "s").ensure()
    state = JobState(
        job_id="s",
        source="/x",
        options=BuildOptions(source="/x"),
        status=JobStatus.COMPLETED,
        stage=JobStage.PREP,
        chapters=[
            ChapterRef(
                index=1,
                title="T",
                source_path=paths.source / "0001-t.md",
                slug="t",
            )
        ],
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    failed = state.model_copy(
        update={
            "status": JobStatus.FAILED,
            "stage": JobStage.TTS,
            "error": "tts fail",
        }
    )

    def boom(*args: Any, **kwargs: Any) -> JobState:
        raise PipelineError("tts fail", state=failed)

    monkeypatch.setattr("audioforge.cli.run_synthesize", boom)
    result = _runner().invoke(app, ["synthesize", "s"])
    assert result.exit_code == 1


def test_package_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_backends(monkeypatch)
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "pkg-job").ensure()
    state = JobState(
        job_id="pkg-job",
        source=str(FIXTURES),
        options=BuildOptions(source=str(FIXTURES)),
        status=JobStatus.COMPLETED,
        stage=JobStage.TTS,
        chapters=[
            ChapterRef(
                index=1,
                title="One",
                source_path=paths.source / "0001-one.md",
                slug="one",
            )
        ],
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    def fake_pkg(settings: Any, *, job_or_path: str, ffmpeg: Any) -> JobState:
        out = state.model_copy(deep=True)
        out.status = JobStatus.COMPLETED
        out.stage = JobStage.PACKAGE
        return out

    monkeypatch.setattr("audioforge.cli.run_package", fake_pkg)
    result = _runner().invoke(app, ["package", "pkg-job"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "stage: package" in result.stdout


def test_package_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))
    result = _runner().invoke(app, ["package", "gone"])
    assert result.exit_code == 1


def test_package_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_backends(monkeypatch)
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "p").ensure()
    state = JobState(
        job_id="p",
        source="/x",
        options=BuildOptions(source="/x"),
        status=JobStatus.COMPLETED,
        stage=JobStage.TTS,
        chapters=[
            ChapterRef(
                index=1,
                title="T",
                source_path=paths.source / "0001-t.md",
                slug="t",
            )
        ],
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    failed = state.model_copy(
        update={
            "status": JobStatus.FAILED,
            "error": "pkg fail",
            "stage": JobStage.PACKAGE,
        }
    )

    def boom(*args: Any, **kwargs: Any) -> JobState:
        raise PipelineError("pkg fail", state=failed)

    monkeypatch.setattr("audioforge.cli.run_package", boom)
    result = _runner().invoke(app, ["package", "p"])
    assert result.exit_code == 1


# --- status ---


def test_status_human(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "st").ensure()
    state = JobState(
        job_id="st",
        source="/books",
        options=BuildOptions(source="/books"),
        status=JobStatus.RUNNING,
        stage=JobStage.TTS,
        error=None,
        chapters=[
            ChapterRef(
                index=1,
                title="A",
                source_path=paths.source / "0001-a.md",
                slug="a",
            ),
            ChapterRef(
                index=2,
                title="B",
                source_path=paths.source / "0002-b.md",
                slug="b",
            ),
        ],
        progress=[
            ChapterProgress(chapter_index=1, prep_done=True, audio_done=True),
            ChapterProgress(chapter_index=2, prep_done=True, audio_done=False),
        ],
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    result = _runner().invoke(app, ["status", "st"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "job_id: st" in result.stdout
    assert "status: running" in result.stdout
    assert "stage: tts" in result.stdout
    assert "chapters: 2" in result.stdout
    assert "prep_done: 2/2" in result.stdout
    assert "audio_done: 1/2" in result.stdout


def test_status_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "work"
    paths = JobPaths.for_job(work, "j").ensure()
    state = JobState(
        job_id="j",
        source="/books",
        options=BuildOptions(source="/books"),
        status=JobStatus.FAILED,
        stage=JobStage.PREP,
        error="oops",
    )
    save_job(state, paths.job_json)
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(work))

    result = _runner().invoke(app, ["status", "j", "--json"])
    assert result.exit_code == 0
    assert '"job_id": "j"' in result.stdout
    assert '"oops"' in result.stdout


def test_status_by_path(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path, "path-job").ensure()
    state = JobState(
        job_id="path-job",
        source="/books",
        options=BuildOptions(source="/books"),
        status=JobStatus.COMPLETED,
        stage=JobStage.PACKAGE,
    )
    save_job(state, paths.job_json)
    result = _runner().invoke(app, ["status", str(paths.root)])
    assert result.exit_code == 0
    assert "job_id: path-job" in result.stdout


def test_status_by_job_json_path(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path, "json-job").ensure()
    state = JobState(
        job_id="json-job",
        source="/books",
        options=BuildOptions(source="/books"),
        status=JobStatus.PENDING,
    )
    save_job(state, paths.job_json)
    result = _runner().invoke(app, ["status", str(paths.job_json)])
    assert result.exit_code == 0
    assert "job_id: json-job" in result.stdout
    assert "stage: —" in result.stdout


def test_status_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))
    result = _runner().invoke(app, ["status", "missing"])
    assert result.exit_code == 1


# --- doctor ---


def test_doctor_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from audioforge.doctor import CheckStatus, DoctorCheck, DoctorReport

    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))

    def fake_doctor(settings: Any, **kwargs: Any) -> DoctorReport:
        return DoctorReport(
            version=__version__,
            checks=[
                DoctorCheck(
                    name="python",
                    status=CheckStatus.OK,
                    message="ok",
                    required=True,
                ),
            ],
        )

    monkeypatch.setattr("audioforge.cli.run_doctor", fake_doctor)
    result = _runner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "audioforge doctor" in result.stdout
    assert "ready" in result.stdout


def test_doctor_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from audioforge.doctor import CheckStatus, DoctorCheck, DoctorReport

    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))

    def fake_doctor(settings: Any, **kwargs: Any) -> DoctorReport:
        return DoctorReport(
            version=__version__,
            checks=[
                DoctorCheck(
                    name="ffmpeg",
                    status=CheckStatus.FAIL,
                    message="missing",
                    required=True,
                ),
            ],
        )

    monkeypatch.setattr("audioforge.cli.run_doctor", fake_doctor)
    result = _runner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "not ready" in result.stdout


def test_doctor_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from audioforge.doctor import CheckStatus, DoctorCheck, DoctorReport

    monkeypatch.setenv("AUDIOFORGE_WORK_DIR", str(tmp_path / "work"))

    def fake_doctor(settings: Any, **kwargs: Any) -> DoctorReport:
        return DoctorReport(
            version=__version__,
            checks=[
                DoctorCheck(
                    name="python",
                    status=CheckStatus.OK,
                    message="ok",
                    required=True,
                ),
            ],
        )

    monkeypatch.setattr("audioforge.cli.run_doctor", fake_doctor)
    result = _runner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["version"] == __version__
    assert payload["checks"][0]["name"] == "python"


# --- serve ---


def test_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    # Also ensure import path works when serve imports uvicorn
    result = _runner().invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9999"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(calls) == 1
    assert calls[0]["kwargs"]["host"] == "0.0.0.0"
    assert calls[0]["kwargs"]["port"] == 9999
    assert calls[0]["kwargs"]["factory"] is True
    assert calls[0]["args"][0] == "audioforge.api.app:create_app"


def test_serve_defaults_from_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setenv("AUDIOFORGE_HOST", "127.0.0.2")
    monkeypatch.setenv("AUDIOFORGE_PORT", "1234")
    result = _runner().invoke(app, ["serve"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls[0]["host"] == "127.0.0.2"
    assert calls[0]["port"] == 1234
