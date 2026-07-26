"""Tests for full pipeline orchestrator (resume / fail-fast / persistence)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import FakeFfmpegRunner
from audioforge.backends.fictionreaper import FakeFictionReaperRunner
from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.io.paths import JobPaths
from audioforge.jobstore import load_job
from audioforge.models import BuildOptions, JobStage, JobStatus
from audioforge.pipeline.orchestrator import (
    PipelineError,
    _book_slug,
    _derive_job_id,
    _resolve_job_id,
    _slugify,
    run_pipeline,
)
from audioforge.settings import AppSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"


class RecordingPrep:
    """Rules prep that counts prepare calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = RulesTextPrep()

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        self.calls += 1
        return self._inner.prepare(text, options=options)


class BrokenPrep:
    """Always fails prepare (fail-fast fixture)."""

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        del text, options
        raise RuntimeError("prep deliberately broken")


class RecordingTts:
    """Fake TTS that counts synthesize calls and writes silent-ish bytes."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = FakeTtsBackend()

    def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path:
        self.calls += 1
        return self._inner.synthesize(text, voice=voice, out_path=out_path)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(work_dir=tmp_path / "work")


def test_run_pipeline_e2e_completed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(
        source=str(FIXTURES),
        job_id="e2e-job",
        skip_prep=True,
        resume=True,
    )
    ffmpeg = FakeFfmpegRunner()
    prep = RulesTextPrep()
    tts = FakeTtsBackend()

    state = run_pipeline(
        options,
        settings,
        prep=prep,
        tts=tts,
        ffmpeg=ffmpeg,
        fictionreaper=None,
    )

    assert state.status == JobStatus.COMPLETED
    assert state.stage == JobStage.PACKAGE
    assert state.error is None
    assert state.job_id == "e2e-job"
    assert len(state.chapters) == 2
    assert all(p.prep_done and p.audio_done for p in state.progress)
    assert state.artifacts is not None
    assert state.artifacts.m4b_path is not None
    assert state.artifacts.m4b_path.is_file()
    assert len(state.artifacts.chapter_audio) == 2
    for wav in state.artifacts.chapter_audio:
        assert wav.is_file()
    assert len(ffmpeg.commands) == 1

    # Persisted job.json matches completed state
    paths = JobPaths.for_job(settings.work_dir, "e2e-job")
    loaded = load_job(paths.job_json)
    assert loaded.status == JobStatus.COMPLETED
    assert loaded.artifacts is not None
    assert loaded.artifacts.m4b_path is not None


def test_run_pipeline_e2e_with_url_and_fake_fictionreaper(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(
        source="https://www.royalroad.com/fiction/999",
        job_id="url-job",
        skip_prep=True,
    )
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        fictionreaper=FakeFictionReaperRunner(),
    )
    assert state.status == JobStatus.COMPLETED
    assert len(state.chapters) == 2
    assert state.artifacts is not None
    assert state.artifacts.m4b_path is not None
    assert state.artifacts.m4b_path.is_file()


def test_run_pipeline_fail_fast_broken_prep(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(
        source=str(FIXTURES),
        job_id="fail-job",
        resume=False,
    )
    ffmpeg = FakeFfmpegRunner()

    with pytest.raises(PipelineError, match="prep deliberately broken") as exc_info:
        run_pipeline(
            options,
            settings,
            prep=BrokenPrep(),
            tts=FakeTtsBackend(),
            ffmpeg=ffmpeg,
        )

    err = exc_info.value
    assert err.state.status == JobStatus.FAILED
    assert err.state.stage == JobStage.PREP
    assert err.state.error is not None
    assert "prep deliberately broken" in err.state.error
    # Progress records chapter failure
    assert any(p.error is not None for p in err.state.progress)
    assert ffmpeg.commands == []

    paths = JobPaths.for_job(settings.work_dir, "fail-job")
    loaded = load_job(paths.job_json)
    assert loaded.status == JobStatus.FAILED
    assert loaded.error is not None
    assert "prep deliberately broken" in loaded.error


def test_run_pipeline_resume_skips_existing_work(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "resume-job"
    options = BuildOptions(
        source=str(FIXTURES),
        job_id=job_id,
        resume=True,
        force=False,
    )
    prep = RecordingPrep()
    tts = RecordingTts()
    ffmpeg = FakeFfmpegRunner()

    first = run_pipeline(
        options,
        settings,
        prep=prep,
        tts=tts,
        ffmpeg=ffmpeg,
        job_id=job_id,
    )
    assert first.status == JobStatus.COMPLETED
    assert prep.calls == 2
    assert tts.calls == 2
    first_m4b = first.artifacts.m4b_path if first.artifacts else None
    assert first_m4b is not None and first_m4b.is_file()

    prep2 = RecordingPrep()
    tts2 = RecordingTts()
    ffmpeg2 = FakeFfmpegRunner()
    second = run_pipeline(
        options,
        settings,
        prep=prep2,
        tts=tts2,
        ffmpeg=ffmpeg2,
        job_id=job_id,
    )
    assert second.status == JobStatus.COMPLETED
    # Resume: prepared + audio already on disk → no re-prep / re-synth
    assert prep2.calls == 0
    assert tts2.calls == 0
    # Package re-runs (no package-level skip)
    assert len(ffmpeg2.commands) == 1
    assert second.artifacts is not None
    assert second.artifacts.m4b_path is not None
    assert second.artifacts.m4b_path.is_file()


def test_run_pipeline_job_id_from_argument(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(source=str(FIXTURES), job_id="from-options")
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        job_id="from-arg",
    )
    assert state.job_id == "from-arg"


def test_run_pipeline_derives_job_id_from_source_slug(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(source=str(FIXTURES))  # no job_id
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
    )
    assert state.status == JobStatus.COMPLETED
    # sample_book slug prefix + short hex (underscores kept)
    assert state.job_id.startswith("sample_book-")
    assert len(state.job_id) > len("sample_book-")


def test_run_pipeline_derives_short_id_for_url(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(source="https://www.royalroad.com/fiction/1")
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        fictionreaper=FakeFictionReaperRunner(),
    )
    assert state.status == JobStatus.COMPLETED
    # UUID hex short (8 chars) when source is a URL
    assert len(state.job_id) == 8
    assert all(c in "0123456789abcdef" for c in state.job_id)


def test_run_pipeline_reloads_existing_job_json(tmp_path: Path) -> None:
    """Second run with same job keeps created_at from first persist."""
    settings = _settings(tmp_path)
    options = BuildOptions(source=str(FIXTURES), job_id="reload-job")
    first = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
    )
    created = first.created_at
    second = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
    )
    assert second.created_at == created
    assert second.status == JobStatus.COMPLETED


def test_run_pipeline_uses_options_job_id_when_arg_blank(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    options = BuildOptions(source=str(FIXTURES), job_id="opts-id")
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        job_id="   ",
    )
    assert state.job_id == "opts-id"


def test_slugify_and_id_helpers() -> None:
    assert _slugify("Foo Bar") == "foo-bar"
    assert _slugify("a--b") == "a-b"
    assert _slugify("@@@") == ""
    assert _slugify("") == ""
    assert _slugify("  Hello!!!World  ") == "hello-world"

    # Empty basename slug → plain short uuid
    short_id = _derive_job_id("@@@")
    assert len(short_id) == 8
    assert all(c in "0123456789abcdef" for c in short_id)

    assert _book_slug("my-job", "@@@") == "my-job"
    assert _book_slug("!!!", "https://example.com/x") == "audiobook"
    assert _book_slug("sample_book-abc", str(FIXTURES)) == "sample_book"

    opts = BuildOptions(source="/books/x", job_id="from-options")
    assert _resolve_job_id(opts, None) == "from-options"
    assert _resolve_job_id(opts, "arg-id") == "arg-id"
    blank_opts = BuildOptions(source="/books/x", job_id="  ")
    derived = _resolve_job_id(blank_opts, None)
    assert derived.startswith("x-") or len(derived) == 8
