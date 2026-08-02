"""Tests for domain models and enums."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from audioforge.models import (
    ArtifactManifest,
    BuildOptions,
    ChapterProgress,
    ChapterRef,
    JobStage,
    JobState,
    JobStatus,
)


def test_job_status_values() -> None:
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.CANCELLED.value == "cancelled"
    assert JobStatus("pending") is JobStatus.PENDING
    assert set(JobStatus) == {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }


def test_job_stage_values() -> None:
    assert JobStage.INGEST.value == "ingest"
    assert JobStage.PREP.value == "prep"
    assert JobStage.TTS.value == "tts"
    assert JobStage.ALIGN.value == "align"
    assert JobStage.PACKAGE.value == "package"
    assert JobStage("tts") is JobStage.TTS
    assert set(JobStage) == {
        JobStage.INGEST,
        JobStage.PREP,
        JobStage.TTS,
        JobStage.ALIGN,
        JobStage.PACKAGE,
    }


def test_build_options_defaults() -> None:
    opts = BuildOptions(source="/books/sample")
    assert opts.source == "/books/sample"
    assert opts.voice == "af_heart"
    assert opts.prep_model == "llama3.2:3b"
    assert opts.skip_prep is False
    assert opts.resume is True
    assert opts.force is False
    assert opts.fictionreaper_bin == "fictionreaper"
    assert opts.output_dir is None
    assert opts.job_id is None


def test_build_options_custom() -> None:
    opts = BuildOptions(
        source="https://www.royalroad.com/fiction/1",
        voice="am_adam",
        prep_model="llama3.1:8b",
        skip_prep=True,
        resume=False,
        force=True,
        fictionreaper_bin="/usr/local/bin/fictionreaper",
        output_dir=Path("/tmp/out"),
        job_id="job-abc",
    )
    assert opts.voice == "am_adam"
    assert opts.skip_prep is True
    assert opts.resume is False
    assert opts.force is True
    assert opts.output_dir == Path("/tmp/out")
    assert opts.job_id == "job-abc"


def test_build_options_requires_source() -> None:
    with pytest.raises(ValidationError):
        BuildOptions()  # type: ignore[call-arg]


def test_chapter_ref_valid() -> None:
    ref = ChapterRef(
        index=1,
        title="Chapter One",
        source_path=Path("source/0001-chapter-one.md"),
        slug="chapter-one",
    )
    assert ref.index == 1
    assert ref.slug == "chapter-one"


def test_chapter_ref_index_must_be_ge_1() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChapterRef(
            index=0,
            title="Bad",
            source_path=Path("x.md"),
            slug="bad",
        )
    assert "index" in str(exc_info.value).lower() or any(
        e["loc"] == ("index",) for e in exc_info.value.errors()
    )


def test_chapter_ref_negative_index_rejected() -> None:
    with pytest.raises(ValidationError):
        ChapterRef(
            index=-1,
            title="Bad",
            source_path=Path("x.md"),
            slug="bad",
        )


def test_chapter_progress_defaults() -> None:
    progress = ChapterProgress(chapter_index=1)
    assert progress.prep_done is False
    assert progress.audio_done is False
    assert progress.error is None


def test_chapter_progress_with_error() -> None:
    progress = ChapterProgress(
        chapter_index=2,
        prep_done=True,
        audio_done=False,
        error="tts failed",
    )
    assert progress.error == "tts failed"
    assert progress.prep_done is True


def test_job_state_happy_path() -> None:
    opts = BuildOptions(source="/books/sample")
    state = JobState(
        job_id="abc123",
        source="/books/sample",
        options=opts,
        status=JobStatus.PENDING,
    )
    assert state.job_id == "abc123"
    assert state.stage is None
    assert state.chapters == []
    assert state.progress == []
    assert state.artifacts is None
    assert state.error is None
    assert state.created_at.tzinfo is not None
    assert state.updated_at.tzinfo is not None


def test_job_state_with_artifacts() -> None:
    opts = BuildOptions(source="/books/sample")
    manifest = ArtifactManifest(
        chapter_audio=[Path("audio/0001.wav")],
        m4b_path=Path("out/book.m4b"),
    )
    state = JobState(
        job_id="with-art",
        source="/books/sample",
        options=opts,
        status=JobStatus.COMPLETED,
        stage=JobStage.PACKAGE,
        artifacts=manifest,
    )
    assert state.artifacts is not None
    assert state.artifacts.m4b_path == Path("out/book.m4b")
    raw = state.model_dump_json()
    restored = JobState.model_validate_json(raw)
    assert restored.artifacts is not None
    assert restored.artifacts.m4b_path == Path("out/book.m4b")


def test_job_state_empty_job_id_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        JobState(
            job_id="",
            source="/x",
            options=BuildOptions(source="/x"),
            status=JobStatus.PENDING,
        )
    assert any(e["loc"] == ("job_id",) for e in exc_info.value.errors())


def test_job_state_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        JobState(
            job_id="x",
            source="/x",
            options=BuildOptions(source="/x"),
            status="not-a-status",  # type: ignore[arg-type]
        )


def test_job_state_invalid_stage_rejected() -> None:
    with pytest.raises(ValidationError):
        JobState(
            job_id="x",
            source="/x",
            options=BuildOptions(source="/x"),
            status=JobStatus.RUNNING,
            stage="warp",  # type: ignore[arg-type]
        )


def test_job_state_with_chapters_and_progress() -> None:
    chapter = ChapterRef(
        index=1,
        title="One",
        source_path=Path("/work/j/source/0001-one.md"),
        slug="one",
    )
    progress = ChapterProgress(chapter_index=1, prep_done=True)
    created = datetime(2026, 1, 1, tzinfo=UTC)
    updated = datetime(2026, 1, 2, tzinfo=UTC)
    state = JobState(
        job_id="job-1",
        source="/books",
        options=BuildOptions(source="/books"),
        status=JobStatus.RUNNING,
        stage=JobStage.PREP,
        chapters=[chapter],
        progress=[progress],
        error=None,
        created_at=created,
        updated_at=updated,
    )
    assert len(state.chapters) == 1
    assert state.stage == JobStage.PREP
    assert state.created_at == created


def test_job_state_path_json_roundtrip() -> None:
    """Paths serialize as strings in JSON and rehydrate to Path."""
    state = JobState(
        job_id="job-rt",
        source="/books",
        options=BuildOptions(source="/books", output_dir=Path("/tmp/out")),
        status=JobStatus.PENDING,
        chapters=[
            ChapterRef(
                index=1,
                title="One",
                source_path=Path("source/0001-one.md"),
                slug="one",
            )
        ],
    )
    raw = state.model_dump_json()
    assert '"source/0001-one.md"' in raw or '"source\\\\0001-one.md"' in raw
    restored = JobState.model_validate_json(raw)
    assert restored.chapters[0].source_path == Path("source/0001-one.md")
    assert restored.options.output_dir == Path("/tmp/out")


def test_artifact_manifest_defaults() -> None:
    manifest = ArtifactManifest(chapter_audio=[])
    assert manifest.chapter_audio == []
    assert manifest.m4b_path is None


def test_artifact_manifest_with_paths() -> None:
    manifest = ArtifactManifest(
        chapter_audio=[Path("audio/0001.wav"), Path("audio/0002.wav")],
        m4b_path=Path("out/book.m4b"),
    )
    assert len(manifest.chapter_audio) == 2
    assert manifest.m4b_path == Path("out/book.m4b")
    raw = manifest.model_dump_json()
    restored = ArtifactManifest.model_validate_json(raw)
    assert restored.m4b_path == Path("out/book.m4b")
