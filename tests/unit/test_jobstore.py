"""Tests for atomic job.json load/save."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from audioforge.jobstore import load_job, save_job
from audioforge.models import (
    BuildOptions,
    ChapterProgress,
    ChapterRef,
    JobStage,
    JobState,
    JobStatus,
)


def _sample_state() -> JobState:
    return JobState(
        job_id="job-store-1",
        source="/books/sample",
        options=BuildOptions(
            source="/books/sample",
            output_dir=Path("/tmp/out"),
        ),
        status=JobStatus.RUNNING,
        stage=JobStage.TTS,
        chapters=[
            ChapterRef(
                index=1,
                title="One",
                source_path=Path("source/0001-one.md"),
                slug="one",
            )
        ],
        progress=[ChapterProgress(chapter_index=1, prep_done=True, audio_done=False)],
        error=None,
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "work" / "job-store-1" / "job.json"
    state = _sample_state()
    save_job(state, path)

    assert path.is_file()
    loaded = load_job(path)
    assert loaded.job_id == state.job_id
    assert loaded.status == JobStatus.RUNNING
    assert loaded.stage == JobStage.TTS
    assert loaded.chapters[0].source_path == Path("source/0001-one.md")
    assert loaded.options.output_dir == Path("/tmp/out")
    assert loaded.progress[0].prep_done is True


def test_save_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    save_job(_sample_state(), path)
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*tmp*"))
    assert leftovers == []
    assert path.read_text(encoding="utf-8")


def test_save_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    first = _sample_state()
    save_job(first, path)
    second = first.model_copy(update={"status": JobStatus.COMPLETED, "stage": None})
    save_job(second, path)
    loaded = load_job(path)
    assert loaded.status == JobStatus.COMPLETED
    assert loaded.stage is None


def test_save_cleans_up_tmp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "job.json"
    state = _sample_state()

    def boom(_self: Path, _target: Path | str) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_job(state, path)
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*tmp*"))
    assert leftovers == []
    assert not path.exists()


def test_save_propagates_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "job.json"
    state = _sample_state()

    def boom_write(
        _self: Path,
        _data: str | bytes,
        *_args: object,
        **_kwargs: object,
    ) -> int:
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "write_text", boom_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_job(state, path)
    assert not path.exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_job(tmp_path / "missing.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_job(path)


def test_load_invalid_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text('{"job_id": ""}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_job(path)
