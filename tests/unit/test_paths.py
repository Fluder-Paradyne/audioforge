"""Tests for JobPaths workspace helpers."""

from __future__ import annotations

from pathlib import Path

from audioforge.io.paths import JobPaths


def test_for_job_layout(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "job-42")
    root = tmp_path / "work" / "job-42"
    assert paths.root == root
    assert paths.source == root / "source"
    assert paths.prepared == root / "prepared"
    assert paths.audio == root / "audio"
    assert paths.aligned == root / "aligned"
    assert paths.out == root / "out"
    assert paths.job_json == root / "job.json"
    assert paths.job_log == root / "job.log"


def test_ensure_creates_directories(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "new-job")
    assert not paths.root.exists()
    returned = paths.ensure()
    assert returned is paths or returned.root == paths.root
    assert paths.root.is_dir()
    assert paths.source.is_dir()
    assert paths.prepared.is_dir()
    assert paths.audio.is_dir()
    assert paths.aligned.is_dir()
    assert paths.out.is_dir()
    # job.json is a file path; ensure should not create the file
    assert not paths.job_json.exists()


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "idem")
    paths.ensure()
    paths.ensure()
    assert paths.source.is_dir()


def test_for_job_different_ids_different_roots(tmp_path: Path) -> None:
    a = JobPaths.for_job(tmp_path, "a")
    b = JobPaths.for_job(tmp_path, "b")
    assert a.root != b.root
    assert a.job_json != b.job_json
