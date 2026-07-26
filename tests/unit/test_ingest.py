"""Tests for the ingest pipeline stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.backends.fictionreaper import FakeFictionReaperRunner
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions
from audioforge.pipeline.ingest import IngestError, ingest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"


def _paths(tmp_path: Path, job_id: str = "job-1") -> JobPaths:
    return JobPaths.for_job(tmp_path / "work", job_id).ensure()


def test_ingest_from_folder_copies_and_discovers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = BuildOptions(source=str(FIXTURES))
    chapters = ingest(source=str(FIXTURES), paths=paths, options=options)

    assert len(chapters) == 2
    assert chapters[0].title == "Chapter One"
    assert chapters[1].title == "Chapter Two"
    # Isolation: chapters live under the job source tree
    for chapter in chapters:
        assert chapter.source_path.exists()
        assert chapter.source_path.parent.resolve() == paths.source.resolve()
    assert (paths.source / "0001-chapter-one.md").is_file()
    assert (paths.source / "0002-chapter-two.md").is_file()


def test_ingest_from_url_with_fake_runner(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = BuildOptions(
        source="https://www.royalroad.com/fiction/12345",
        fictionreaper_bin="fictionreaper",
    )
    chapters = ingest(
        source=options.source,
        paths=paths,
        options=options,
        runner=FakeFictionReaperRunner(),
    )
    assert len(chapters) == 2
    assert all(
        c.source_path.parent.resolve() == paths.source.resolve() for c in chapters
    )


def test_ingest_url_without_runner_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = BuildOptions(source="https://www.royalroad.com/fiction/1")
    with pytest.raises(IngestError, match="no FictionReaper runner"):
        ingest(source=options.source, paths=paths, options=options, runner=None)


def test_ingest_missing_directory_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    missing = tmp_path / "does-not-exist"
    options = BuildOptions(source=str(missing))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ingest(source=str(missing), paths=paths, options=options)


def test_ingest_single_md_file_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    single = tmp_path / "only.md"
    single.write_text("# Only\n", encoding="utf-8")
    options = BuildOptions(source=str(single))
    with pytest.raises(IngestError, match="single Markdown file"):
        ingest(source=str(single), paths=paths, options=options)


def test_ingest_non_md_file_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    blob = tmp_path / "book.epub"
    blob.write_text("x", encoding="utf-8")
    options = BuildOptions(source=str(blob))
    with pytest.raises(IngestError, match="must be a directory"):
        ingest(source=str(blob), paths=paths, options=options)


def test_ingest_source_already_job_source(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.source / "0001-already.md").write_text("# Already\n", encoding="utf-8")
    options = BuildOptions(source=str(paths.source))
    chapters = ingest(source=str(paths.source), paths=paths, options=options)
    assert len(chapters) == 1
    assert chapters[0].title == "Already"


def test_ingest_url_runner_returns_nested_dir(tmp_path: Path) -> None:
    """If the runner drops chapters in a nested folder, copy into paths.source."""

    class NestedRunner:
        def run(self, url: str, output_dir: Path, *, bin_path: str) -> Path:
            del url, bin_path
            nested = Path(output_dir) / "book" / "chapters"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "0001-nested.md").write_text("# Nested\n", encoding="utf-8")
            return nested

    paths = _paths(tmp_path)
    options = BuildOptions(source="https://example.com/fiction/9")
    chapters = ingest(
        source=options.source,
        paths=paths,
        options=options,
        runner=NestedRunner(),
    )
    assert len(chapters) == 1
    assert chapters[0].title == "Nested"
    assert (paths.source / "0001-nested.md").is_file()


def test_ingest_copies_only_markdown(tmp_path: Path) -> None:
    src = tmp_path / "book"
    src.mkdir()
    (src / "0001-a.md").write_text("# A\n", encoding="utf-8")
    (src / "meta.json").write_text("{}", encoding="utf-8")
    (src / "book.epub").write_text("epub", encoding="utf-8")

    paths = _paths(tmp_path)
    options = BuildOptions(source=str(src))
    chapters = ingest(source=str(src), paths=paths, options=options)
    assert len(chapters) == 1
    assert (paths.source / "0001-a.md").is_file()
    assert not (paths.source / "meta.json").exists()
    assert not (paths.source / "book.epub").exists()


def test_ingest_http_url_without_netloc_treated_as_path(tmp_path: Path) -> None:
    """Bare scheme without host is not treated as a download URL."""
    paths = _paths(tmp_path)
    # "http://" alone is a URL scheme but no netloc — treat as path (missing)
    with pytest.raises(FileNotFoundError):
        ingest(
            source="http://",
            paths=paths,
            options=BuildOptions(source="http://"),
        )


def test_pipeline_package_exports() -> None:
    from audioforge.pipeline import IngestError as E
    from audioforge.pipeline import ingest as ingest_fn

    assert E is IngestError
    assert ingest_fn is ingest


def test_backends_package_exports() -> None:
    from audioforge.backends import (
        FakeFictionReaperRunner as F,
    )
    from audioforge.backends import (
        FictionReaperError as FE,
    )
    from audioforge.backends import (
        SubprocessFictionReaperRunner as S,
    )

    assert F is FakeFictionReaperRunner
    assert FE is not None
    assert S is not None


def test_ingest_special_file_not_dir_raises(tmp_path: Path) -> None:
    """Non-file, non-directory paths (e.g. FIFO) are rejected clearly."""
    import os

    paths = _paths(tmp_path)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    options = BuildOptions(source=str(fifo))
    with pytest.raises(IngestError, match="not a directory or URL"):
        ingest(source=str(fifo), paths=paths, options=options)
