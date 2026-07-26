"""Tests for prep pipeline stage and backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions, ChapterProgress, ChapterRef
from audioforge.pipeline.prep import (
    PrepError,
    prep_chapters,
    prepared_filename,
    select_prep_backend,
)
from audioforge.settings import AppSettings


class RecordingBackend:
    """Records inputs and returns a fixed prepared string."""

    def __init__(self, result: str = "prepared\n", *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.result = result
        self.fail = fail

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        del options
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("backend boom")
        return self.result


def _paths(tmp_path: Path, job_id: str = "job-prep") -> JobPaths:
    return JobPaths.for_job(tmp_path / "work", job_id).ensure()


def _chapter(
    tmp_path: Path,
    *,
    index: int = 1,
    slug: str = "chapter-one",
    body: str = "# Chapter One\n\nHello.\n",
) -> ChapterRef:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    path = src / f"{index:04d}-{slug}.md"
    path.write_text(body, encoding="utf-8")
    return ChapterRef(index=index, title="Chapter One", source_path=path, slug=slug)


def test_prepared_filename() -> None:
    ch = ChapterRef(
        index=12,
        title="T",
        source_path=Path("x.md"),
        slug="my-slug",
    )
    assert prepared_filename(ch) == "0012-my-slug.txt"


def test_prep_writes_files_and_progress(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(tmp_path, index=1, slug="one", body="First\n"),
        _chapter(tmp_path, index=2, slug="two", body="Second\n"),
    ]
    backend = RecordingBackend(result="clean text\n")
    options = BuildOptions(source=".", resume=False)

    progress = prep_chapters(
        chapters=chapters,
        paths=paths,
        options=options,
        backend=backend,
    )

    assert len(progress) == 2
    assert all(p.prep_done for p in progress)
    assert all(p.error is None for p in progress)
    out1 = paths.prepared / "0001-one.txt"
    out2 = paths.prepared / "0002-two.txt"
    assert out1.read_text(encoding="utf-8") == "clean text\n"
    assert out2.read_text(encoding="utf-8") == "clean text\n"
    assert backend.calls == ["First\n", "Second\n"]


def test_prep_resume_skips_existing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(tmp_path, body="Source body\n")
    existing = paths.prepared / "0001-chapter-one.txt"
    existing.write_text("already prepared\n", encoding="utf-8")
    backend = RecordingBackend()
    options = BuildOptions(source=".", resume=True, force=False)

    progress = prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )

    assert progress[0].prep_done is True
    assert backend.calls == []
    assert existing.read_text(encoding="utf-8") == "already prepared\n"


def test_prep_force_rewrites_even_with_resume(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(tmp_path, body="New source\n")
    existing = paths.prepared / "0001-chapter-one.txt"
    existing.write_text("old\n", encoding="utf-8")
    backend = RecordingBackend(result="new prepared\n")
    options = BuildOptions(source=".", resume=True, force=True)

    prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )

    assert backend.calls == ["New source\n"]
    assert existing.read_text(encoding="utf-8") == "new prepared\n"


def test_prep_no_resume_rewrites(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(tmp_path, body="Body\n")
    existing = paths.prepared / "0001-chapter-one.txt"
    existing.write_text("old\n", encoding="utf-8")
    backend = RecordingBackend(result="fresh\n")
    options = BuildOptions(source=".", resume=False, force=False)

    prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )
    assert existing.read_text(encoding="utf-8") == "fresh\n"


def test_prep_fail_fast_records_error(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(tmp_path, index=1, slug="one", body="ok\n"),
        _chapter(tmp_path, index=2, slug="two", body="fail me\n"),
    ]
    backend = RecordingBackend(fail=True)
    options = BuildOptions(source=".", resume=False)
    progress = [ChapterProgress(chapter_index=1), ChapterProgress(chapter_index=2)]

    with pytest.raises(PrepError, match="chapter 1"):
        prep_chapters(
            chapters=chapters,
            paths=paths,
            options=options,
            backend=backend,
            progress=progress,
        )

    assert progress[0].prep_done is False
    assert progress[0].error is not None
    assert "backend boom" in progress[0].error
    assert not (paths.prepared / "0001-one.txt").exists()


def test_prep_reuses_progress_entries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(tmp_path, body="Hi\n")
    progress = [ChapterProgress(chapter_index=1, audio_done=True)]
    backend = RecordingBackend(result="p\n")
    options = BuildOptions(source=".", resume=False)

    result = prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
        progress=progress,
    )

    assert result is not progress  # returns ordered list, may be new list of same objs
    assert result[0] is progress[0]
    assert result[0].prep_done is True
    assert result[0].audio_done is True


def test_prep_creates_progress_when_none(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(tmp_path, body="Hi\n")
    options = BuildOptions(source=".", resume=False)
    result = prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=RecordingBackend(),
        progress=None,
    )
    assert len(result) == 1
    assert result[0].chapter_index == 1
    assert result[0].prep_done is True


def test_prep_with_rules_backend_integration(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    body = "She said, \u201chello\u201d.\n\n\n\n![x](y.png)\n"
    ch = _chapter(tmp_path, body=body)
    options = BuildOptions(source=".", resume=False, skip_prep=True)
    progress = prep_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=RulesTextPrep(),
    )
    text = (paths.prepared / "0001-chapter-one.txt").read_text(encoding="utf-8")
    assert 'She said, "hello".' in text
    assert "![x]" not in text
    assert progress[0].prep_done


def test_select_prep_backend_skip_uses_rules() -> None:
    settings = AppSettings()
    options = BuildOptions(source=".", skip_prep=True)
    backend = select_prep_backend(settings, options, ollama_available=True)
    assert isinstance(backend, RulesTextPrep)


def test_select_prep_backend_ollama_unavailable_uses_rules() -> None:
    settings = AppSettings()
    options = BuildOptions(source=".", skip_prep=False)
    backend = select_prep_backend(settings, options, ollama_available=False)
    assert isinstance(backend, RulesTextPrep)


def test_select_prep_backend_ollama_when_available() -> None:
    from audioforge.backends.ollama_prep import OllamaTextPrep

    settings = AppSettings(ollama_base_url="http://example:11434")
    options = BuildOptions(source=".", skip_prep=False)
    backend = select_prep_backend(settings, options, ollama_available=True)
    assert isinstance(backend, OllamaTextPrep)
    assert backend._base_url == "http://example:11434"
    backend.close()


def test_pipeline_exports_prep() -> None:
    from audioforge.pipeline import PrepError as PE
    from audioforge.pipeline import prep_chapters as pc
    from audioforge.pipeline import select_prep_backend as spb

    assert PE is PrepError
    assert pc is prep_chapters
    assert spb is select_prep_backend


def test_backends_export_prep() -> None:
    from audioforge.backends import OllamaPrepError, OllamaTextPrep, RulesTextPrep

    assert RulesTextPrep is not None
    assert OllamaTextPrep is not None
    assert OllamaPrepError is not None
