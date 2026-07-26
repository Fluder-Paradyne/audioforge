"""Tests for TTS pipeline stage (resume / force / fail-fast)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.backends.fake import FakeTtsBackend
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions, ChapterProgress, ChapterRef
from audioforge.pipeline.tts import (
    TtsError,
    audio_filename,
    prepared_filename,
    synthesize_chapters,
)


class RecordingTts:
    """Records synthesize calls; optionally fails."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, Path]] = []
        self.fail = fail

    def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path:
        self.calls.append((text, voice, out_path))
        if self.fail:
            raise RuntimeError("synth boom")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF-fake")
        return out_path


def _paths(tmp_path: Path, job_id: str = "job-tts") -> JobPaths:
    return JobPaths.for_job(tmp_path / "work", job_id).ensure()


def _chapter(
    *,
    index: int = 1,
    slug: str = "chapter-one",
    source: Path | None = None,
) -> ChapterRef:
    return ChapterRef(
        index=index,
        title="Chapter One",
        source_path=source or Path("src.md"),
        slug=slug,
    )


def _write_prepared(paths: JobPaths, chapter: ChapterRef, text: str) -> Path:
    path = paths.prepared / prepared_filename(chapter)
    path.write_text(text, encoding="utf-8")
    return path


def test_prepared_and_audio_filenames() -> None:
    ch = ChapterRef(
        index=12,
        title="T",
        source_path=Path("x.md"),
        slug="my-slug",
    )
    assert prepared_filename(ch) == "0012-my-slug.txt"
    assert audio_filename(ch) == "0012-my-slug.wav"


def test_synthesize_writes_audio_and_progress(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(index=1, slug="one"),
        _chapter(index=2, slug="two"),
    ]
    _write_prepared(paths, chapters[0], "First chapter text\n")
    _write_prepared(paths, chapters[1], "Second chapter text\n")
    backend = RecordingTts()
    options = BuildOptions(source=".", voice="af_bella", resume=False)

    progress = synthesize_chapters(
        chapters=chapters,
        paths=paths,
        options=options,
        backend=backend,
    )

    assert len(progress) == 2
    assert all(p.audio_done for p in progress)
    assert all(p.error is None for p in progress)
    out1 = paths.audio / "0001-one.wav"
    out2 = paths.audio / "0002-two.wav"
    assert out1.is_file()
    assert out2.is_file()
    assert backend.calls[0][0] == "First chapter text\n"
    assert backend.calls[0][1] == "af_bella"
    assert backend.calls[0][2] == out1


def test_tts_resume_skips_existing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "Prepared body\n")
    existing = paths.audio / "0001-chapter-one.wav"
    existing.write_bytes(b"already-audio")
    backend = RecordingTts()
    options = BuildOptions(source=".", resume=True, force=False)

    progress = synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )

    assert progress[0].audio_done is True
    assert backend.calls == []
    assert existing.read_bytes() == b"already-audio"


def test_tts_force_rewrites_even_with_resume(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "New prepared\n")
    existing = paths.audio / "0001-chapter-one.wav"
    existing.write_bytes(b"old-audio")
    backend = RecordingTts()
    options = BuildOptions(source=".", resume=True, force=True)

    synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )

    assert len(backend.calls) == 1
    assert existing.read_bytes() == b"RIFF-fake"


def test_tts_no_resume_rewrites(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "Body\n")
    existing = paths.audio / "0001-chapter-one.wav"
    existing.write_bytes(b"old")
    backend = RecordingTts()
    options = BuildOptions(source=".", resume=False, force=False)

    synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=backend,
    )
    assert existing.read_bytes() == b"RIFF-fake"


def test_tts_missing_prepared_fail_fast(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(slug="missing")
    backend = RecordingTts()
    options = BuildOptions(source=".", resume=False)
    progress = [ChapterProgress(chapter_index=1)]

    with pytest.raises(TtsError, match="chapter 1"):
        synthesize_chapters(
            chapters=[ch],
            paths=paths,
            options=options,
            backend=backend,
            progress=progress,
        )

    assert progress[0].audio_done is False
    assert progress[0].error is not None
    assert "Prepared text missing" in progress[0].error
    assert backend.calls == []


def test_tts_backend_fail_fast_records_error(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(index=1, slug="one"),
        _chapter(index=2, slug="two"),
    ]
    _write_prepared(paths, chapters[0], "ok\n")
    _write_prepared(paths, chapters[1], "also\n")
    backend = RecordingTts(fail=True)
    options = BuildOptions(source=".", resume=False)
    progress = [
        ChapterProgress(chapter_index=1),
        ChapterProgress(chapter_index=2),
    ]

    with pytest.raises(TtsError, match="chapter 1"):
        synthesize_chapters(
            chapters=chapters,
            paths=paths,
            options=options,
            backend=backend,
            progress=progress,
        )

    assert progress[0].audio_done is False
    assert progress[0].error is not None
    assert "synth boom" in progress[0].error
    assert progress[1].audio_done is False  # never reached


def test_tts_reuses_progress_entries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "Hi\n")
    progress = [ChapterProgress(chapter_index=1, prep_done=True)]
    options = BuildOptions(source=".", resume=False)

    result = synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=RecordingTts(),
        progress=progress,
    )

    assert result[0] is progress[0]
    assert result[0].audio_done is True
    assert result[0].prep_done is True


def test_tts_creates_progress_when_none(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "Hi\n")
    options = BuildOptions(source=".", resume=False)
    result = synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=RecordingTts(),
        progress=None,
    )
    assert len(result) == 1
    assert result[0].chapter_index == 1
    assert result[0].audio_done is True


def test_tts_with_fake_backend_integration(tmp_path: Path) -> None:
    import wave

    paths = _paths(tmp_path)
    ch = _chapter()
    _write_prepared(paths, ch, "Speak this chapter aloud.\n")
    options = BuildOptions(source=".", resume=False, voice="af_heart")
    progress = synthesize_chapters(
        chapters=[ch],
        paths=paths,
        options=options,
        backend=FakeTtsBackend(),
    )
    audio = paths.audio / "0001-chapter-one.wav"
    assert audio.is_file()
    with wave.open(str(audio), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
    assert progress[0].audio_done


def test_pipeline_exports_tts() -> None:
    from audioforge.pipeline import TtsError as TE
    from audioforge.pipeline import synthesize_chapters as sc

    assert TE is TtsError
    assert sc is synthesize_chapters


def test_backends_export_tts() -> None:
    from audioforge.backends import (
        FakeTtsBackend,
        KokoroNotInstalledError,
        KokoroTtsBackend,
        TtsBackend,
    )

    assert FakeTtsBackend is not None
    assert KokoroTtsBackend is not None
    assert KokoroNotInstalledError is not None
    assert TtsBackend is not None
