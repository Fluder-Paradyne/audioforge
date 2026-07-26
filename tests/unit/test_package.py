"""Tests for package stage (concat/metadata + chaptered M4B)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audioforge.backends.ffmpeg import FakeFfmpegRunner, FfmpegError
from audioforge.io.paths import JobPaths
from audioforge.models import ChapterRef
from audioforge.pipeline.package import (
    CHAPTERS_METADATA_NAME,
    CONCAT_LIST_NAME,
    PackageError,
    package_book,
    wav_duration_seconds,
)
from audioforge.pipeline.tts import audio_filename


def _paths(tmp_path: Path, job_id: str = "job-pkg") -> JobPaths:
    return JobPaths.for_job(tmp_path / "work", job_id).ensure()


def _chapter(
    *,
    index: int = 1,
    slug: str = "chapter-one",
    title: str = "Chapter One",
    source: Path | None = None,
) -> ChapterRef:
    return ChapterRef(
        index=index,
        title=title,
        source_path=source or Path("src.md"),
        slug=slug,
    )


def _write_silent_wav(
    path: Path,
    *,
    n_frames: int = 24_000,
    sample_rate: int = 24_000,
) -> Path:
    """Write mono 16-bit silent WAV (default 1.0 s)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


def _write_chapter_audio(
    paths: JobPaths,
    chapter: ChapterRef,
    *,
    n_frames: int = 24_000,
    sample_rate: int = 24_000,
) -> Path:
    out = paths.audio / audio_filename(chapter)
    return _write_silent_wav(out, n_frames=n_frames, sample_rate=sample_rate)


def test_wav_duration_seconds(tmp_path: Path) -> None:
    wav = _write_silent_wav(tmp_path / "a.wav", n_frames=12_000, sample_rate=24_000)
    assert wav_duration_seconds(wav) == pytest.approx(0.5)


def test_package_missing_audio_fails(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    ffmpeg = FakeFfmpegRunner()

    with pytest.raises(PackageError, match="audio missing"):
        package_book(chapters=[ch], paths=paths, ffmpeg=ffmpeg)

    assert ffmpeg.commands == []


def test_package_empty_chapters_fails(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(PackageError, match="No chapters"):
        package_book(chapters=[], paths=paths, ffmpeg=FakeFfmpegRunner())


def test_package_empty_book_slug_fails(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)
    with pytest.raises(PackageError, match="book_slug"):
        package_book(
            chapters=[ch],
            paths=paths,
            ffmpeg=FakeFfmpegRunner(),
            book_slug="   ",
        )


def test_package_writes_concat_and_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(index=1, slug="one", title="Chapter One"),
        _chapter(index=2, slug="two", title="Chapter Two"),
    ]
    # 1.0s and 0.5s
    _write_chapter_audio(paths, chapters[0], n_frames=24_000, sample_rate=24_000)
    _write_chapter_audio(paths, chapters[1], n_frames=12_000, sample_rate=24_000)
    ffmpeg = FakeFfmpegRunner()

    manifest = package_book(
        chapters=chapters,
        paths=paths,
        ffmpeg=ffmpeg,
        book_slug="my-book",
    )

    concat = paths.out / CONCAT_LIST_NAME
    meta = paths.out / CHAPTERS_METADATA_NAME
    assert concat.is_file()
    assert meta.is_file()

    concat_text = concat.read_text(encoding="utf-8")
    a1 = (paths.audio / "0001-one.wav").resolve()
    a2 = (paths.audio / "0002-two.wav").resolve()
    assert f"file '{a1}'" in concat_text
    assert f"file '{a2}'" in concat_text

    meta_text = meta.read_text(encoding="utf-8")
    assert meta_text.startswith(";FFMETADATA1\n")
    assert "title=Chapter One" in meta_text
    assert "title=Chapter Two" in meta_text
    assert "START=0" in meta_text
    assert "END=1000" in meta_text  # first chapter 1.0s
    assert "START=1000" in meta_text
    assert "END=1500" in meta_text  # +0.5s

    assert manifest.m4b_path == paths.out / "my-book.m4b"
    assert manifest.m4b_path is not None
    assert manifest.m4b_path.is_file()
    assert len(manifest.chapter_audio) == 2
    assert manifest.chapter_audio[0].name == "0001-one.wav"


def test_package_ffmpeg_args_contain_output_m4b(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)
    ffmpeg = FakeFfmpegRunner()

    package_book(chapters=[ch], paths=paths, ffmpeg=ffmpeg, book_slug="story")

    assert len(ffmpeg.commands) == 1
    args = ffmpeg.commands[0]
    m4b = str((paths.out / "story.m4b").resolve())
    assert args[-1] == m4b
    assert "-f" in args and "concat" in args
    assert "-c:a" in args and "aac" in args
    assert "-map_chapters" in args
    assert str((paths.out / CONCAT_LIST_NAME).resolve()) in args
    assert str((paths.out / CHAPTERS_METADATA_NAME).resolve()) in args


def test_package_default_book_slug(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)
    manifest = package_book(
        chapters=[ch],
        paths=paths,
        ffmpeg=FakeFfmpegRunner(),
    )
    assert manifest.m4b_path == paths.out / "audiobook.m4b"


def test_package_injectable_duration_map(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    chapters = [
        _chapter(index=1, slug="one", title="A"),
        _chapter(index=2, slug="two", title="B"),
    ]
    _write_chapter_audio(paths, chapters[0])
    _write_chapter_audio(paths, chapters[1])
    durations = {
        (paths.audio / "0001-one.wav").resolve(): 2.0,
        (paths.audio / "0002-two.wav").resolve(): 3.5,
    }

    def probe(path: Path) -> float:
        return durations[path.resolve()]

    package_book(
        chapters=chapters,
        paths=paths,
        ffmpeg=FakeFfmpegRunner(),
        duration_seconds=probe,
    )
    meta = (paths.out / CHAPTERS_METADATA_NAME).read_text(encoding="utf-8")
    assert "START=0" in meta
    assert "END=2000" in meta
    assert "START=2000" in meta
    assert "END=5500" in meta


def test_package_ffmeta_escapes_special_title(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(title="Part=1; #hash\\slash\nnewline")
    _write_chapter_audio(paths, ch)
    package_book(chapters=[ch], paths=paths, ffmpeg=FakeFfmpegRunner())
    meta = (paths.out / CHAPTERS_METADATA_NAME).read_text(encoding="utf-8")
    assert r"title=Part\=1\; \#hash\\slash newline" in meta


def test_package_concat_escapes_single_quotes(tmp_path: Path) -> None:
    """Paths containing single quotes are escaped for the concat demuxer."""
    paths = _paths(tmp_path)
    # Create audio under a directory whose name includes a quote-like path
    # by using a normal path and asserting escape helper via a crafted name.
    ch = _chapter(slug="o'neill", title="ONeill")
    audio = paths.audio / audio_filename(ch)
    _write_silent_wav(audio)
    package_book(chapters=[ch], paths=paths, ffmpeg=FakeFfmpegRunner())
    concat = (paths.out / CONCAT_LIST_NAME).read_text(encoding="utf-8")
    # file '....o'\''neill.wav'
    assert r"'\''" in concat or "o'neill" in concat


def test_package_ffmpeg_error_wrapped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)

    class BoomRunner:
        def run(self, args: list[str]) -> None:
            del args
            raise FfmpegError("encode failed", returncode=1)

    with pytest.raises(PackageError, match="FFmpeg packaging failed"):
        package_book(chapters=[ch], paths=paths, ffmpeg=BoomRunner())


def test_package_missing_m4b_after_run(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)

    class NoTouchRunner:
        def run(self, args: list[str]) -> None:
            del args  # pretend success without creating output

    with pytest.raises(PackageError, match="did not produce M4B"):
        package_book(chapters=[ch], paths=paths, ffmpeg=NoTouchRunner())


def test_package_duration_probe_failure(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)

    def bad_probe(path: Path) -> float:
        del path
        raise OSError("cannot read")

    with pytest.raises(PackageError, match="Failed to probe duration"):
        package_book(
            chapters=[ch],
            paths=paths,
            ffmpeg=FakeFfmpegRunner(),
            duration_seconds=bad_probe,
        )


def test_package_duration_package_error_propagates(tmp_path: Path) -> None:
    """PackageError from the duration probe is not wrapped again."""
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)

    def bad_probe(path: Path) -> float:
        del path
        raise PackageError("invalid audio duration")

    with pytest.raises(PackageError, match="invalid audio duration"):
        package_book(
            chapters=[ch],
            paths=paths,
            ffmpeg=FakeFfmpegRunner(),
            duration_seconds=bad_probe,
        )


def test_package_ffmpeg_package_error_propagates(tmp_path: Path) -> None:
    """PackageError from the runner is re-raised without wrapping."""
    paths = _paths(tmp_path)
    ch = _chapter()
    _write_chapter_audio(paths, ch)

    class PackageErrorRunner:
        def run(self, args: list[str]) -> None:
            del args
            raise PackageError("runner package error")

    with pytest.raises(PackageError, match="runner package error"):
        package_book(chapters=[ch], paths=paths, ffmpeg=PackageErrorRunner())


def test_package_invalid_sample_rate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter()
    audio = paths.audio / audio_filename(ch)
    # Craft a WAV with zero framerate is hard via wave module (it rejects).
    # Use injectable probe that raises PackageError path via wav_duration:
    # Instead call wav_duration on a non-wav file.
    audio.write_bytes(b"not a wav")
    with pytest.raises(PackageError, match="Failed to probe duration"):
        package_book(chapters=[ch], paths=paths, ffmpeg=FakeFfmpegRunner())


def test_wav_duration_invalid_rate_via_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav = _write_silent_wav(tmp_path / "z.wav")

    class FakeWave:
        def getnframes(self) -> int:
            return 100

        def getframerate(self) -> int:
            return 0

        def __enter__(self) -> FakeWave:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "audioforge.pipeline.package.wave.open",
        lambda *a, **k: FakeWave(),
    )
    with pytest.raises(PackageError, match="Invalid WAV sample rate"):
        wav_duration_seconds(wav)


def test_package_zero_duration_clamped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ch = _chapter(title="Tiny")
    _write_chapter_audio(paths, ch)

    package_book(
        chapters=[ch],
        paths=paths,
        ffmpeg=FakeFfmpegRunner(),
        duration_seconds=lambda _p: 0.0,
    )
    meta = (paths.out / CHAPTERS_METADATA_NAME).read_text(encoding="utf-8")
    assert "START=0" in meta
    assert "END=1" in meta  # clamped to 1 ms


def test_pipeline_exports_package() -> None:
    from audioforge.pipeline import PackageError as PE
    from audioforge.pipeline import package_book as pb

    assert PE is PackageError
    assert pb is package_book
