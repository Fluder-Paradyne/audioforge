"""Tests for alignment backend, align stage, WebVTT, and package subtitles."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audioforge.backends.alignment import (
    AlignmentError,
    FakeAlignmentBackend,
    alignment_filename,
    load_chapter_alignment,
    proportional_cues,
    save_chapter_alignment,
    split_phrases,
    wav_duration_seconds,
)
from audioforge.backends.ffmpeg import FakeFfmpegRunner
from audioforge.io.paths import JobPaths
from audioforge.models import (
    BuildOptions,
    ChapterAlignment,
    ChapterRef,
    TimedCue,
)
from audioforge.pipeline.align import AlignStageError, align_chapters
from audioforge.pipeline.package import package_book
from audioforge.pipeline.subtitles import (
    SubtitleError,
    build_book_vtt,
    format_vtt_timestamp,
    offset_cues,
)
from audioforge.pipeline.tts import audio_filename, prepared_filename


def _write_silent_wav(path: Path, *, seconds: float = 1.0, rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * nframes)


def test_split_phrases_and_proportional_cues() -> None:
    phrases = split_phrases("Hello world. Next sentence!")
    assert len(phrases) == 2
    cues = proportional_cues("Hello world. Next sentence!", 2.0)
    assert len(cues) == 2
    assert cues[0].start_s == 0.0
    assert cues[-1].end_s == 2.0
    assert all(c.end_s > c.start_s for c in cues)


def test_proportional_empty_text() -> None:
    cues = proportional_cues("   ", 1.0)
    assert len(cues) == 1
    assert cues[0].text == "…"


def test_fake_align_backend(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _write_silent_wav(wav, seconds=1.5)
    backend = FakeAlignmentBackend()
    cues = backend.align(
        wav,
        "One. Two.",
        options=BuildOptions(source="."),
    )
    assert len(cues) >= 1
    assert cues[-1].end_s == pytest.approx(1.5, abs=0.05)


def test_fake_align_missing_audio(tmp_path: Path) -> None:
    backend = FakeAlignmentBackend()
    with pytest.raises(AlignmentError, match="missing"):
        backend.align(tmp_path / "no.wav", "hi", options=BuildOptions(source="."))


def test_alignment_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / alignment_filename(1, "one")
    alignment = ChapterAlignment(
        chapter_index=1,
        cues=[TimedCue(start_s=0.0, end_s=1.0, text="Hi")],
    )
    save_chapter_alignment(path, alignment)
    loaded = load_chapter_alignment(path)
    assert loaded.chapter_index == 1
    assert loaded.cues[0].text == "Hi"


def test_format_vtt_timestamp() -> None:
    assert format_vtt_timestamp(0) == "00:00:00.000"
    assert format_vtt_timestamp(3661.5) == "01:01:01.500"


def test_offset_cues_clamp() -> None:
    cues = [TimedCue(start_s=0.0, end_s=5.0, text="long")]
    out = offset_cues(cues, chapter_start_s=10.0, chapter_end_s=12.0)
    assert out[0].start_s == 10.0
    assert out[0].end_s == 12.0


def test_build_book_vtt_basic() -> None:
    chapters = [
        ChapterRef(index=1, title="One", source_path=Path("a.md"), slug="one"),
        ChapterRef(index=2, title="Two", source_path=Path("b.md"), slug="two"),
    ]
    alignments = [
        ChapterAlignment(
            chapter_index=1,
            cues=[TimedCue(start_s=0.0, end_s=1.0, text="First")],
        ),
        ChapterAlignment(
            chapter_index=2,
            cues=[TimedCue(start_s=0.0, end_s=1.0, text="Second")],
        ),
    ]
    vtt = build_book_vtt(chapters, alignments, [1.0, 1.0])
    assert vtt.startswith("WEBVTT")
    assert "First" in vtt
    assert "Second" in vtt
    assert "00:00:00.000 -->" in vtt


def test_build_book_vtt_missing_alignment() -> None:
    chapters = [
        ChapterRef(index=1, title="One", source_path=Path("a.md"), slug="one"),
    ]
    with pytest.raises(SubtitleError, match="same length"):
        build_book_vtt(chapters, [], [1.0])
    with pytest.raises(SubtitleError, match="Missing alignment"):
        build_book_vtt(
            chapters,
            [
                ChapterAlignment(
                    chapter_index=99,
                    cues=[TimedCue(start_s=0.0, end_s=1.0, text="x")],
                )
            ],
            [1.0],
        )


def test_align_stage_and_package_with_subtitles(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "job").ensure()
    chapters = [
        ChapterRef(
            index=1,
            title="One",
            source_path=paths.source / "0001-one.md",
            slug="one",
        ),
    ]
    prepared = paths.prepared / prepared_filename(chapters[0])
    prepared.write_text("Hello world. Second phrase.", encoding="utf-8")
    audio = paths.audio / audio_filename(chapters[0])
    _write_silent_wav(audio, seconds=2.0)

    progress = align_chapters(
        chapters=chapters,
        paths=paths,
        options=BuildOptions(source=".", resume=False),
        backend=FakeAlignmentBackend(),
    )
    assert progress[0].align_done is True
    assert (paths.aligned / alignment_filename(1, "one")).is_file()

    # resume skip
    progress2 = align_chapters(
        chapters=chapters,
        paths=paths,
        options=BuildOptions(source=".", resume=True, force=False),
        backend=FakeAlignmentBackend(),
        progress=progress,
    )
    assert progress2[0].align_done is True

    alignment = load_chapter_alignment(paths.aligned / alignment_filename(1, "one"))
    ffmpeg = FakeFfmpegRunner()
    manifest = package_book(
        chapters=chapters,
        paths=paths,
        ffmpeg=ffmpeg,
        book_slug="book",
        alignments=[alignment],
        include_subtitles=True,
    )
    assert manifest.m4b_path is not None
    assert manifest.subtitles_vtt is not None
    assert manifest.subtitles_vtt.is_file()
    cmd = ffmpeg.commands[0]
    assert "mov_text" in cmd
    assert any("subtitles.vtt" in str(a) for a in cmd)


def test_package_without_subtitles_no_mov_text(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "job2").ensure()
    chapters = [
        ChapterRef(
            index=1,
            title="One",
            source_path=paths.source / "0001-one.md",
            slug="one",
        ),
    ]
    audio = paths.audio / audio_filename(chapters[0])
    _write_silent_wav(audio, seconds=0.5)
    ffmpeg = FakeFfmpegRunner()
    manifest = package_book(
        chapters=chapters,
        paths=paths,
        ffmpeg=ffmpeg,
        book_slug="book",
        include_subtitles=False,
    )
    assert manifest.subtitles_vtt is None
    assert "mov_text" not in ffmpeg.commands[0]


def test_align_stage_fail_missing_prepared(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "bad").ensure()
    chapters = [
        ChapterRef(index=1, title="One", source_path=Path("x.md"), slug="one"),
    ]
    _write_silent_wav(paths.audio / audio_filename(chapters[0]), seconds=0.5)
    with pytest.raises(AlignStageError, match="Prepared text missing"):
        align_chapters(
            chapters=chapters,
            paths=paths,
            options=BuildOptions(source=".", resume=False),
            backend=FakeAlignmentBackend(),
        )


def test_wav_duration_helper(tmp_path: Path) -> None:
    wav = tmp_path / "t.wav"
    _write_silent_wav(wav, seconds=0.25, rate=8000)
    assert wav_duration_seconds(wav) == pytest.approx(0.25, abs=0.01)


def test_package_requires_alignments_when_subtitles(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "j").ensure()
    chapters = [
        ChapterRef(index=1, title="T", source_path=Path("a.md"), slug="one"),
    ]
    _write_silent_wav(paths.audio / audio_filename(chapters[0]), seconds=0.5)
    with pytest.raises(Exception, match="alignments"):
        package_book(
            chapters=chapters,
            paths=paths,
            ffmpeg=FakeFfmpegRunner(),
            include_subtitles=True,
            alignments=None,
        )


def test_offset_invalid_span() -> None:
    with pytest.raises(SubtitleError, match="Invalid chapter span"):
        offset_cues(
            [TimedCue(start_s=0.0, end_s=1.0, text="x")],
            chapter_start_s=5.0,
            chapter_end_s=5.0,
        )


def test_format_vtt_negative_clamped() -> None:
    assert format_vtt_timestamp(-1.0) == "00:00:00.000"


def test_align_missing_audio(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "bad2").ensure()
    chapters = [
        ChapterRef(index=1, title="One", source_path=Path("x.md"), slug="one"),
    ]
    (paths.prepared / prepared_filename(chapters[0])).write_text("hi", encoding="utf-8")
    with pytest.raises(AlignStageError, match="Chapter audio missing"):
        align_chapters(
            chapters=chapters,
            paths=paths,
            options=BuildOptions(source=".", resume=False),
            backend=FakeAlignmentBackend(),
        )


def test_align_backend_raises(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "bad3").ensure()
    chapters = [
        ChapterRef(index=1, title="One", source_path=Path("x.md"), slug="one"),
    ]
    (paths.prepared / prepared_filename(chapters[0])).write_text("hi", encoding="utf-8")
    _write_silent_wav(paths.audio / audio_filename(chapters[0]), seconds=0.5)

    class Boom:
        def align(
            self, audio_path: Path, text: str, *, options: BuildOptions
        ) -> list[TimedCue]:
            raise AlignmentError("no cues")

    with pytest.raises(AlignStageError, match="no cues"):
        align_chapters(
            chapters=chapters,
            paths=paths,
            options=BuildOptions(source=".", resume=False),
            backend=Boom(),
        )


def test_pipeline_skip_subtitles(tmp_path: Path) -> None:
    from audioforge.backends.fake import FakeTtsBackend
    from audioforge.backends.rules_prep import RulesTextPrep
    from audioforge.pipeline.orchestrator import run_pipeline
    from audioforge.settings import AppSettings

    fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"
    settings = AppSettings(work_dir=tmp_path / "work")
    options = BuildOptions(
        source=str(fixtures),
        job_id="nosub",
        skip_prep=True,
        subtitles=False,
    )
    state = run_pipeline(
        options,
        settings,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        aligner=None,
    )
    assert state.status.value == "completed"
    assert state.artifacts is not None
    assert state.artifacts.subtitles_vtt is None


def test_run_pipeline_requires_aligner_when_subs(tmp_path: Path) -> None:
    from audioforge.backends.fake import FakeTtsBackend
    from audioforge.backends.rules_prep import RulesTextPrep
    from audioforge.pipeline.orchestrator import run_pipeline
    from audioforge.settings import AppSettings

    fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"
    settings = AppSettings(work_dir=tmp_path / "work")
    options = BuildOptions(source=str(fixtures), job_id="need-al", skip_prep=True)
    with pytest.raises(ValueError, match="aligner backend is required"):
        run_pipeline(
            options,
            settings,
            prep=RulesTextPrep(),
            tts=FakeTtsBackend(),
            ffmpeg=FakeFfmpegRunner(),
            aligner=None,
        )


def test_wav_duration_invalid_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "z.wav"
    _write_silent_wav(path, seconds=0.1)

    class FakeWave:
        def getnframes(self) -> int:
            return 100

        def getframerate(self) -> int:
            return 0

        def __enter__(self) -> FakeWave:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    monkeypatch.setattr(
        "audioforge.backends.alignment.wave.open",
        lambda *a, **k: FakeWave(),
    )
    with pytest.raises(AlignmentError, match="sample rate"):
        wav_duration_seconds(path)


def test_load_alignments_missing(tmp_path: Path) -> None:
    from audioforge.pipeline.align import load_alignments_for_chapters

    paths = JobPaths.for_job(tmp_path / "work", "m").ensure()
    chapters = [
        ChapterRef(index=1, title="T", source_path=Path("a.md"), slug="one"),
    ]
    with pytest.raises(AlignStageError, match="Alignment missing"):
        load_alignments_for_chapters(chapters, paths)


def test_package_subtitle_build_error(tmp_path: Path) -> None:
    paths = JobPaths.for_job(tmp_path / "work", "sb").ensure()
    chapters = [
        ChapterRef(index=1, title="T", source_path=Path("a.md"), slug="one"),
    ]
    _write_silent_wav(paths.audio / audio_filename(chapters[0]), seconds=0.5)
    # Wrong chapter_index forces build_book_vtt Missing alignment
    bad = ChapterAlignment(
        chapter_index=99,
        cues=[TimedCue(start_s=0.0, end_s=0.5, text="x")],
    )
    with pytest.raises(Exception, match="Failed to build subtitles"):
        package_book(
            chapters=chapters,
            paths=paths,
            ffmpeg=FakeFfmpegRunner(),
            include_subtitles=True,
            alignments=[bad],
        )


def test_run_package_without_subtitles(tmp_path: Path) -> None:
    from audioforge.backends.fake import FakeTtsBackend
    from audioforge.backends.rules_prep import RulesTextPrep
    from audioforge.models import JobStatus
    from audioforge.pipeline.orchestrator import (
        run_package,
        run_prepare,
        run_synthesize,
    )
    from audioforge.settings import AppSettings

    fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"
    settings = AppSettings(work_dir=tmp_path / "work")
    options = BuildOptions(
        source=str(fixtures),
        job_id="pkg-nosub",
        skip_prep=True,
        subtitles=False,
    )
    run_prepare(options, settings, prep=RulesTextPrep())
    run_synthesize(settings, job_or_path="pkg-nosub", tts=FakeTtsBackend())
    state = run_package(
        settings,
        job_or_path="pkg-nosub",
        ffmpeg=FakeFfmpegRunner(),
        aligner=None,
    )
    assert state.status == JobStatus.COMPLETED
    assert state.artifacts is not None
    assert state.artifacts.subtitles_vtt is None


def test_run_package_requires_aligner(tmp_path: Path) -> None:
    from audioforge.jobstore import save_job
    from audioforge.models import JobState, JobStatus
    from audioforge.pipeline.orchestrator import PipelineError, run_package
    from audioforge.settings import AppSettings

    settings = AppSettings(work_dir=tmp_path / "work")
    paths = JobPaths.for_job(settings.work_dir, "needal").ensure()
    state = JobState(
        job_id="needal",
        source=str(tmp_path),
        options=BuildOptions(source=str(tmp_path), subtitles=True),
        status=JobStatus.PENDING,
        chapters=[
            ChapterRef(index=1, title="T", source_path=Path("a.md"), slug="one"),
        ],
    )
    save_job(state, paths.job_json)
    with pytest.raises(PipelineError, match="aligner backend is required"):
        run_package(
            settings,
            job_or_path="needal",
            ffmpeg=FakeFfmpegRunner(),
            aligner=None,
        )


def test_api_fills_aligner_when_other_backends_injected(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from audioforge.api.app import create_app
    from audioforge.backends.fake import FakeTtsBackend
    from audioforge.backends.rules_prep import RulesTextPrep
    from audioforge.settings import AppSettings

    fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"
    app = create_app(
        AppSettings(work_dir=tmp_path / "work"),
        run_sync=True,
        prep=RulesTextPrep(),
        tts=FakeTtsBackend(),
        ffmpeg=FakeFfmpegRunner(),
        # aligner omitted → ProportionalAlignmentBackend
    )
    client = TestClient(app)
    response = client.post(
        "/jobs",
        json={"source": str(fixtures), "job_id": "fill-al", "skip_prep": True},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "completed"
