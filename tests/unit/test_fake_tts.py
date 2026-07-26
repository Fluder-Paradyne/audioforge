"""Tests for FakeTtsBackend WAV writing."""

from __future__ import annotations

import wave
from pathlib import Path

from audioforge.backends.fake import FakeTtsBackend


def test_fake_tts_writes_readable_mono_pcm_wav(tmp_path: Path) -> None:
    backend = FakeTtsBackend()
    out = tmp_path / "nested" / "out.wav"
    result = backend.synthesize("hello world", voice="af_heart", out_path=out)

    assert result == out
    assert out.is_file()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24_000
        assert wf.getnframes() > 0
        frames = wf.readframes(wf.getnframes())
        assert len(frames) == wf.getnframes() * 2


def test_fake_tts_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c.wav"
    FakeTtsBackend().synthesize("x", voice="v", out_path=out)
    assert out.is_file()


def test_fake_tts_longer_text_produces_more_frames(tmp_path: Path) -> None:
    backend = FakeTtsBackend()
    short = tmp_path / "short.wav"
    long = tmp_path / "long.wav"
    backend.synthesize("hi", voice="v", out_path=short)
    backend.synthesize("hi" * 200, voice="v", out_path=long)
    with wave.open(str(short), "rb") as a, wave.open(str(long), "rb") as b:
        assert b.getnframes() > a.getnframes()


def test_fake_tts_empty_text_still_valid_wav(tmp_path: Path) -> None:
    out = tmp_path / "empty.wav"
    FakeTtsBackend().synthesize("", voice="v", out_path=out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() >= 1


def test_fake_tts_custom_sample_rate(tmp_path: Path) -> None:
    out = tmp_path / "sr.wav"
    FakeTtsBackend(sample_rate=16_000).synthesize("abc", voice="v", out_path=out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 16_000
