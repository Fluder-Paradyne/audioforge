"""Tests for KokoroTtsBackend with injectable engines."""

from __future__ import annotations

import types
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from audioforge.backends.kokoro_tts import (
    KokoroNotInstalledError,
    KokoroTtsBackend,
    engine_from_kpipeline,
    resolve_default_kokoro_engine,
    write_pcm16_mono_wav,
)


def test_kokoro_with_mock_engine_writes_and_returns_path(tmp_path: Path) -> None:
    calls: list[tuple[str, str, Path]] = []

    def engine(text: str, voice: str, out_path: Path) -> None:
        calls.append((text, voice, out_path))
        out_path.write_bytes(b"RIFF....")  # opaque payload is fine for this test

    backend = KokoroTtsBackend(engine=engine)
    out = tmp_path / "nested" / "speech.wav"
    result = backend.synthesize("Hello", voice="af_heart", out_path=out)

    assert result == out
    assert out.is_file()
    assert calls == [("Hello", "af_heart", out)]


def test_kokoro_not_installed_eager_init(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.importlib.import_module",
        boom,
    )
    with pytest.raises(KokoroNotInstalledError, match="not installed"):
        KokoroTtsBackend()


def test_kokoro_not_installed_lazy_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.resolve_default_kokoro_engine",
        MagicMock(side_effect=KokoroNotInstalledError("missing")),
    )
    backend = KokoroTtsBackend(resolve_eagerly=False)
    with pytest.raises(KokoroNotInstalledError, match="missing"):
        backend.synthesize("x", voice="v", out_path=tmp_path / "a.wav")


def test_kokoro_lazy_resolves_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def engine(text: str, voice: str, out_path: Path) -> None:
        del voice
        out_path.write_bytes(b"ok")
        calls.append(text)

    resolve = MagicMock(return_value=engine)
    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.resolve_default_kokoro_engine",
        resolve,
    )
    backend = KokoroTtsBackend(resolve_eagerly=False)
    backend.synthesize("one", voice="v", out_path=tmp_path / "1.wav")
    backend.synthesize("two", voice="v", out_path=tmp_path / "2.wav")
    assert resolve.call_count == 1
    assert calls == ["one", "two"]


def test_write_pcm16_mono_wav_empty_and_clamped(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    write_pcm16_mono_wav(empty, [], sample_rate=8000)
    with wave.open(str(empty), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 1

    loud = tmp_path / "loud.wav"
    write_pcm16_mono_wav(loud, [2.0, -2.0, 0.5], sample_rate=16_000)
    with wave.open(str(loud), "rb") as wf:
        assert wf.getnframes() == 3


def test_engine_from_kpipeline_tuple_chunks(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            self.lang_code = lang_code

        def __call__(
            self, text: str, voice: str = ""
        ) -> Iterator[tuple[str, str, list[float]]]:
            del voice
            yield ("g", "p", [0.0, 0.25])
            yield ("g2", "p2", [-0.25, 0.0])

    engine = engine_from_kpipeline(FakePipeline, sample_rate=12_000)
    out = tmp_path / "out.wav"
    engine("hello", "af_heart", out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 12_000
        assert wf.getnframes() == 4


def test_engine_from_kpipeline_raw_list_chunks(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[list[float]]:
            del text, voice
            yield [0.1, 0.2]

    engine = engine_from_kpipeline(FakePipeline)
    out = tmp_path / "raw.wav"
    engine("t", "v", out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() == 2


def test_engine_from_kpipeline_tolist_chunk(tmp_path: Path) -> None:
    class Arrayish:
        def tolist(self) -> list[float]:
            return [0.0, 1.0]

    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[Arrayish]:
            del text, voice
            yield Arrayish()

    engine = engine_from_kpipeline(FakePipeline)
    out = tmp_path / "arr.wav"
    engine("t", "v", out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() == 2


def test_engine_from_kpipeline_tuple_with_tolist_audio(tmp_path: Path) -> None:
    class Arrayish:
        def tolist(self) -> list[float]:
            return [0.5]

    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(
            self, text: str, voice: str = ""
        ) -> Iterator[tuple[str, str, Arrayish]]:
            del text, voice
            yield ("g", "p", Arrayish())

    engine = engine_from_kpipeline(FakePipeline)
    out = tmp_path / "t.wav"
    engine("t", "v", out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() == 1


def test_engine_from_kpipeline_empty_yield_writes_one_sample(tmp_path: Path) -> None:
    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[object]:
            del text, voice
            return iter(())

    engine = engine_from_kpipeline(FakePipeline)
    out = tmp_path / "empty.wav"
    engine("", "v", out)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() == 1


def test_resolve_default_with_fake_kokoro_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            self.lang_code = lang_code

        def __call__(
            self, text: str, voice: str = ""
        ) -> Iterator[tuple[str, str, list[float]]]:
            del text, voice
            yield ("g", "p", [0.0])

    mod = types.ModuleType("kokoro")
    mod.KPipeline = FakePipeline  # type: ignore[attr-defined]

    def import_module(name: str) -> types.ModuleType:
        if name != "kokoro":
            raise ImportError(name)
        return mod

    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.importlib.import_module",
        import_module,
    )

    engine = resolve_default_kokoro_engine()
    out = tmp_path / "from_mod.wav"
    engine("hi", "af_heart", out)
    assert out.is_file()


def test_resolve_default_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.importlib.import_module",
        MagicMock(side_effect=ImportError("missing")),
    )
    with pytest.raises(KokoroNotInstalledError) as exc_info:
        resolve_default_kokoro_engine()
    assert isinstance(exc_info.value, ImportError)


def test_resolve_default_missing_kpipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = types.ModuleType("kokoro")
    monkeypatch.setattr(
        "audioforge.backends.kokoro_tts.importlib.import_module",
        MagicMock(return_value=mod),
    )
    with pytest.raises(KokoroNotInstalledError, match="KPipeline"):
        resolve_default_kokoro_engine()


def test_extract_audio_iterable_fallback() -> None:
    from audioforge.backends.kokoro_tts import _extract_audio_chunk

    class IterableOnly:
        def __iter__(self) -> Iterator[float]:
            yield 0.1
            yield 0.2

    assert list(_extract_audio_chunk(IterableOnly())) == [0.1, 0.2]
    assert list(_extract_audio_chunk((0, 1, IterableOnly()))) == [0.1, 0.2]
