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


def test_engine_writes_cues_sidecar_from_result_tokens(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[object]:
            del text, voice
            tokens = [
                SimpleNamespace(text="Hello", start_ts=0.0, end_ts=0.25),
                SimpleNamespace(text="world", start_ts=0.25, end_ts=0.45),
                SimpleNamespace(text=".", start_ts=0.45, end_ts=0.5),
            ]
            # 0.5s at 24kHz = 12000 samples
            audio = [0.0] * 12_000
            yield SimpleNamespace(audio=audio, tokens=tokens)

    engine = engine_from_kpipeline(FakePipeline, sample_rate=24_000)
    out = tmp_path / "speech.wav"
    engine("Hello world", "af_heart", out)
    assert out.is_file()
    from audioforge.backends.kokoro_tts import cues_sidecar_path, load_cues_sidecar

    assert cues_sidecar_path(out).is_file()
    cues = load_cues_sidecar(out)
    assert cues is not None
    assert len(cues) == 2  # punctuation-only "." dropped
    assert cues[0].text == "Hello"
    assert cues[0].start_s == 0.0
    assert cues[1].text == "world"
    assert cues[1].end_s == 0.45


def test_load_cues_sidecar_missing(tmp_path: Path) -> None:
    from audioforge.backends.kokoro_tts import load_cues_sidecar

    assert load_cues_sidecar(tmp_path / "no.wav") is None


def test_load_cues_sidecar_empty_or_invalid(tmp_path: Path) -> None:
    from audioforge.backends.kokoro_tts import cues_sidecar_path, load_cues_sidecar

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    side = cues_sidecar_path(wav)
    side.write_text('{"cues": []}\n', encoding="utf-8")
    assert load_cues_sidecar(wav) is None
    side.write_text('"not-a-dict"\n', encoding="utf-8")
    assert load_cues_sidecar(wav) is None
    side.write_text("{not json", encoding="utf-8")
    assert load_cues_sidecar(wav) is None
    side.write_text(
        '{"cues": [{"start_s": 1.0, "end_s": 0.5, "text": "bad"}]}\n',
        encoding="utf-8",
    )
    assert load_cues_sidecar(wav) is None


def test_write_cues_sidecar_empty_removes_stale(tmp_path: Path) -> None:
    from audioforge.backends.kokoro_tts import (
        cues_sidecar_path,
        load_cues_sidecar,
        write_cues_sidecar,
    )
    from audioforge.models import TimedCue

    wav = tmp_path / "speech.wav"
    wav.write_bytes(b"x")
    write_cues_sidecar(wav, [TimedCue(start_s=0.0, end_s=1.0, text="OLD")])
    assert load_cues_sidecar(wav) is not None
    assert write_cues_sidecar(wav, []) is None
    assert not cues_sidecar_path(wav).is_file()
    assert load_cues_sidecar(wav) is None


def test_cues_skip_bad_tokens_and_zero_audio_offset(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[object]:
            del text, voice
            # No audio samples, but tokens with inverted times / empty text
            tokens = [
                SimpleNamespace(text="", start_ts=0.0, end_ts=0.1),
                SimpleNamespace(text="Hi", start_ts=None, end_ts=0.1),
                SimpleNamespace(text="Ok", start_ts=0.5, end_ts=0.4),  # inverted
            ]
            yield SimpleNamespace(audio=[], tokens=tokens)
            # second chunk with audio via output.audio
            tokens2 = [SimpleNamespace(text="There", start_ts=0.0, end_ts=0.2)]
            yield SimpleNamespace(
                audio=None,
                output=SimpleNamespace(audio=[0.0, 0.1]),
                tokens=tokens2,
            )

    engine = engine_from_kpipeline(FakePipeline, sample_rate=2)
    out = tmp_path / "x.wav"
    engine("t", "v", out)
    from audioforge.backends.kokoro_tts import load_cues_sidecar

    cues = load_cues_sidecar(out)
    assert cues is not None
    # skip empty/missing; fix inverted; second chunk offsets
    texts = [c.text for c in cues]
    assert "Ok" in texts
    assert "There" in texts


def test_extract_numpy_like_and_detach(tmp_path: Path) -> None:
    from audioforge.backends.kokoro_tts import _as_float_iterable, _extract_audio_chunk

    class Detachable:
        def detach(self) -> Detachable:
            return self

        def cpu(self) -> Detachable:
            return self

        def float(self) -> Detachable:
            return self

        def reshape(self, *a: object) -> Detachable:
            return self

        def tolist(self) -> list[float]:
            return [0.25]

    assert list(_as_float_iterable(None)) == []
    assert list(_as_float_iterable(Detachable())) == [0.25]

    class WithNumpy:
        def numpy(self) -> list[float]:
            return [0.5, 0.6]

    assert list(_as_float_iterable(WithNumpy())) == [0.5, 0.6]

    class OutWrap:
        def __init__(self) -> None:
            self.audio = [0.1]
            self.output = None

    # audio attr preferred
    assert list(_extract_audio_chunk(OutWrap())) == [0.1]

    from types import SimpleNamespace

    o = SimpleNamespace(audio=None, output=SimpleNamespace(audio=[0.2, 0.3]))
    assert list(_extract_audio_chunk(o)) == [0.2, 0.3]


def test_extract_audio_skips_string_audio_attr() -> None:
    from types import SimpleNamespace

    from audioforge.backends.kokoro_tts import _extract_audio_chunk

    item2 = SimpleNamespace(audio="x", output=SimpleNamespace(audio=[0.0]))
    assert list(_extract_audio_chunk(item2)) == [0.0]


def test_as_float_iterable_generator() -> None:
    from audioforge.backends.kokoro_tts import _as_float_iterable

    def gen() -> Iterator[float]:
        yield 0.1
        yield 0.2

    assert list(_as_float_iterable(gen())) == [0.1, 0.2]


def test_offset_uses_token_end_when_no_audio_samples(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class FakePipeline:
        def __init__(self, lang_code: str = "a") -> None:
            del lang_code

        def __call__(self, text: str, voice: str = "") -> Iterator[object]:
            del text, voice
            yield SimpleNamespace(
                audio=[],
                tokens=[SimpleNamespace(text="A", start_ts=0.0, end_ts=0.3)],
            )
            yield SimpleNamespace(
                audio=[0.0] * 100,
                tokens=[SimpleNamespace(text="B", start_ts=0.0, end_ts=0.1)],
            )

    engine = engine_from_kpipeline(FakePipeline, sample_rate=100)
    out = tmp_path / "o.wav"
    engine("t", "v", out)
    from audioforge.backends.kokoro_tts import load_cues_sidecar

    cues = load_cues_sidecar(out)
    assert cues is not None
    assert cues[0].text == "A"
    assert cues[1].start_s == 0.3  # offset advanced from previous cue end


def test_as_float_iterable_numpy_list_and_reshape(tmp_path: Path) -> None:
    from audioforge.backends.kokoro_tts import _as_float_iterable

    class N:
        def numpy(self) -> list[float]:
            return [1.0, 2.0]

    assert list(_as_float_iterable(N())) == [1.0, 2.0]

    class Arr:
        def reshape(self, *a: object) -> Arr:
            return self

        def tolist(self) -> list[float]:
            return [3.0]

    class N2:
        def numpy(self) -> Arr:
            return Arr()

    assert list(_as_float_iterable(N2())) == [3.0]
