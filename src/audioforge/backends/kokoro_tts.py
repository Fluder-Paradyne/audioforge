"""Kokoro TTS backend with injectable engine for tests.

Heavy ML packages (``kokoro``, torch, etc.) are **not** hard dependencies of
AudioForge. Install the optional ``tts`` extra when using real Kokoro synthesis,
or inject a custom ``engine`` (tests use this path exclusively).
"""

from __future__ import annotations

import importlib
import struct
import wave
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable


class KokoroNotInstalledError(ImportError):
    """Raised when Kokoro is not installed and no engine was injected.

    Install optional TTS dependencies (see the ``tts`` optional-dependency
    group in ``pyproject.toml``) or pass an ``engine`` to
    :class:`KokoroTtsBackend` for tests / custom synthesizers.
    """


@runtime_checkable
class KokoroEngine(Protocol):
    """Callable that synthesizes *text* with *voice* into *out_path*."""

    def __call__(self, text: str, voice: str, out_path: Path) -> None:
        """Write audio for *text* to *out_path* (side-effecting)."""
        ...


def write_pcm16_mono_wav(
    out_path: Path,
    samples: Iterable[float],
    *,
    sample_rate: int = 24_000,
) -> None:
    """Write mono 16-bit PCM WAV from float samples in roughly ``[-1, 1]``."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for raw in samples:
        clamped = max(-1.0, min(1.0, float(raw)))
        frames.extend(struct.pack("<h", int(clamped * 32767.0)))
    if not frames:
        frames.extend(struct.pack("<h", 0))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def engine_from_kpipeline(
    pipeline_cls: Any,
    *,
    sample_rate: int = 24_000,
    lang_code: str = "a",
) -> KokoroEngine:
    """Build a :class:`KokoroEngine` from a ``KPipeline``-like class.

    The class is constructed as ``pipeline_cls(lang_code=lang_code)``. Calling
    the instance as ``pipeline(text, voice=voice)`` must yield either audio
    sample sequences or tuples whose third element is the sample sequence
    (hexgrad/kokoro style: ``(graphemes, phonemes, audio)``).
    """

    def synthesize_to_file(text: str, voice: str, out_path: Path) -> None:
        pipeline = pipeline_cls(lang_code=lang_code)
        samples: list[float] = []
        for item in pipeline(text, voice=voice):
            audio = _extract_audio_chunk(item)
            samples.extend(float(x) for x in audio)
        write_pcm16_mono_wav(out_path, samples, sample_rate=sample_rate)

    return synthesize_to_file


def _as_float_iterable(value: object) -> Iterable[float]:
    """Coerce list/tuple/numpy-like/iterable objects to float samples."""
    if isinstance(value, (list, tuple)):
        return cast(Iterable[float], value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return cast(Iterable[float], tolist())
    return cast(Iterable[float], list(cast(Iterator[object], value)))


def _extract_audio_chunk(item: object) -> Iterable[float]:
    if isinstance(item, tuple) and len(item) >= 3:
        return _as_float_iterable(item[2])
    return _as_float_iterable(item)


def resolve_default_kokoro_engine() -> KokoroEngine:
    """Import ``kokoro.KPipeline`` and wrap it, or raise not-installed.

    Raises :class:`KokoroNotInstalledError` when the package is missing.
    Production entry point for resolving a real engine. Unit tests should inject
    engines instead of relying on this import.
    """
    try:
        kokoro_mod = importlib.import_module("kokoro")
    except ImportError as exc:
        raise KokoroNotInstalledError(
            "Kokoro TTS is not installed. Install the optional 'tts' "
            "dependency group (e.g. `uv sync --extra tts`) or inject a "
            "custom engine into KokoroTtsBackend(engine=...)."
        ) from exc
    pipeline_cls = getattr(kokoro_mod, "KPipeline", None)
    if pipeline_cls is None:
        raise KokoroNotInstalledError(
            "Package 'kokoro' is installed but does not expose KPipeline. "
            "Inject a custom engine into KokoroTtsBackend(engine=...)."
        )
    return engine_from_kpipeline(pipeline_cls)


class KokoroTtsBackend:
    """Synthesize speech via Kokoro, or an injected :class:`KokoroEngine`.

    Parameters
    ----------
    engine:
        Optional callable ``(text, voice, out_path) -> None``. When omitted,
        a default engine is resolved (see *resolve_eagerly*).
    resolve_eagerly:
        If true (default), resolve the default engine during ``__init__`` so
        missing installs fail fast. When false, resolution is deferred until
        the first :meth:`synthesize` call.
    """

    def __init__(
        self,
        engine: Callable[[str, str, Path], None] | KokoroEngine | None = None,
        *,
        resolve_eagerly: bool = True,
    ) -> None:
        self._engine: Callable[[str, str, Path], None] | KokoroEngine | None = engine
        if engine is None and resolve_eagerly:
            self._engine = resolve_default_kokoro_engine()

    def synthesize(self, text: str, *, voice: str, out_path: Path) -> Path:
        """Write audio for *text* to *out_path* and return that path."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = self._engine
        if engine is None:
            engine = resolve_default_kokoro_engine()
            self._engine = engine
        engine(text, voice, path)
        return path
