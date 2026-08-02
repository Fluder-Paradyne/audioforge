"""Kokoro TTS backend with injectable engine for tests.

Heavy ML packages (``kokoro``, torch, etc.) are **not** hard dependencies of
AudioForge. Install the optional ``tts`` extra when using real Kokoro synthesis,
or inject a custom ``engine`` (tests use this path exclusively).

When the real KPipeline is used, token timestamps (``start_ts`` / ``end_ts``)
are written next to the WAV as ``*.cues.json`` for the align stage.
"""

from __future__ import annotations

import importlib
import json
import struct
import wave
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from audioforge.models import TimedCue

# Sidecar written beside chapter audio when Kokoro emits token timestamps.
CUES_JSON_SUFFIX = ".cues.json"


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


def cues_sidecar_path(audio_path: Path) -> Path:
    """Return path for Kokoro token-timing sidecar next to *audio_path*."""
    return Path(audio_path).with_suffix(CUES_JSON_SUFFIX)


def write_cues_sidecar(audio_path: Path, cues: list[TimedCue]) -> Path | None:
    """Write cues JSON next to *audio_path*; return path or None if empty."""
    if not cues:
        return None
    path = cues_sidecar_path(audio_path)
    payload = {"cues": [c.model_dump() for c in cues]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_cues_sidecar(audio_path: Path) -> list[TimedCue] | None:
    """Load cues from sidecar if present; else None."""
    path = cues_sidecar_path(audio_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("cues") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    return [TimedCue.model_validate(item) for item in raw]


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
    the instance as ``pipeline(text, voice=voice)`` must yield either:

    * audio sample sequences,
    * tuples whose third element is audio (legacy),
    * or Result-like objects with ``.audio`` / ``.tokens`` (hexgrad/kokoro).

    When Result tokens include ``start_ts`` / ``end_ts``, a ``*.cues.json``
    sidecar is written next to the output WAV (chapter-relative seconds).
    """

    def synthesize_to_file(text: str, voice: str, out_path: Path) -> None:
        pipeline = pipeline_cls(lang_code=lang_code)
        samples: list[float] = []
        cues: list[TimedCue] = []
        offset_s = 0.0
        for item in pipeline(text, voice=voice):
            audio_list = list(_extract_audio_chunk(item))
            samples.extend(float(x) for x in audio_list)
            chunk_dur = len(audio_list) / float(sample_rate) if audio_list else 0.0
            cues.extend(_cues_from_pipeline_item(item, offset_s=offset_s))
            # Prefer audio length for offset so concatenation stays consistent.
            if chunk_dur > 0:
                offset_s += chunk_dur
            elif cues:
                offset_s = max(offset_s, cues[-1].end_s)
        write_pcm16_mono_wav(out_path, samples, sample_rate=sample_rate)
        write_cues_sidecar(out_path, cues)

    return synthesize_to_file


def _cues_from_pipeline_item(item: object, *, offset_s: float) -> list[TimedCue]:
    """Extract TimedCues from a Kokoro Result-like object, if present.

    Skips tokens without timestamps and punctuation-only tokens (e.g. ``.``
    ``,``) so subtitle tracks stay word-level rather than noise-filled.
    """
    tokens = getattr(item, "tokens", None)
    if not tokens:
        return []
    cues: list[TimedCue] = []
    for token in tokens:
        start = getattr(token, "start_ts", None)
        end = getattr(token, "end_ts", None)
        raw_text = getattr(token, "text", None)
        if start is None or end is None:
            continue
        text = str(raw_text).strip() if raw_text is not None else ""
        if not text or not any(ch.isalnum() for ch in text):
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            end_f = start_f + 0.02
        cues.append(
            TimedCue(
                start_s=offset_s + start_f,
                end_s=offset_s + end_f,
                text=text,
            )
        )
    return cues


def _as_float_iterable(value: object) -> Iterable[float]:
    """Coerce list/tuple/numpy-like/iterable objects to float samples."""
    if value is None:
        return []
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().float().reshape(-1)
    if isinstance(value, (list, tuple)):
        return cast(Iterable[float], value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return cast(Iterable[float], tolist())
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        arr = numpy_fn()
        reshape = getattr(arr, "reshape", None)
        if callable(reshape):
            arr = reshape(-1)
        tolist2 = getattr(arr, "tolist", None)
        if callable(tolist2):
            return cast(Iterable[float], tolist2())
        return cast(Iterable[float], list(arr))
    return cast(Iterable[float], list(cast(Iterator[object], value)))


def _extract_audio_chunk(item: object) -> Iterable[float]:
    """Return float samples from a pipeline yield (Result, tuple, or list)."""
    audio_attr = getattr(item, "audio", None)
    if audio_attr is not None and not isinstance(audio_attr, (str, bytes)):
        return _as_float_iterable(audio_attr)
    output = getattr(item, "output", None)
    if output is not None:
        out_audio = getattr(output, "audio", None)
        if out_audio is not None:
            return _as_float_iterable(out_audio)
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
