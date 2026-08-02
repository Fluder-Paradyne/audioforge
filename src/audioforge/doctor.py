"""Environment diagnostics for AudioForge (``audioforge doctor``).

Probes local dependencies without starting a full pipeline build. Required
checks (Python, work dir, FFmpeg, Kokoro unless fake TTS is allowed) gate the
process exit code; optional tools (Ollama, FictionReaper) report as warnings.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from audioforge import __version__
from audioforge.factory import _ALLOW_FAKE_TTS_ENV
from audioforge.settings import AppSettings

# Default timeouts for external probes (keep doctor snappy).
_FFMPEG_TIMEOUT_S = 10.0
_OLLAMA_TIMEOUT_S = 2.0


class CheckStatus(StrEnum):
    """Outcome of a single doctor check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class DoctorCheck(BaseModel):
    """One diagnostic line."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: CheckStatus
    message: str
    required: bool = False
    hint: str | None = None


class DoctorReport(BaseModel):
    """Full doctor run result."""

    model_config = ConfigDict(extra="forbid")

    version: str
    checks: list[DoctorCheck] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no required check failed."""
        return not any(c.status is CheckStatus.FAIL and c.required for c in self.checks)


def run_doctor(
    settings: AppSettings,
    *,
    fictionreaper_bin: str = "fictionreaper",
    check_python: Callable[[], DoctorCheck] | None = None,
    check_work_dir: Callable[[Path], DoctorCheck] | None = None,
    check_ffmpeg: Callable[[str], DoctorCheck] | None = None,
    check_kokoro: Callable[[], DoctorCheck] | None = None,
    check_ollama: Callable[[str, str], DoctorCheck] | None = None,
    check_fictionreaper: Callable[[str], DoctorCheck] | None = None,
) -> DoctorReport:
    """Run all diagnostics and return a structured report.

    Optional callables replace real probes (unit tests inject fakes).
    """
    py_fn = check_python if check_python is not None else _check_python
    work_fn = check_work_dir if check_work_dir is not None else _check_work_dir
    ffmpeg_fn = check_ffmpeg if check_ffmpeg is not None else _check_ffmpeg
    kokoro_fn = check_kokoro if check_kokoro is not None else _check_kokoro
    ollama_fn = check_ollama if check_ollama is not None else _check_ollama
    fr_fn = (
        check_fictionreaper if check_fictionreaper is not None else _check_fictionreaper
    )

    checks: list[DoctorCheck] = [
        DoctorCheck(
            name="audioforge",
            status=CheckStatus.OK,
            message=f"version {__version__}",
            required=False,
        ),
        py_fn(),
        work_fn(settings.work_dir),
        ffmpeg_fn(settings.ffmpeg_path),
        kokoro_fn(),
        ollama_fn(settings.ollama_base_url, settings.default_prep_model),
        fr_fn(fictionreaper_bin),
        DoctorCheck(
            name="defaults",
            status=CheckStatus.OK,
            message=(
                f"voice={settings.default_voice} "
                f"prep_model={settings.default_prep_model} "
                f"log={settings.log_level}/{settings.log_format} "
                f"api={settings.host}:{settings.port}"
            ),
            required=False,
        ),
    ]
    return DoctorReport(version=__version__, checks=checks)


def format_doctor_report(report: DoctorReport) -> str:
    """Render a human-readable multi-line summary."""
    lines = [f"audioforge doctor {report.version}", ""]
    for check in report.checks:
        mark = {
            CheckStatus.OK: "ok  ",
            CheckStatus.WARN: "warn",
            CheckStatus.FAIL: "FAIL",
        }[check.status]
        req = " (required)" if check.required else ""
        lines.append(f"[{mark}] {check.name}{req}: {check.message}")
        if check.hint:
            lines.append(f"       hint: {check.hint}")
    lines.append("")
    if report.ok:
        lines.append("Overall: ready (required checks passed).")
    else:
        lines.append("Overall: not ready — fix required FAILs above.")
    return "\n".join(lines) + "\n"


def _check_python() -> DoctorCheck:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 12):
        return DoctorCheck(
            name="python",
            status=CheckStatus.OK,
            message=f"{version} ({sys.executable})",
            required=True,
        )
    return DoctorCheck(
        name="python",
        status=CheckStatus.FAIL,
        message=f"{version} is below the required 3.12+",
        required=True,
        hint="Install Python ≥ 3.12 and reinstall audioforge.",
    )


def _check_work_dir(work_dir: Path) -> DoctorCheck:
    path = Path(work_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".audioforge_doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="work_dir",
            status=CheckStatus.FAIL,
            message=f"not writable: {path} ({exc})",
            required=True,
            hint="Set AUDIOFORGE_WORK_DIR to a writable directory.",
        )
    return DoctorCheck(
        name="work_dir",
        status=CheckStatus.OK,
        message=f"writable: {path.resolve()}",
        required=True,
    )


def _check_ffmpeg(ffmpeg_path: str) -> DoctorCheck:
    cmd = [ffmpeg_path, "-version"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_S,
        )
    except FileNotFoundError:
        return DoctorCheck(
            name="ffmpeg",
            status=CheckStatus.FAIL,
            message=f"binary not found: {ffmpeg_path}",
            required=True,
            hint="Install FFmpeg or set AUDIOFORGE_FFMPEG_PATH.",
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            name="ffmpeg",
            status=CheckStatus.FAIL,
            message=f"timed out running: {ffmpeg_path} -version",
            required=True,
        )
    except OSError as exc:
        return DoctorCheck(
            name="ffmpeg",
            status=CheckStatus.FAIL,
            message=f"could not run {ffmpeg_path}: {exc}",
            required=True,
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no output").strip()
        return DoctorCheck(
            name="ffmpeg",
            status=CheckStatus.FAIL,
            message=f"exited {result.returncode}: {detail[:200]}",
            required=True,
            hint="Install FFmpeg or set AUDIOFORGE_FFMPEG_PATH.",
        )

    first = (result.stdout or result.stderr or "ffmpeg ok").strip().splitlines()
    summary = first[0] if first else "ffmpeg ok"
    resolved = shutil.which(ffmpeg_path) or ffmpeg_path
    return DoctorCheck(
        name="ffmpeg",
        status=CheckStatus.OK,
        message=f"{summary} ({resolved})",
        required=True,
    )


def _check_kokoro() -> DoctorCheck:
    allow_fake = os.environ.get(_ALLOW_FAKE_TTS_ENV) == "1"
    try:
        kokoro_mod = importlib.import_module("kokoro")
    except ImportError:
        if allow_fake:
            return DoctorCheck(
                name="kokoro",
                status=CheckStatus.WARN,
                message=(
                    f"not installed; {_ALLOW_FAKE_TTS_ENV}=1 will use silent fake TTS"
                ),
                required=False,
                hint="Install with: uv sync --extra tts (or tool install [tts]).",
            )
        return DoctorCheck(
            name="kokoro",
            status=CheckStatus.FAIL,
            message="not installed (required for real TTS)",
            required=True,
            hint=(
                "Install optional extra `tts`, or set "
                f"{_ALLOW_FAKE_TTS_ENV}=1 for plumbing-only fake audio."
            ),
        )

    if getattr(kokoro_mod, "KPipeline", None) is None:
        return DoctorCheck(
            name="kokoro",
            status=CheckStatus.FAIL,
            message="package installed but KPipeline is missing",
            required=True,
            hint="Reinstall the tts extra or inject a custom engine.",
        )
    return DoctorCheck(
        name="kokoro",
        status=CheckStatus.OK,
        message="package importable (KPipeline present)",
        required=True,
    )


def _check_ollama(base_url: str, prep_model: str) -> DoctorCheck:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with httpx.Client(timeout=_OLLAMA_TIMEOUT_S) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.WARN,
            message=f"unreachable at {base_url} ({exc})",
            required=False,
            hint="Start Ollama or rely on rules prep / --skip-prep.",
        )

    if response.status_code != 200:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.WARN,
            message=f"HTTP {response.status_code} from {url}",
            required=False,
            hint="Start Ollama or rely on rules prep / --skip-prep.",
        )

    try:
        payload = response.json()
    except ValueError:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.WARN,
            message=f"reachable but non-JSON body from {url}",
            required=False,
        )

    models = payload.get("models") if isinstance(payload, dict) else None
    names: list[str] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str):
                    names.append(name)

    if not names:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.WARN,
            message=f"reachable at {base_url} (no models listed)",
            required=False,
            hint=f"Pull a model: ollama pull {prep_model}",
        )

    model_ok = any(
        n == prep_model or n.startswith(f"{prep_model}:") or n.startswith(prep_model)
        for n in names
    )
    # Also match when prep_model is "llama3.2:3b" and list has "llama3.2:3b"
    if not model_ok:
        base = prep_model.split(":")[0]
        model_ok = any(n == base or n.startswith(f"{base}:") for n in names)

    if model_ok:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.OK,
            message=f"reachable; model available for prep ({prep_model})",
            required=False,
        )
    preview = ", ".join(names[:5])
    more = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
    return DoctorCheck(
        name="ollama",
        status=CheckStatus.WARN,
        message=(
            f"reachable; default prep model {prep_model!r} not found "
            f"(have: {preview}{more})"
        ),
        required=False,
        hint=f"ollama pull {prep_model}",
    )


def _check_fictionreaper(binary: str) -> DoctorCheck:
    resolved = shutil.which(binary)
    if resolved is None:
        return DoctorCheck(
            name="fictionreaper",
            status=CheckStatus.WARN,
            message=f"not on PATH: {binary}",
            required=False,
            hint="Only needed for URL ingest; folder ingest works without it.",
        )
    return DoctorCheck(
        name="fictionreaper",
        status=CheckStatus.OK,
        message=f"found: {resolved}",
        required=False,
    )
