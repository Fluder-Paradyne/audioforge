"""Tests for audioforge doctor diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from audioforge import __version__
from audioforge.doctor import (
    CheckStatus,
    DoctorCheck,
    DoctorReport,
    _check_ffmpeg,
    _check_fictionreaper,
    _check_kokoro,
    _check_ollama,
    _check_python,
    _check_work_dir,
    format_doctor_report,
    run_doctor,
)
from audioforge.factory import _ALLOW_FAKE_TTS_ENV
from audioforge.settings import AppSettings


def test_doctor_report_ok_when_no_required_fails() -> None:
    report = DoctorReport(
        version="0.1.0",
        checks=[
            DoctorCheck(name="a", status=CheckStatus.OK, message="ok", required=True),
            DoctorCheck(name="b", status=CheckStatus.WARN, message="w", required=False),
        ],
    )
    assert report.ok is True


def test_doctor_report_not_ok_on_required_fail() -> None:
    report = DoctorReport(
        version="0.1.0",
        checks=[
            DoctorCheck(name="a", status=CheckStatus.FAIL, message="no", required=True),
        ],
    )
    assert report.ok is False


def test_format_doctor_report_includes_overall() -> None:
    report = DoctorReport(
        version="0.1.0",
        checks=[
            DoctorCheck(
                name="ffmpeg",
                status=CheckStatus.FAIL,
                message="missing",
                required=True,
                hint="install ffmpeg",
            ),
        ],
    )
    text = format_doctor_report(report)
    assert "audioforge doctor 0.1.0" in text
    assert "[FAIL]" in text
    assert "hint: install ffmpeg" in text
    assert "not ready" in text


def test_run_doctor_with_injected_probes(tmp_path: Path) -> None:
    settings = AppSettings(work_dir=tmp_path / "work")

    def ok_python() -> DoctorCheck:
        return DoctorCheck(
            name="python", status=CheckStatus.OK, message="3.12", required=True
        )

    def ok_work(path: Path) -> DoctorCheck:
        return DoctorCheck(
            name="work_dir",
            status=CheckStatus.OK,
            message=str(path),
            required=True,
        )

    def ok_ffmpeg(path: str) -> DoctorCheck:
        return DoctorCheck(
            name="ffmpeg",
            status=CheckStatus.OK,
            message=path,
            required=True,
        )

    def ok_kokoro() -> DoctorCheck:
        return DoctorCheck(
            name="kokoro", status=CheckStatus.OK, message="yes", required=True
        )

    def warn_ollama(base: str, model: str) -> DoctorCheck:
        return DoctorCheck(
            name="ollama",
            status=CheckStatus.WARN,
            message=f"{base} {model}",
            required=False,
        )

    def warn_fr(binary: str) -> DoctorCheck:
        return DoctorCheck(
            name="fictionreaper",
            status=CheckStatus.WARN,
            message=binary,
            required=False,
        )

    report = run_doctor(
        settings,
        fictionreaper_bin="fr-bin",
        check_python=ok_python,
        check_work_dir=ok_work,
        check_ffmpeg=ok_ffmpeg,
        check_kokoro=ok_kokoro,
        check_ollama=warn_ollama,
        check_fictionreaper=warn_fr,
    )
    assert report.ok is True
    assert report.version == __version__
    names = [c.name for c in report.checks]
    assert names == [
        "audioforge",
        "python",
        "work_dir",
        "ffmpeg",
        "kokoro",
        "ollama",
        "fictionreaper",
        "defaults",
    ]
    assert report.checks[5].message.endswith("llama3.2:3b") or "fr-bin" in [
        c.message for c in report.checks
    ]


def test_check_python_ok() -> None:
    check = _check_python()
    assert check.name == "python"
    assert check.required is True
    assert check.status is CheckStatus.OK


def test_check_python_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from collections import namedtuple

    version_info = namedtuple(
        "version_info",
        "major minor micro releaselevel serial",
    )
    monkeypatch.setattr(
        sys,
        "version_info",
        version_info(3, 11, 9, "final", 0),
    )
    check = _check_python()
    assert check.status is CheckStatus.FAIL
    assert "3.11" in check.message


def test_check_work_dir_ok(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "work"
    check = _check_work_dir(target)
    assert check.status is CheckStatus.OK
    assert target.is_dir()


def test_check_work_dir_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "work"

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "mkdir", boom)
    check = _check_work_dir(target)
    assert check.status is CheckStatus.FAIL
    assert check.required is True
    assert check.hint is not None


def test_check_ffmpeg_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_fn(*_a: object, **_k: object) -> object:
        raise FileNotFoundError("nope")

    monkeypatch.setattr("audioforge.doctor.subprocess.run", raise_fn)
    check = _check_ffmpeg("/no/such/ffmpeg")
    assert check.status is CheckStatus.FAIL
    assert "not found" in check.message


def test_check_ffmpeg_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0
        stdout = "ffmpeg version 6.0 Copyright\nconfiguration: --..."
        stderr = ""

    monkeypatch.setattr(
        "audioforge.doctor.subprocess.run",
        lambda *_a, **_k: Result(),
    )
    monkeypatch.setattr("audioforge.doctor.shutil.which", lambda p: f"/usr/bin/{p}")
    check = _check_ffmpeg("ffmpeg")
    assert check.status is CheckStatus.OK
    assert "ffmpeg version 6.0" in check.message
    assert check.required is True


def test_check_ffmpeg_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stdout = ""
        stderr = "broken"

    monkeypatch.setattr(
        "audioforge.doctor.subprocess.run",
        lambda *_a, **_k: Result(),
    )
    check = _check_ffmpeg("ffmpeg")
    assert check.status is CheckStatus.FAIL


def test_check_ffmpeg_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    def raise_timeout(*_a: object, **_k: object) -> object:
        raise sp.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr("audioforge.doctor.subprocess.run", raise_timeout)
    check = _check_ffmpeg("ffmpeg")
    assert check.status is CheckStatus.FAIL
    assert "timed out" in check.message


def test_check_ffmpeg_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_os(*_a: object, **_k: object) -> object:
        raise OSError("exec format error")

    monkeypatch.setattr("audioforge.doctor.subprocess.run", raise_os)
    check = _check_ffmpeg("ffmpeg")
    assert check.status is CheckStatus.FAIL
    assert "could not run" in check.message


def test_check_kokoro_missing_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ALLOW_FAKE_TTS_ENV, raising=False)

    def boom(name: str) -> object:
        raise ImportError("no kokoro")

    monkeypatch.setattr("audioforge.doctor.importlib.import_module", boom)
    check = _check_kokoro()
    assert check.status is CheckStatus.FAIL
    assert check.required is True


def test_check_kokoro_missing_warn_when_fake_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ALLOW_FAKE_TTS_ENV, "1")

    def boom(name: str) -> object:
        raise ImportError("no kokoro")

    monkeypatch.setattr("audioforge.doctor.importlib.import_module", boom)
    check = _check_kokoro()
    assert check.status is CheckStatus.WARN
    assert check.required is False


def test_check_kokoro_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMod:
        KPipeline = object()

    monkeypatch.setattr(
        "audioforge.doctor.importlib.import_module",
        lambda name: FakeMod(),
    )
    check = _check_kokoro()
    assert check.status is CheckStatus.OK
    assert "KPipeline" in check.message


def test_check_kokoro_missing_kpipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMod:
        pass

    monkeypatch.setattr(
        "audioforge.doctor.importlib.import_module",
        lambda name: FakeMod(),
    )
    check = _check_kokoro()
    assert check.status is CheckStatus.FAIL


def test_check_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> BoomClient:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> object:
            raise httpx.ConnectError("down", request=httpx.Request("GET", url))

    monkeypatch.setattr("audioforge.doctor.httpx.Client", BoomClient)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "unreachable" in check.message


def test_check_ollama_model_present(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "llama3.2:3b"}, {"name": "other:latest"}]}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.OK
    assert "model available" in check.message


def test_check_ollama_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "mistral:7b"}]}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "not found" in check.message


def test_check_ollama_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 500

        def json(self) -> dict[str, Any]:
            return {}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "HTTP 500" in check.message


def test_check_ollama_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            raise ValueError("not json")

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "non-JSON" in check.message


def test_check_ollama_empty_models(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": [{"name": 123}, "skip", {"model": "only-via-model-key"}]}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    # only-via-model-key listed; not the default prep model → warn not found
    assert check.status is CheckStatus.WARN


def test_check_ollama_no_models_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": []}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "no models listed" in check.message


def test_check_ollama_models_not_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"models": "unexpected"}

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.WARN
    assert "no models listed" in check.message


def test_check_ollama_base_name_match(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            # Base match path: prep llama3.2:3b, list has llama3.2:latest
            return {
                "models": [
                    {"name": f"extra-{i}:1"} for i in range(6)
                ]
                + [{"name": "llama3.2:latest"}]
            }

    class Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> Resp:
            return Resp()

    monkeypatch.setattr("audioforge.doctor.httpx.Client", Client)
    check = _check_ollama("http://127.0.0.1:11434", "llama3.2:3b")
    assert check.status is CheckStatus.OK


def test_check_fictionreaper_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audioforge.doctor.shutil.which", lambda _b: None)
    check = _check_fictionreaper("fictionreaper")
    assert check.status is CheckStatus.WARN
    assert "not on PATH" in check.message


def test_check_fictionreaper_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "audioforge.doctor.shutil.which",
        lambda b: f"/opt/bin/{b}",
    )
    check = _check_fictionreaper("fictionreaper")
    assert check.status is CheckStatus.OK
    assert "/opt/bin/fictionreaper" in check.message


def test_format_ready_report() -> None:
    report = DoctorReport(
        version="0.1.0",
        checks=[
            DoctorCheck(name="x", status=CheckStatus.OK, message="fine", required=True),
        ],
    )
    assert "ready" in format_doctor_report(report)


def test_doctor_report_json_roundtrip() -> None:
    report = DoctorReport(
        version="0.1.0",
        checks=[
            DoctorCheck(name="x", status=CheckStatus.OK, message="m", required=False),
        ],
    )
    payload = json.loads(report.model_dump_json())
    assert payload["version"] == "0.1.0"
    assert payload["checks"][0]["status"] == "ok"
