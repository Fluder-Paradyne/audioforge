"""Tests for the FastAPI jobs API and health endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audioforge.api.app import create_app
from audioforge.backends.fake import FakeTtsBackend
from audioforge.backends.ffmpeg import FakeFfmpegRunner
from audioforge.backends.fictionreaper import FakeFictionReaperRunner
from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.factory import DefaultBackends
from audioforge.models import BuildOptions, JobStatus
from audioforge.settings import AppSettings

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"


class BrokenPrep:
    """Always fails prepare (pipeline failure fixture)."""

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        del text, options
        raise RuntimeError("prep deliberately broken")


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        work_dir=tmp_path / "work",
        ffmpeg_path="ffmpeg",
        ollama_base_url="http://127.0.0.1:11434",
    )


def _fake_kwargs() -> dict[str, object]:
    return {
        "prep": RulesTextPrep(),
        "tts": FakeTtsBackend(),
        "ffmpeg": FakeFfmpegRunner(),
        "fictionreaper": FakeFictionReaperRunner(),
    }


def _sync_client(tmp_path: Path, **overrides: object) -> TestClient:
    kwargs: dict[str, object] = {
        "settings": _settings(tmp_path),
        "run_sync": True,
        **_fake_kwargs(),
    }
    kwargs.update(overrides)
    application = create_app(**kwargs)  # type: ignore[arg-type]
    return TestClient(application)


# --- health ---


def test_create_app_health(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ffmpeg_configured"] is True
    assert body["ollama_base_url"] == "http://127.0.0.1:11434"


def test_create_app_accepts_settings(tmp_path: Path) -> None:
    settings = AppSettings(
        work_dir=tmp_path / "work-x",
        ffmpeg_path="",
        ollama_base_url="http://ollama.local:11434",
    )
    application = create_app(settings, run_sync=True, **_fake_kwargs())  # type: ignore[arg-type]
    client = TestClient(application)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ffmpeg_configured"] is False
    assert body["ollama_base_url"] == "http://ollama.local:11434"


def test_create_app_default_settings_health() -> None:
    """Factory works with no settings and no injected backends (uvicorn path)."""
    application = create_app()
    client = TestClient(application)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "ffmpeg_configured" in body
    assert "ollama_base_url" in body


# --- POST /jobs ---


def test_create_job_sync_completed(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "api-job-1",
            "skip_prep": True,
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "api-job-1"
    assert body["status"] == JobStatus.COMPLETED.value


def test_create_job_generates_job_id(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.post(
        "/jobs",
        json={"source": str(FIXTURES), "skip_prep": True},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["status"] == JobStatus.COMPLETED.value


def test_create_job_pipeline_failure(tmp_path: Path) -> None:
    client = _sync_client(tmp_path, prep=BrokenPrep())
    response = client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "fail-job",
            "skip_prep": False,
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "fail-job"
    assert body["status"] == JobStatus.FAILED.value

    detail = client.get("/jobs/fail-job")
    assert detail.status_code == 200
    job = detail.json()
    assert job["status"] == JobStatus.FAILED.value
    assert "broken" in (job["error"] or "")


def test_create_job_async_background(tmp_path: Path) -> None:
    """BackgroundTasks run before TestClient returns the response."""
    application = create_app(
        _settings(tmp_path),
        run_sync=False,
        **_fake_kwargs(),  # type: ignore[arg-type]
    )
    client = TestClient(application)
    response = client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "async-job",
            "skip_prep": True,
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == "async-job"
    # Accepted response is pending; background task then completes.
    assert body["status"] == JobStatus.PENDING.value

    job = client.get("/jobs/async-job")
    assert job.status_code == 200
    assert job.json()["status"] == JobStatus.COMPLETED.value


def test_create_job_uses_default_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When backends are not injected, factory defaults are used."""
    called: list[str] = []

    def fake_create_default_backends(
        settings: AppSettings,
        options: BuildOptions,
        **kwargs: object,
    ) -> DefaultBackends:
        del settings, options, kwargs
        called.append("yes")
        return DefaultBackends(
            prep=RulesTextPrep(),
            tts=FakeTtsBackend(),
            ffmpeg=FakeFfmpegRunner(),
            fictionreaper=FakeFictionReaperRunner(),
        )

    monkeypatch.setattr(
        "audioforge.api.app.create_default_backends",
        fake_create_default_backends,
    )
    application = create_app(_settings(tmp_path), run_sync=True)
    client = TestClient(application)
    response = client.post(
        "/jobs",
        json={"source": str(FIXTURES), "job_id": "factory-job", "skip_prep": True},
    )
    assert response.status_code == 202
    assert response.json()["status"] == JobStatus.COMPLETED.value
    assert called == ["yes"]


def test_create_job_partial_backend_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial injection fills missing backends from the factory."""
    application = create_app(
        _settings(tmp_path),
        run_sync=True,
        prep=RulesTextPrep(),
        # tts / ffmpeg / fictionreaper omitted → factory
    )
    monkeypatch.setattr(
        "audioforge.api.app.create_default_backends",
        lambda settings, options, **kwargs: DefaultBackends(
            prep=RulesTextPrep(),
            tts=FakeTtsBackend(),
            ffmpeg=FakeFfmpegRunner(),
            fictionreaper=FakeFictionReaperRunner(),
        ),
    )
    client = TestClient(application)
    response = client.post(
        "/jobs",
        json={"source": str(FIXTURES), "job_id": "partial-job", "skip_prep": True},
    )
    assert response.status_code == 202
    assert response.json()["status"] == JobStatus.COMPLETED.value


# --- GET /jobs/{id} ---


def test_get_job_not_found(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.get("/jobs/missing-job")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_job_after_create(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "get-job",
            "skip_prep": True,
            "voice": "af_bella",
        },
    )
    response = client.get("/jobs/get-job")
    assert response.status_code == 200
    job = response.json()
    assert job["job_id"] == "get-job"
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["options"]["voice"] == "af_bella"
    assert len(job["chapters"]) == 2
    assert job["artifacts"] is not None


# --- GET /jobs/{id}/artifacts ---


def test_get_artifacts_not_found(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.get("/jobs/no-such/artifacts")
    assert response.status_code == 404


def test_get_artifacts_after_complete(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "art-job",
            "skip_prep": True,
        },
    )
    response = client.get("/jobs/art-job/artifacts")
    assert response.status_code == 200
    body = response.json()
    assert "chapter_audio" in body
    assert len(body["chapter_audio"]) == 2
    assert body["m4b_path"] is not None
    for p in body["chapter_audio"]:
        assert Path(p).is_file()
    assert Path(body["m4b_path"]).is_file()


def test_get_artifacts_conflict_when_not_ready(tmp_path: Path) -> None:
    client = _sync_client(tmp_path, prep=BrokenPrep())
    client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "no-art",
            "skip_prep": False,
        },
    )
    response = client.get("/jobs/no-art/artifacts")
    assert response.status_code == 409
    assert "not ready" in response.json()["detail"].lower()


def test_create_job_validation_error(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.post("/jobs", json={})
    assert response.status_code == 422


def test_create_job_whitespace_job_id_gets_new_id(tmp_path: Path) -> None:
    client = _sync_client(tmp_path)
    response = client.post(
        "/jobs",
        json={
            "source": str(FIXTURES),
            "job_id": "   ",
            "skip_prep": True,
        },
    )
    assert response.status_code == 202
    assert response.json()["job_id"].strip()
    assert response.json()["job_id"] != "   "
