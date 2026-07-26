"""Tests for the minimal FastAPI app stub."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from audioforge.api.app import create_app
from audioforge.settings import AppSettings


def test_create_app_health() -> None:
    application = create_app()
    client = TestClient(application)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_accepts_settings() -> None:
    application = create_app(AppSettings(work_dir=Path("work-x")))
    client = TestClient(application)
    assert client.get("/health").json()["status"] == "ok"
