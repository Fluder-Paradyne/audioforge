"""Minimal FastAPI application factory (health only until full jobs API)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from audioforge.settings import AppSettings


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Build the AudioForge ASGI app.

    Parameters
    ----------
    settings:
        Optional settings (reserved for future jobs API wiring). Unused in the
        health-only stub but accepted so callers can pass config early.
    """
    del settings  # reserved for issue #10
    application = FastAPI(
        title="AudioForge",
        version="0.1.0",
        description="Local FictionReaper → audiobook pipeline API.",
    )

    @application.get("/health")
    def health() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "ok"}

    return application
