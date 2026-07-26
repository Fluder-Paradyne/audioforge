"""Tests for the AudioForge CLI scaffolding."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from audioforge import __version__
from audioforge.cli import app, run


def test_version_flag() -> None:
    """``--version`` prints package version and exits 0."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"audioforge {__version__}" in result.stdout


def test_version_short_flag() -> None:
    """``-V`` is an alias for ``--version``."""
    runner = CliRunner()
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help() -> None:
    """``--help`` documents the CLI and exits 0."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AudioForge" in result.stdout or "audiobook" in result.stdout.lower()


def test_no_args_invokes_callback() -> None:
    """Bare invoke runs the root callback without error."""
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0


def test_run_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run()`` delegates to the Typer app."""
    called: list[bool] = []

    def fake_app() -> None:
        called.append(True)

    monkeypatch.setattr("audioforge.cli.app", fake_app)
    run()
    assert called == [True]
