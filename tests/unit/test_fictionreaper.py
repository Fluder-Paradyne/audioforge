"""Tests for FictionReaper runners."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioforge.backends.fictionreaper import (
    FakeFictionReaperRunner,
    FictionReaperError,
    SubprocessFictionReaperRunner,
)
from audioforge.io.chapters import discover_chapters


def test_fake_runner_writes_sample_chapters(tmp_path: Path) -> None:
    runner = FakeFictionReaperRunner()
    out = runner.run(
        "https://www.royalroad.com/fiction/1",
        tmp_path / "dl",
        bin_path="fictionreaper",
    )
    assert out == tmp_path / "dl"
    chapters = discover_chapters(out)
    assert len(chapters) == 2
    assert chapters[0].title == "Chapter One"
    assert chapters[1].title == "Chapter Two"


def test_subprocess_runner_success(tmp_path: Path) -> None:
    out_dir = tmp_path / "src"
    completed = MagicMock()
    completed.returncode = 0
    completed.stderr = ""
    completed.stdout = "ok"

    with patch(
        "audioforge.backends.fictionreaper.subprocess.run",
        return_value=completed,
    ) as mock_run:
        result = SubprocessFictionReaperRunner().run(
            "https://example.com/fiction/1",
            out_dir,
            bin_path="/usr/bin/fictionreaper",
        )

    assert result == out_dir
    assert out_dir.is_dir()
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd == [
        "/usr/bin/fictionreaper",
        "download",
        "https://example.com/fiction/1",
        "--output-dir",
        str(out_dir),
    ]
    assert mock_run.call_args.kwargs["capture_output"] is True
    assert mock_run.call_args.kwargs["text"] is True
    assert mock_run.call_args.kwargs["check"] is False


def test_subprocess_runner_nonzero_exit(tmp_path: Path) -> None:
    completed = MagicMock()
    completed.returncode = 2
    completed.stderr = "boom"
    completed.stdout = ""

    with (
        patch(
            "audioforge.backends.fictionreaper.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FictionReaperError, match="exit code 2") as exc_info,
    ):
        SubprocessFictionReaperRunner().run(
            "https://example.com/f/1",
            tmp_path,
            bin_path="fictionreaper",
        )

    err = exc_info.value
    assert err.returncode == 2
    assert err.stderr == "boom"
    assert err.cmd[0] == "fictionreaper"


def test_subprocess_runner_nonzero_uses_stdout_when_no_stderr(tmp_path: Path) -> None:
    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = ""
    completed.stdout = "stdout only"

    with (
        patch(
            "audioforge.backends.fictionreaper.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FictionReaperError, match="stdout only"),
    ):
        SubprocessFictionReaperRunner().run(
            "https://example.com/f/1",
            tmp_path,
            bin_path="fictionreaper",
        )


def test_subprocess_runner_nonzero_no_output(tmp_path: Path) -> None:
    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = ""
    completed.stdout = ""

    with (
        patch(
            "audioforge.backends.fictionreaper.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FictionReaperError, match="no output"),
    ):
        SubprocessFictionReaperRunner().run(
            "https://example.com/f/1",
            tmp_path,
            bin_path="fictionreaper",
        )


def test_subprocess_runner_binary_not_found(tmp_path: Path) -> None:
    with (
        patch(
            "audioforge.backends.fictionreaper.subprocess.run",
            side_effect=FileNotFoundError("missing"),
        ),
        pytest.raises(FictionReaperError, match="binary not found") as exc_info,
    ):
        SubprocessFictionReaperRunner().run(
            "https://example.com/f/1",
            tmp_path,
            bin_path="/missing/fictionreaper",
        )

    assert exc_info.value.returncode is None
    assert "/missing/fictionreaper" in str(exc_info.value)


def test_fictionreaper_error_defaults() -> None:
    err = FictionReaperError("x")
    assert err.returncode is None
    assert err.stderr == ""
    assert err.cmd == []


def test_subprocess_uses_real_completed_process_type(tmp_path: Path) -> None:
    """Smoke: kwargs match subprocess.run signature (no accidental kwargs bugs)."""
    with patch(
        "audioforge.backends.fictionreaper.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["fictionreaper"],
            returncode=0,
            stdout="",
            stderr="",
        ),
    ):
        path = SubprocessFictionReaperRunner().run(
            "https://example.com/f/1",
            tmp_path / "o",
            bin_path="fictionreaper",
        )
    assert path.is_dir()
