"""Tests for FFmpeg runners."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audioforge.backends.ffmpeg import (
    FakeFfmpegRunner,
    FfmpegError,
    SubprocessFfmpegRunner,
)


def test_fake_runner_records_commands_and_touches_m4b(tmp_path: Path) -> None:
    runner = FakeFfmpegRunner()
    out = tmp_path / "book.m4b"
    runner.run(["-y", "-i", "in.wav", str(out)])
    assert runner.commands == [["-y", "-i", "in.wav", str(out)]]
    assert out.is_file()


def test_fake_runner_empty_args_noop() -> None:
    runner = FakeFfmpegRunner()
    runner.run([])
    assert runner.commands == [[]]


def test_fake_runner_non_media_last_arg_no_touch(tmp_path: Path) -> None:
    runner = FakeFfmpegRunner()
    meta = tmp_path / "chapters.ffmetadata"
    runner.run(["-i", str(meta)])
    assert not meta.exists()
    assert runner.commands[0][-1] == str(meta)


def test_subprocess_runner_success() -> None:
    completed = MagicMock()
    completed.returncode = 0
    completed.stderr = ""
    completed.stdout = "ok"

    with patch(
        "audioforge.backends.ffmpeg.subprocess.run",
        return_value=completed,
    ) as mock_run:
        SubprocessFfmpegRunner(ffmpeg_path="/usr/bin/ffmpeg").run(
            ["-y", "-i", "a.wav", "out.m4b"]
        )

    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd == ["/usr/bin/ffmpeg", "-y", "-i", "a.wav", "out.m4b"]
    assert mock_run.call_args.kwargs["capture_output"] is True
    assert mock_run.call_args.kwargs["text"] is True
    assert mock_run.call_args.kwargs["check"] is False


def test_subprocess_runner_nonzero_exit() -> None:
    completed = MagicMock()
    completed.returncode = 2
    completed.stderr = "boom"
    completed.stdout = ""

    with (
        patch(
            "audioforge.backends.ffmpeg.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FfmpegError, match="exit code 2") as exc_info,
    ):
        SubprocessFfmpegRunner().run(["-version"])

    err = exc_info.value
    assert err.returncode == 2
    assert err.stderr == "boom"
    assert err.cmd[0] == "ffmpeg"


def test_subprocess_runner_nonzero_uses_stdout_when_no_stderr() -> None:
    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = ""
    completed.stdout = "stdout only"

    with (
        patch(
            "audioforge.backends.ffmpeg.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FfmpegError, match="stdout only"),
    ):
        SubprocessFfmpegRunner().run(["-i", "x"])


def test_subprocess_runner_nonzero_no_output() -> None:
    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = ""
    completed.stdout = ""

    with (
        patch(
            "audioforge.backends.ffmpeg.subprocess.run",
            return_value=completed,
        ),
        pytest.raises(FfmpegError, match="no output"),
    ):
        SubprocessFfmpegRunner().run(["-i", "x"])


def test_subprocess_runner_binary_missing() -> None:
    with (
        patch(
            "audioforge.backends.ffmpeg.subprocess.run",
            side_effect=FileNotFoundError("nope"),
        ),
        pytest.raises(FfmpegError, match="not found") as exc_info,
    ):
        SubprocessFfmpegRunner(ffmpeg_path="/missing/ffmpeg").run(["-version"])

    err = exc_info.value
    assert err.returncode is None
    assert err.cmd[0] == "/missing/ffmpeg"


def test_backends_export_ffmpeg() -> None:
    from audioforge.backends import (
        FakeFfmpegRunner as F,
    )
    from audioforge.backends import (
        FfmpegError as E,
    )
    from audioforge.backends import (
        FfmpegRunner,
        SubprocessFfmpegRunner,
    )

    assert F is FakeFfmpegRunner
    assert E is FfmpegError
    assert SubprocessFfmpegRunner is not None
    assert FfmpegRunner is not None
