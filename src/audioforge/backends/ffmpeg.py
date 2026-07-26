"""FFmpeg subprocess runner and test double."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Extensions treated as output media when faking a successful encode.
_OUTPUT_SUFFIXES = frozenset({".m4b", ".m4a", ".mp4", ".aac", ".mp3", ".wav"})


class FfmpegError(Exception):
    """FFmpeg binary missing or exited unsuccessfully."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
        cmd: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.cmd = cmd if cmd is not None else []


class SubprocessFfmpegRunner:
    """Invoke the FFmpeg CLI via subprocess."""

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    def run(self, args: list[str]) -> None:
        """Run ``[ffmpeg_path, *args]``; raise :class:`FfmpegError` on failure."""
        cmd: list[str] = [self.ffmpeg_path, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FfmpegError(
                f"FFmpeg binary not found: {self.ffmpeg_path}. "
                "Install FFmpeg or set AUDIOFORGE_FFMPEG_PATH / "
                "AppSettings.ffmpeg_path.",
                returncode=None,
                cmd=cmd,
            ) from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no output").strip()
            raise FfmpegError(
                f"FFmpeg failed with exit code {result.returncode}: {detail}",
                returncode=result.returncode,
                stderr=result.stderr or "",
                cmd=cmd,
            )


class FakeFfmpegRunner:
    """Test double that records FFmpeg argument lists and touches outputs.

    When the last argument looks like a media output path (``.m4b``, ``.m4a``,
    etc.), that file is created empty so callers can assert the path exists.
    """

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, args: list[str]) -> None:
        """Record *args* and optionally create the last path as an empty file."""
        recorded = list(args)
        self.commands.append(recorded)
        if not recorded:
            return
        out = Path(recorded[-1])
        if out.suffix.lower() in _OUTPUT_SUFFIXES:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.touch()
