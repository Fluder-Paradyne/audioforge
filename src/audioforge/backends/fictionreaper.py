"""FictionReaper subprocess runner and test double."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FictionReaperError(Exception):
    """FictionReaper binary missing or exited unsuccessfully."""

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


class SubprocessFictionReaperRunner:
    """Invoke the FictionReaper CLI via subprocess."""

    def run(self, url: str, output_dir: Path, *, bin_path: str) -> Path:
        """Run ``fictionreaper download`` into *output_dir*; return that directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd: list[str] = [
            bin_path,
            "download",
            url,
            "--output-dir",
            str(output_dir),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FictionReaperError(
                f"FictionReaper binary not found: {bin_path}. "
                "Install fictionreaper or set --fictionreaper-bin / "
                "BuildOptions.fictionreaper_bin.",
                returncode=None,
                cmd=cmd,
            ) from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no output").strip()
            raise FictionReaperError(
                f"FictionReaper failed with exit code {result.returncode}: {detail}",
                returncode=result.returncode,
                stderr=result.stderr or "",
                cmd=cmd,
            )
        return output_dir


class FakeFictionReaperRunner:
    """Test double that writes sample chapter Markdown into *output_dir*."""

    def run(self, url: str, output_dir: Path, *, bin_path: str) -> Path:
        """Ignore *url*/*bin_path* and write two sample chapters under *output_dir*."""
        del url, bin_path  # protocol-compatible; unused by the fake
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        samples: list[tuple[str, str]] = [
            (
                "0001-chapter-one.md",
                "# Chapter One\n\nOnce upon a time in a quiet village.\n",
            ),
            (
                "0002-chapter-two.md",
                "# Chapter Two\n\nThe adventure continues.\n",
            ),
        ]
        for name, content in samples:
            (output_dir / name).write_text(content, encoding="utf-8")
        return output_dir
