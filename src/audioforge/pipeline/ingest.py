"""Ingest stage: copy or download chapter sources, then discover chapters."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from audioforge.backends.protocols import FictionReaperRunner
from audioforge.io.chapters import discover_chapters
from audioforge.io.paths import JobPaths
from audioforge.models import BuildOptions, ChapterRef


class IngestError(Exception):
    """Raised when the ingest stage cannot process the given source."""


def ingest(
    *,
    source: str,
    paths: JobPaths,
    options: BuildOptions,
    runner: FictionReaperRunner | None = None,
) -> list[ChapterRef]:
    """Populate *paths.source* from *source* and return sorted chapter refs.

    *source* may be an ``http(s)`` URL (requires *runner*) or a directory of
    FictionReaper-style ``NNNN-slug.md`` files. A bare single ``.md`` file is
    rejected in v1.
    """
    paths.ensure()

    if _is_http_url(source):
        return _ingest_url(
            source,
            paths=paths,
            options=options,
            runner=runner,
        )

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() == ".md":
            raise IngestError(
                f"Source is a single Markdown file ({source_path}). "
                "Provide a directory of FictionReaper-style chapter files "
                "(e.g. 0001-chapter-one.md) or an http(s) fiction URL."
            )
        raise IngestError(
            f"Source must be a directory of chapter Markdown files or an "
            f"http(s) URL; got file: {source_path}"
        )

    if not source_path.is_dir():
        raise IngestError(f"Source is not a directory or URL: {source_path}")

    return _ingest_directory(source_path, paths=paths)


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _ingest_url(
    source: str,
    *,
    paths: JobPaths,
    options: BuildOptions,
    runner: FictionReaperRunner | None,
) -> list[ChapterRef]:
    if runner is None:
        raise IngestError(
            "Source is a URL but no FictionReaper runner was provided. "
            "Install fictionreaper (and pass a FictionReaperRunner), or "
            "provide a local chapter directory instead of a URL."
        )
    download_dir = Path(
        runner.run(source, paths.source, bin_path=options.fictionreaper_bin)
    )
    if download_dir.resolve() != paths.source.resolve():
        _copy_markdown_files(download_dir, paths.source)
    return discover_chapters(paths.source)


def _ingest_directory(source_path: Path, *, paths: JobPaths) -> list[ChapterRef]:
    if source_path.resolve() != paths.source.resolve():
        _copy_markdown_files(source_path, paths.source)
    return discover_chapters(paths.source)


def _copy_markdown_files(src_dir: Path, dest_dir: Path) -> None:
    """Copy ``*.md`` files from *src_dir* into *dest_dir* for workspace isolation."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".md":
            shutil.copy2(path, dest_dir / path.name)
