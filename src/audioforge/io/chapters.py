"""Discover FictionReaper-style chapter Markdown files."""

from __future__ import annotations

import re
from pathlib import Path

from audioforge.models import ChapterRef

# Numeric prefix (prefer 4+ zero-padded digits; any digit length accepted) + slug.
_CHAPTER_NAME_RE = re.compile(r"^(\d+)-(.+)\.md$", re.IGNORECASE)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def discover_chapters(source_dir: Path) -> list[ChapterRef]:
    """Return chapter refs for ``NNNN-slug.md`` files under *source_dir*.

    Non-``.md`` files (including EPUB) and ``.md`` files without a numeric prefix are
    skipped. Title is the first Markdown H1 if present, otherwise a humanized slug.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Chapter source directory not found: {source_dir}")

    discovered: list[tuple[int, str, Path]] = []
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        # Explicitly skip EPUB and any non-Markdown extension.
        suffix = path.suffix.lower()
        if suffix == ".epub" or suffix != ".md":
            continue
        match = _CHAPTER_NAME_RE.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index < 1:
            continue
        slug = match.group(2)
        discovered.append((index, slug, path.resolve()))

    discovered.sort(key=lambda item: item[0])

    chapters: list[ChapterRef] = []
    for index, slug, resolved in discovered:
        title = _title_for(resolved, slug)
        chapters.append(
            ChapterRef(
                index=index,
                title=title,
                source_path=resolved,
                slug=slug,
            )
        )
    return chapters


def _title_for(path: Path, slug: str) -> str:
    text = path.read_text(encoding="utf-8")
    heading = _H1_RE.search(text)
    if heading is not None:
        return heading.group(1).strip()
    return humanize_slug(slug)


def humanize_slug(slug: str) -> str:
    """Turn ``chapter-one`` / ``chapter_one`` into a readable title."""
    words = slug.replace("_", "-").split("-")
    return " ".join(word.capitalize() for word in words if word)
