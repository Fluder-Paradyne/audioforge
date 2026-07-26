"""I/O helpers: chapter discovery and job workspace paths."""

from __future__ import annotations

from audioforge.io.chapters import discover_chapters, humanize_slug
from audioforge.io.paths import JobPaths

__all__ = ["JobPaths", "discover_chapters", "humanize_slug"]
