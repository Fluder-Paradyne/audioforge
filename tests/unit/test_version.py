"""Tests for package version metadata."""

from __future__ import annotations

import re

from audioforge import __version__


def test_version_is_semver_ish() -> None:
    """``__version__`` matches major.minor.patch (optional pre-release suffix)."""
    pattern = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9.-]+)?$")
    assert pattern.match(__version__), f"unexpected version: {__version__!r}"


def test_version_is_0_1_0() -> None:
    """Scaffold ships as 0.1.0."""
    assert __version__ == "0.1.0"
