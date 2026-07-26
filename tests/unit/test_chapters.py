"""Tests for FictionReaper-style chapter discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from audioforge.io.chapters import discover_chapters, humanize_slug

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sample_book"


def test_discover_sample_book_order_and_titles() -> None:
    chapters = discover_chapters(FIXTURES)
    assert len(chapters) == 2
    assert chapters[0].index == 1
    assert chapters[0].slug == "chapter-one"
    assert chapters[0].title == "Chapter One"
    assert chapters[0].source_path == (FIXTURES / "0001-chapter-one.md").resolve()
    assert chapters[1].index == 2
    assert chapters[1].slug == "chapter-two"
    assert chapters[1].title == "Chapter Two"
    assert chapters[1].source_path.is_absolute()


def test_discover_sorts_by_numeric_index(tmp_path: Path) -> None:
    (tmp_path / "0010-ten.md").write_text("# Ten\n", encoding="utf-8")
    (tmp_path / "0002-two.md").write_text("# Two\n", encoding="utf-8")
    (tmp_path / "0001-one.md").write_text("# One\n", encoding="utf-8")
    chapters = discover_chapters(tmp_path)
    assert [c.index for c in chapters] == [1, 2, 10]
    assert [c.slug for c in chapters] == ["one", "two", "ten"]


def test_discover_skips_non_md_and_junk(tmp_path: Path) -> None:
    (tmp_path / "0001-good.md").write_text("# Good\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Readme\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")
    (tmp_path / "book.epub").write_text("fake-epub\n", encoding="utf-8")
    (tmp_path / "0000-zero.md").write_text("# Zero\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "0003-nested.md").write_text("# Nested\n", encoding="utf-8")

    chapters = discover_chapters(tmp_path)
    assert len(chapters) == 1
    assert chapters[0].slug == "good"


def test_title_falls_back_to_humanized_slug(tmp_path: Path) -> None:
    (tmp_path / "0003-the-dark-forest.md").write_text(
        "No heading here.\n\nJust body text.\n",
        encoding="utf-8",
    )
    chapters = discover_chapters(tmp_path)
    assert len(chapters) == 1
    assert chapters[0].title == "The Dark Forest"


def test_humanize_slug() -> None:
    assert humanize_slug("chapter-one") == "Chapter One"
    assert humanize_slug("the_dark_forest") == "The Dark Forest"
    assert humanize_slug("already") == "Already"


def test_discover_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="not found"):
        discover_chapters(missing)


def test_discover_empty_directory(tmp_path: Path) -> None:
    assert discover_chapters(tmp_path) == []


def test_accepts_short_digit_prefix(tmp_path: Path) -> None:
    (tmp_path / "1-short.md").write_text("# Short\n", encoding="utf-8")
    chapters = discover_chapters(tmp_path)
    assert len(chapters) == 1
    assert chapters[0].index == 1
    assert chapters[0].slug == "short"
