"""Tests for RulesTextPrep."""

from __future__ import annotations

from audioforge.backends.rules_prep import RulesTextPrep
from audioforge.models import BuildOptions


def _options() -> BuildOptions:
    return BuildOptions(source=".")


def test_collapse_three_plus_newlines() -> None:
    text = "Para one.\n\n\n\nPara two.\n"
    result = RulesTextPrep().prepare(text, options=_options())
    assert "\n\n\n" not in result
    assert result == "Para one.\n\nPara two.\n"


def test_strip_html_tags() -> None:
    text = '<div class="chapter">Hello <b>world</b></div>\n'
    result = RulesTextPrep().prepare(text, options=_options())
    assert "<" not in result
    assert "Hello world" in result


def test_normalize_fancy_quotes_and_dashes() -> None:
    text = "\u201cHello\u201d\u2014she said\u2014it\u2019s fine\u2026\n"
    result = RulesTextPrep().prepare(text, options=_options())
    assert result == '"Hello"-she said-it\'s fine...\n'


def test_strip_markdown_images_keep_dialogue() -> None:
    text = (
        'She said, "Look at this!"\n\n'
        "![cover art](https://example.com/img.png)\n\n"
        'He replied, "Nice."\n'
    )
    result = RulesTextPrep().prepare(text, options=_options())
    assert "![cover" not in result
    assert 'She said, "Look at this!"' in result
    assert 'He replied, "Nice."' in result


def test_strips_trailing_whitespace_adds_final_newline() -> None:
    text = "  Hello world.  \n\n  "
    result = RulesTextPrep().prepare(text, options=_options())
    assert result == "Hello world.\n"


def test_empty_input() -> None:
    assert RulesTextPrep().prepare("", options=_options()) == ""
    assert RulesTextPrep().prepare("   \n\n  ", options=_options()) == ""


def test_messy_markdown_html_combo() -> None:
    text = (
        "# Chapter\n\n\n\n"
        '<p>Once upon a time\u2014"long ago"\u2014there was a map.</p>\n\n'
        "![map](./map.png)\n\n\n"
        "The end.\n"
    )
    result = RulesTextPrep().prepare(text, options=_options())
    assert "<p>" not in result
    assert "![map]" not in result
    assert "\n\n\n" not in result
    assert 'Once upon a time-"long ago"-there was a map.' in result
    assert result.endswith("\n")


def test_reference_style_image_stripped() -> None:
    text = "Before\n\n![alt][ref]\n\nAfter\n"
    result = RulesTextPrep().prepare(text, options=_options())
    assert "![alt]" not in result
    assert "Before" in result
    assert "After" in result


def test_keeps_single_and_double_newlines() -> None:
    text = "Line one\nLine two\n\nNew para\n"
    result = RulesTextPrep().prepare(text, options=_options())
    assert result == "Line one\nLine two\n\nNew para\n"
