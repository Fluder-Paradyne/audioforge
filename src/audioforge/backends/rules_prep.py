"""Rule-based text preparation for TTS narration."""

from __future__ import annotations

import re

from audioforge.models import BuildOptions

# 3+ consecutive newlines → paragraph break (exactly two newlines).
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# HTML tags (simple; good enough for RR chrome remnants).
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Markdown images: ![alt](url) or ![alt][ref]
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")

# Fancy punctuation → ASCII-ish for TTS engines.
_FANCY_TRANS = str.maketrans(
    {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201a": "'",  # ‚
        "\u201b": "'",  # ‛
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u201e": '"',  # „
        "\u201f": '"',  # ‟
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2015": "-",  # ―
        "\u2212": "-",  # −
        "\u00a0": " ",  # nbsp
        "\u2026": "...",  # …
    }
)


class RulesTextPrep:
    """Deterministic cleanup: whitespace, HTML, images, fancy punctuation."""

    def prepare(self, text: str, *, options: BuildOptions) -> str:
        """Return speech-ready text derived from *text*."""
        del options  # protocol-compatible; rules ignore build options
        cleaned = text
        cleaned = _HTML_TAG_RE.sub("", cleaned)
        cleaned = _MD_IMAGE_RE.sub("", cleaned)
        cleaned = cleaned.translate(_FANCY_TRANS)
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned + "\n"
        return ""
