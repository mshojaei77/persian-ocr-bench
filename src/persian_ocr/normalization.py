"""
Persian-aware text normalization for OCR scoring.

Provides two scoring modes:
  - **strict**: only Unicode NFC + line whitespace cleanup.
  - **norm**  : full Persian normalization (Yeh/Kaf mapping, diacritic removal,
                zero-width cleanup).

Use ``strict`` for apples-to-apples comparison, ``norm`` for forgiving
leaderboard scores that reward correct content over spelling variants.
"""

from __future__ import annotations

import re
import unicodedata


# ── Character mappings ──────────────────────────────────────────────

# Arabic letters → Persian equivalents
TRANSLATION_TABLE = str.maketrans(
    {
        "\u064A": "\u06CC",  # Arabic Yeh        → Persian Yeh
        "\u0649": "\u06CC",  # Alif Maqsura      → Persian Yeh
        "\u0643": "\u06A9",  # Arabic Kaf         → Persian Kaf
        "\u06C0": "\u0647\u0654",  # Heh+Ye Above → Heh + Hamza
        "\u0640": "",  # Tatweel / Kashida (remove)
    }
)


# ── Public API ──────────────────────────────────────────────────────


def clean_lines(text: str) -> str:
    """Collapse horizontal whitespace per line and strip empty lines."""
    lines = [
        re.sub(r"[ \t]+", " ", ln).strip()
        for ln in text.replace("\r\n", "\n").split("\n")
    ]
    return "\n".join(ln for ln in lines if ln).strip()


def normalize_persian(text: str) -> str:
    """
    Full Persian-aware normalization:

    - NFKC Unicode normalisation
    - Arabic Yeh/Kaf → Persian
    - Remove optional Arabic diacritics (tashkeel)
    - Remove zero-width / invisible characters
    - Collapse whitespace, preserve line structure
    - Strip empty lines
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATION_TABLE)

    # Arabic diacritics (Fatha, Kasra, Damma, Shadda, Sukun, …)
    # and superscript Alef.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Zero-width characters, BOM, LRM, RLM, invisible operators
    text = re.sub(r"[\u200B-\u200F\uFEFF\u2060-\u2064]", "", text)

    return clean_lines(text)


def normalize_strict(text: str) -> str:
    """Minimal normalisation — NFC + line whitespace only."""
    return clean_lines(unicodedata.normalize("NFC", text))
