"""Explicit Persian OCR text-normalization policies."""

from __future__ import annotations

import re
import unicodedata

import regex


PERSIAN_YEH = "ی"
ARABIC_YEH = "ي"
PERSIAN_KAF = "ک"
ARABIC_KAF = "ك"
ZWNJ = "‌"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"

_MD_HEADING = re.compile(r"^#+\s*", flags=re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_MD_RULE = re.compile(r"^[-*_]{3,}\s*$", flags=re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_BIDI_CONTROLS = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def strip_markdown(text: str) -> str:
    """Remove formatting syntax while preserving visible annotation text."""
    text = _MD_HEADING.sub("", text or "")
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC_STAR.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    text = _HTML_TAG.sub("", text).replace("`", "")
    lines = []
    for line in text.splitlines():
        if _TABLE_SEPARATOR.match(line):
            continue
        if line.count("|") >= 2:
            line = line.strip().strip("|").replace("|", " ")
        lines.append(line)
    return "\n".join(lines).strip()


def graphemes(text: str) -> list[str]:
    """Return user-perceived Unicode characters, including combining marks."""
    return regex.findall(r"\X", text or "")


def diagnostic_units(text: str) -> list[str]:
    """Return base code points plus standalone ZWNJ for diagnostics."""
    normalized = unicodedata.normalize("NFC", text or "")
    return [
        character
        for character in normalized
        if character == ZWNJ or not unicodedata.category(character).startswith("M")
    ]


def normalize_transport(text: str) -> str:
    """Normalize transport-only differences without changing orthography."""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\f", "")
    return unicodedata.normalize("NFC", normalized).strip()


def normalize_digits_optional(text: str) -> str:
    """Canonicalize Arabic-Indic and Persian digits to ASCII."""
    return (text or "").translate(
        str.maketrans(ARABIC_DIGITS + PERSIAN_DIGITS, ASCII_DIGITS + ASCII_DIGITS)
    )


def normalize_fa(text: str, policy: str = "canonical") -> str:
    """Apply strict, canonical, or search normalization; always preserve ZWNJ."""
    normalized = normalize_transport(text)
    if policy == "strict":
        return normalized
    if policy not in {"canonical", "search"}:
        raise ValueError(f"Unknown normalization policy: {policy}")
    normalized = _BIDI_CONTROLS.sub("", normalized)
    normalized = normalized.replace(ARABIC_YEH, PERSIAN_YEH)
    normalized = normalized.replace(ARABIC_KAF, PERSIAN_KAF)
    if policy == "search":
        normalized = normalize_digits_optional(normalized).replace("ـ", "")
    return re.sub(r"\s+", " ", normalized).strip()


__all__ = [
    "ARABIC_DIGITS",
    "ARABIC_KAF",
    "ARABIC_YEH",
    "ASCII_DIGITS",
    "PERSIAN_DIGITS",
    "PERSIAN_KAF",
    "PERSIAN_YEH",
    "ZWNJ",
    "diagnostic_units",
    "graphemes",
    "normalize_digits_optional",
    "normalize_fa",
    "normalize_transport",
    "strip_markdown",
]
