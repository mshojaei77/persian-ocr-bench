"""
Edit distance, CER, WER, and line-exact scoring.

All scores are symmetric on the normalized text — call
``normalize_persian`` or ``normalize_strict`` before passing strings in.
"""

from __future__ import annotations

from typing import Sequence


def edit_distance(left: Sequence[str] | str, right: Sequence[str] | str) -> int:
    """Levenshtein distance (character-level for str, token-level otherwise)."""
    if len(left) < len(right):
        left, right = right, left
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        cur = [i]
        for j, right_item in enumerate(right, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (left_item != right_item),
                )
            )
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    """CER = edit_distance(chars) / len(reference_chars)."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(reference, prediction) / len(reference)


def word_error_rate(reference: str, prediction: str) -> float:
    """WER = edit_distance(words) / len(reference_words)."""
    ref_words = reference.split()
    pred_words = prediction.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(ref_words, pred_words) / len(ref_words)


def line_exact_match(reference: str, prediction: str) -> float:
    """Fraction of reference non-empty lines found verbatim in prediction."""
    ref_lines = [ln for ln in reference.split("\n") if ln.strip()]
    pred_lines = {ln for ln in prediction.split("\n") if ln.strip()}
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    return sum(1 for ln in ref_lines if ln in pred_lines) / len(ref_lines)


__all__ = [
    "edit_distance",
    "character_error_rate",
    "word_error_rate",
    "line_exact_match",
]
