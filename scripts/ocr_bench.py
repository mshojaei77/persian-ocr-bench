"""
Persian OCR Benchmark — shared scoring and normalization utilities.

Every benchmark script should import from here instead of duplicating
these functions.  This ensures all models are scored identically.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any

__all__ = [
    "TRANSLATION_TABLE",
    "clean_lines",
    "normalize_text",
    "normalize_text_strict",
    "edit_distance",
    "error_rate",
    "line_exact",
    "csv_fieldnames",
    "score_predictions",
    "bench_items",
    "repo_path",
    "serializable",
]

# ── Persian character normalization ───────────────────────────────────

TRANSLATION_TABLE = str.maketrans(
    {
        "\u064A": "\u06CC",  # Arabic Yeh  → Persian Yeh
        "\u0649": "\u06CC",  # Alif Maqsura → Persian Yeh
        "\u0643": "\u06A9",  # Arabic Kaf  → Persian Kaf
        "\u06C0": "\u0647\u0654",  # Heh with Yeh above → Heh + Hamza
        "\u0640": "",  # Tatweel/kashida (removed)
    }
)


def clean_lines(text: str) -> str:
    """Collapse whitespace per line and strip, preserving line breaks."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.replace("\r\n", "\n").split("\n")
    ]
    return "\n".join(line for line in lines if line).strip()


def normalize_text(text: str) -> str:
    """
    Persian-aware normalization:
      - NFKC Unicode normalization
      - Arabic Yeh/Kaf → Persian equivalents
      - Remove optional diacritics (tashkeel)
      - Remove zero-width characters (ZWJ, ZWNBSP, BOM)
      - Normalize whitespace while preserving line structure
      - Strip empty lines
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATION_TABLE)

    # Remove Arabic diacritics (Fatha, Kasra, Damma, Sukun, Shadda, etc.)
    # and superscript Alef.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Remove zero-width characters (ZWJ, ZWNJ, ZWNBSP, BOM, LRM, RLM)
    text = re.sub(r"[\u200B-\u200F\uFEFF\u2060-\u2064]", "", text)

    # Normalize half-space to standard ZWNJ
    # (already done by NFKC in most cases)

    return clean_lines(text)


def normalize_text_strict(text: str) -> str:
    """Minimal normalization: Unicode NFC + line whitespace only."""
    return clean_lines(unicodedata.normalize("NFC", text))


# ── Edit distance and error rates ────────────────────────────────────


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    """Standard Levenshtein distance for strings or token lists."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str] | str, prediction: list[str] | str) -> float:
    """CER or WER depending on whether you pass characters or words."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(reference, prediction) / len(reference)


def line_exact(reference: str, prediction: str) -> float:
    """Fraction of reference lines found verbatim in prediction."""
    ref_lines = [line for line in reference.split("\n") if line.strip()]
    pred_lines = {line for line in prediction.split("\n") if line.strip()}
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    return sum(1 for line in ref_lines if line in pred_lines) / len(ref_lines)


# ── Scoring pipeline ─────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "model",
    "split",
    "item",
    "cer_strict",
    "wer_strict",
    "cer_norm",
    "wer_norm",
    "line_exact_norm",
    "ref_chars",
    "pred_chars",
]

SUMMARY_FIELDNAMES = [
    "model",
    "split",
    "mean_cer_norm",
    "mean_wer_norm",
    "mean_line_exact_norm",
    "items",
]


def score_predictions(
    bench_root: Path,
    output_root: Path,
    model_name: str,
) -> None:
    """
    Read reference + prediction pairs under *bench_root*, compute strict
    and normalized scores, and write ``scores.csv`` + ``summary.csv``.

    Directory layout expected::

        bench_root/
          typed/   1.jpg  1.md  …
          hand-written/  1.jpg  1.md  …
        output_root/
          typed/  1.md  …
          hand-written/  1.md  …
    """
    rows: list[dict[str, Any]] = []
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        ref_path = image_path.with_suffix(".md")
        pred_path = output_root / split / f"{image_path.stem}.md"
        reference = ref_path.read_text(encoding="utf-8")
        prediction = pred_path.read_text(encoding="utf-8") if pred_path.exists() else ""

        # Strict scores
        ref_strict = normalize_text_strict(reference)
        pred_strict = normalize_text_strict(prediction)

        # Normalized scores
        ref_norm = normalize_text(reference)
        pred_norm = normalize_text(prediction)

        rows.append(
            {
                "model": model_name,
                "split": split,
                "item": image_path.stem,
                "cer_strict": round(error_rate(ref_strict, pred_strict), 6),
                "wer_strict": round(error_rate(ref_strict.split(), pred_strict.split()), 6),
                "cer_norm": round(error_rate(ref_norm, pred_norm), 6),
                "wer_norm": round(error_rate(ref_norm.split(), pred_norm.split()), 6),
                "line_exact_norm": round(line_exact(ref_norm, pred_norm), 6),
                "ref_chars": len(ref_norm),
                "pred_chars": len(pred_norm),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    summary_rows = []
    for split in ["typed", "hand-written", "all"]:
        selected = (
            rows if split == "all"
            else [r for r in rows if r["split"] == split]
        )
        if not selected:
            continue
        summary_rows.append(
            {
                "model": model_name,
                "split": split,
                "mean_cer_norm": round(
                    sum(float(r["cer_norm"]) for r in selected) / len(selected), 6
                ),
                "mean_wer_norm": round(
                    sum(float(r["wer_norm"]) for r in selected) / len(selected), 6
                ),
                "mean_line_exact_norm": round(
                    sum(float(r["line_exact_norm"]) for r in selected) / len(selected), 6
                ),
                "items": len(selected),
            }
        )

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(output_root / "summary.csv")


# ── Helpers ──────────────────────────────────────────────────────────


def bench_items(root: Path) -> list[Path]:
    """Discover all ``*/*.jpg`` under *root*, sorted by split then number."""
    return sorted(
        root.glob("*/*.jpg"),
        key=lambda p: (p.parent.name, int(p.stem)),
    )


def repo_path(path_str: str, repo_root: Path | None = None) -> Path:
    """Resolve *path_str* relative to the repo root if it is not absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / path


def serializable(value: Any) -> Any:
    """Convert a pydantic/dataclass/object to JSON-safe dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    return str(value)
