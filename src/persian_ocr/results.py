"""
Result persistence — append-only JSONL + CSV aggregation.

Each image produces one JSONL record written immediately so the benchmark
is resumable.  After a run, ``scores.csv`` and ``summary.csv`` are
generated from the JSONL file.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from persian_ocr.normalization import normalize_persian, normalize_strict
from persian_ocr.scoring import (
    character_error_rate,
    line_exact_match,
    word_error_rate,
)

SCORES_FIELDS = [
    "model",
    "sample_id",
    "split",
    "cer_strict",
    "wer_strict",
    "cer_norm",
    "wer_norm",
    "line_exact_norm",
    "ref_chars",
    "pred_chars",
    "latency_seconds",
    "error",
]

SUMMARY_FIELDS = [
    "model",
    "split",
    "mean_cer_norm",
    "mean_wer_norm",
    "mean_line_exact_norm",
    "items",
]


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to *path*, flushing immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_completed(path: Path) -> set[str]:
    """Return set of sample_ids that completed without error."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("error"):
                done.add(rec["sample_id"])
    return done


def build_record(
    model_id: str,
    sample_id: str,
    split: str,
    reference: str,
    prediction_raw: str,
    latency: float,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a complete record with both strict and normalised scores."""
    ref_strict = normalize_strict(reference)
    pred_strict = normalize_strict(prediction_raw)
    ref_norm = normalize_persian(reference)
    pred_norm = normalize_persian(prediction_raw)

    return {
        "model_id": model_id,
        "sample_id": sample_id,
        "split": split,
        "reference": reference,
        "prediction_raw": prediction_raw,
        "latency_seconds": round(latency, 3),
        "cer_strict": round(character_error_rate(ref_strict, pred_strict), 6),
        "wer_strict": round(word_error_rate(ref_strict, pred_strict), 6),
        "cer_norm": round(character_error_rate(ref_norm, pred_norm), 6),
        "wer_norm": round(word_error_rate(ref_norm, pred_norm), 6),
        "line_exact_norm": round(line_exact_match(ref_norm, pred_norm), 6),
        "ref_chars": len(ref_norm),
        "pred_chars": len(pred_norm),
        "error": error,
    }


def write_scores_csv(predictions_path: Path) -> None:
    """Regenerate ``scores.csv`` from the JSONL predictions file."""
    if not predictions_path.exists():
        return
    records: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    out_dir = predictions_path.parent
    rows = []
    for rec in records:
        rows.append(
            {
                "model": rec["model_id"],
                "sample_id": rec["sample_id"],
                "split": rec["split"],
                "cer_strict": rec["cer_strict"],
                "wer_strict": rec["wer_strict"],
                "cer_norm": rec["cer_norm"],
                "wer_norm": rec["wer_norm"],
                "line_exact_norm": rec["line_exact_norm"],
                "ref_chars": rec["ref_chars"],
                "pred_chars": rec["pred_chars"],
                "latency_seconds": rec["latency_seconds"],
                "error": rec.get("error"),
            }
        )

    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    summary_rows = []
    for split in ["typed", "hand-written", "all"]:
        selected = (
            rows
            if split == "all"
            else [r for r in rows if r["split"] == split]
        )
        if not selected or all(r.get("error") for r in selected):
            continue
        valid = [r for r in selected if not r.get("error")]
        if not valid:
            continue
        summary_rows.append(
            {
                "model": model_id_from_path(predictions_path),
                "split": split,
                "mean_cer_norm": round(
                    sum(float(r["cer_norm"]) for r in valid) / len(valid), 6
                ),
                "mean_wer_norm": round(
                    sum(float(r["wer_norm"]) for r in valid) / len(valid), 6
                ),
                "mean_line_exact_norm": round(
                    sum(float(r["line_exact_norm"]) for r in valid) / len(valid), 6
                ),
                "items": len(valid),
            }
        )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def model_id_from_path(predictions_path: Path) -> str:
    """Infer model ID from predictions path parent directory name."""
    return predictions_path.parent.name
