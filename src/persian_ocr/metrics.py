"""Shared recognition metrics and aggregate summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import random
import statistics
from typing import Any, Optional

from rapidfuzz.distance import Levenshtein

from .normalization import (
    ARABIC_KAF,
    ARABIC_YEH,
    PERSIAN_KAF,
    PERSIAN_YEH,
    ZWNJ,
    diagnostic_units,
    graphemes,
    normalize_fa,
    normalize_transport,
)


def edit_distance(left: Sequence[Any] | str, right: Sequence[Any] | str) -> int:
    return int(Levenshtein.distance(left, right))


def cer(reference: str, prediction: str, *, unit: str = "grapheme") -> float:
    if unit not in {"grapheme", "codepoint"}:
        raise ValueError(f"Unknown CER unit: {unit}")
    ref_units = graphemes(reference) if unit == "grapheme" else list(reference or "")
    hyp_units = graphemes(prediction) if unit == "grapheme" else list(prediction or "")
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return edit_distance(ref_units, hyp_units) / len(ref_units)


def wer(reference: str, prediction: str) -> float:
    ref_words = (reference or "").split()
    hyp_words = (prediction or "").split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def bag_of_words_wer(reference: str, prediction: str) -> float:
    """Order-insensitive token error rate, used only as an order diagnostic."""
    reference_counts = Counter((reference or "").split())
    prediction_counts = Counter((prediction or "").split())
    reference_total = sum(reference_counts.values())
    if not reference_total:
        return 0.0 if not prediction_counts else 1.0
    return sum(
        abs(reference_counts[token] - prediction_counts[token])
        for token in reference_counts.keys() | prediction_counts.keys()
    ) / reference_total


def token_faithfulness(reference: str, prediction: str) -> dict[str, float | int]:
    """Report omission/insertion balance without hiding repeated OCR tokens."""
    reference_counts = Counter((reference or "").split())
    prediction_counts = Counter((prediction or "").split())
    true_positive = sum((reference_counts & prediction_counts).values())
    omitted = sum((reference_counts - prediction_counts).values())
    inserted = sum((prediction_counts - reference_counts).values())
    precision = true_positive / (true_positive + inserted) if true_positive + inserted else 1.0
    recall = true_positive / (true_positive + omitted) if true_positive + omitted else 1.0
    return {
        "token_true_positive": true_positive,
        "omitted_tokens": omitted,
        "inserted_tokens": inserted,
        "duplicated_tokens": inserted,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
    }


def exact_token_rate(reference: str, prediction: str) -> float:
    """Aligned exact-token accuracy; page/line exactness is reported separately."""
    reference_words = (reference or "").split()
    prediction_words = (prediction or "").split()
    if not reference_words:
        return 1.0 if not prediction_words else 0.0
    aligned = _aligned_units(reference, prediction, lambda value: value.split())
    return sum(ref == hyp for ref, hyp in aligned if ref is not None) / len(reference_words)


def _aligned_units(
    reference: str,
    prediction: str,
    unit_fn=graphemes,
) -> list[tuple[Optional[str], Optional[str]]]:
    left, right = unit_fn(reference), unit_fn(prediction)
    aligned: list[tuple[Optional[str], Optional[str]]] = []
    for tag, i1_raw, i2_raw, j1_raw, j2_raw in Levenshtein.opcodes(left, right):
        i1, i2, j1, j2 = map(int, (i1_raw, i2_raw, j1_raw, j2_raw))
        if tag in {"equal", "replace"}:
            width = max(i2 - i1, j2 - j1)
            aligned.extend(
                zip(
                    left[i1:i2] + [None] * (width - (i2 - i1)),
                    right[j1:j2] + [None] * (width - (j2 - j1)),
                )
            )
        elif tag == "delete":
            aligned.extend((item, None) for item in left[i1:i2])
        else:
            aligned.extend((None, item) for item in right[j1:j2])
    return aligned


def edit_statistics(reference: str, prediction: str) -> dict[str, int]:
    aligned = _aligned_units(reference, prediction, graphemes)
    substitutions = sum(
        1 for ref, hyp in aligned if ref is not None and hyp is not None and ref != hyp
    )
    deletions = sum(1 for ref, hyp in aligned if ref is not None and hyp is None)
    insertions = sum(1 for ref, hyp in aligned if ref is None and hyp is not None)
    return {
        "ref_graphemes": sum(ref is not None for ref, _ in aligned),
        "hyp_graphemes": sum(hyp is not None for _, hyp in aligned),
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "edit_distance": insertions + deletions + substitutions,
    }


def orthographic_diagnostics(
    reference: str, prediction: str
) -> dict[str, Optional[float] | int]:
    aligned = _aligned_units(reference, prediction, diagnostic_units)
    output: dict[str, Optional[float] | int] = {}
    for name, target, alternate in (
        ("yeh", PERSIAN_YEH, ARABIC_YEH),
        ("kaf", PERSIAN_KAF, ARABIC_KAF),
    ):
        ref_count = sum(ref == target for ref, _ in aligned)
        correct = sum(ref == target and hyp == target for ref, hyp in aligned)
        substitutions = sum(
            ref == target and hyp not in {target, None} for ref, hyp in aligned
        )
        deletions = sum(ref == target and hyp is None for ref, hyp in aligned)
        insertions = sum(
            ref is None and hyp in {target, alternate} for ref, hyp in aligned
        )
        output.update(
            {
                f"{name}_ref_count": ref_count,
                f"{name}_correct": correct,
                f"{name}_substitutions": substitutions,
                f"{name}_deletions": deletions,
                f"{name}_insertions": insertions,
                f"{name}_recall": correct / ref_count if ref_count else None,
            }
        )
    true_positive = sum(ref == ZWNJ and hyp == ZWNJ for ref, hyp in aligned)
    false_positive = sum(ref != ZWNJ and hyp == ZWNJ for ref, hyp in aligned)
    false_negative = sum(ref == ZWNJ and hyp != ZWNJ for ref, hyp in aligned)
    output.update(
        zwnj_ref_count=true_positive + false_negative,
        zwnj_pred_count=true_positive + false_positive,
        zwnj_precision=(
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        zwnj_recall=(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        zwnj_f1=(
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else None
        ),
    )
    return output


def unicode_variant_diagnostics(
    reference: str, prediction: str
) -> dict[str, Optional[float] | int]:
    aligned = _aligned_units(reference, prediction, diagnostic_units)
    output_yeh = [hyp for _, hyp in aligned if hyp in {PERSIAN_YEH, ARABIC_YEH}]
    output_kaf = [hyp for _, hyp in aligned if hyp in {PERSIAN_KAF, ARABIC_KAF}]
    return {
        "output_yeh_count": len(output_yeh),
        "persian_yeh_fraction": (
            output_yeh.count(PERSIAN_YEH) / len(output_yeh) if output_yeh else None
        ),
        "output_kaf_count": len(output_kaf),
        "persian_kaf_fraction": (
            output_kaf.count(PERSIAN_KAF) / len(output_kaf) if output_kaf else None
        ),
    }


def punctuation_diagnostics(
    reference: str, prediction: str
) -> dict[str, dict[str, int]]:
    aligned = _aligned_units(reference, prediction, diagnostic_units)
    return {
        mark: {
            "ref": sum(ref == mark for ref, _ in aligned),
            "correct": sum(ref == mark and hyp == mark for ref, hyp in aligned),
            "missed": sum(ref == mark and hyp != mark for ref, hyp in aligned),
            "inserted": sum(ref != mark and hyp == mark for ref, hyp in aligned),
        }
        for mark in ("،", "؛", "؟", "«", "»", "(", ")")
    }


def score_text(reference: str, prediction: str) -> dict[str, Any]:
    """Return the complete shared metric contract for one prediction."""
    ref_strict = normalize_fa(reference, "strict")
    ref_canonical = normalize_fa(reference, "canonical")
    ref_search = normalize_fa(reference, "search")
    hyp_strict = normalize_fa(prediction, "strict")
    hyp_canonical = normalize_fa(prediction, "canonical")
    hyp_search = normalize_fa(prediction, "search")
    edits = edit_statistics(ref_canonical, hyp_canonical)
    canonical_wer = wer(ref_canonical, hyp_canonical)
    bow_wer = bag_of_words_wer(ref_canonical, hyp_canonical)
    return {
        "raw_unicode_cer": round(cer(ref_strict, hyp_strict, unit="codepoint"), 6),
        "persian_normalized_cer": round(cer(ref_canonical, hyp_canonical), 6),
        "raw_unicode_wer": round(wer(ref_strict, hyp_strict), 6),
        "persian_normalized_wer": round(canonical_wer, 6),
        "cer_codepoint_strict": round(cer(ref_strict, hyp_strict, unit="codepoint"), 6),
        "cer_grapheme_strict": round(cer(ref_strict, hyp_strict), 6),
        "cer_grapheme_canonical": round(cer(ref_canonical, hyp_canonical), 6),
        "cer_grapheme_search": round(cer(ref_search, hyp_search), 6),
        "wer_canonical": round(canonical_wer, 6),
        "exact_token_rate": round(exact_token_rate(ref_canonical, hyp_canonical), 6),
        "exact_line": ref_canonical == hyp_canonical,
        "exact_page": ref_canonical == hyp_canonical,
        "faithfulness": token_faithfulness(ref_canonical, hyp_canonical),
        "reading_order": {
            "sequential_wer": round(canonical_wer, 6),
            "bag_of_words_wer": round(bow_wer, 6),
            "order_gap": round(max(0.0, canonical_wer - bow_wer), 6),
        },
        "canonical_ref_graphemes": edits["ref_graphemes"],
        "canonical_hyp_graphemes": edits["hyp_graphemes"],
        "canonical_insertions": edits["insertions"],
        "canonical_deletions": edits["deletions"],
        "canonical_substitutions": edits["substitutions"],
        "canonical_edit_distance": edits["edit_distance"],
        "orthographic": orthographic_diagnostics(ref_canonical, hyp_canonical),
        "unicode_variants": unicode_variant_diagnostics(
            normalize_transport(reference), normalize_transport(prediction)
        ),
        "punctuation": punctuation_diagnostics(ref_canonical, hyp_canonical),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be between zero and one")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def bootstrap_ci(
    values: Sequence[float], *, iterations: int = 10_000, seed: int = 20260712
) -> Optional[list[float]]:
    if len(values) < 2:
        return None
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations)
    ]
    return [round(percentile(samples, 0.025), 6), round(percentile(samples, 0.975), 6)]


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
        metrics = record.get("metrics")
        if isinstance(metrics, Mapping) and name in metrics:
            return metrics[name]
        if "." in name:
            current: Any = metrics if isinstance(metrics, Mapping) else record
            for part in name.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    return default
                current = current[part]
            return current
        return default
    return getattr(record, name, default)


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [value for value in values if value is not None]
    return round(statistics.mean(finite), 6) if finite else None


def summarize_records(records: Sequence[Any]) -> dict[str, object]:
    """Summarize either legacy result dataclasses or v2 result mappings."""
    successful = [
        record
        for record in records
        if not _field(record, "error") and _field(record, "status", "ok") == "ok"
    ]
    values = [_field(record, "cer_grapheme_canonical") for record in successful]
    valid_values = [value for value in values if value is not None]
    edit_distance_total = sum(
        _field(record, "canonical_edit_distance", 0) or 0 for record in successful
    )
    ref_total = sum(
        _field(record, "canonical_ref_graphemes", 0) or 0 for record in successful
    )
    return {
        "n_runs": len(records),
        "n_ok": len(successful),
        "n_err": len(records) - len(successful),
        "macro_page_CER_canonical": _safe_mean(values),
        "median_page_CER_canonical": (
            round(statistics.median(valid_values), 6) if valid_values else None
        ),
        "mean_grapheme_CER_strict": _safe_mean(
            [_field(record, "cer_grapheme_strict") for record in successful]
        ),
        "mean_WER_canonical": _safe_mean(
            [_field(record, "wer_canonical") for record in successful]
        ),
        "mean_raw_unicode_CER": _safe_mean(
            [_field(record, "raw_unicode_cer") for record in successful]
        ),
        "mean_raw_unicode_WER": _safe_mean(
            [_field(record, "raw_unicode_wer") for record in successful]
        ),
        "mean_exact_token_rate": _safe_mean(
            [_field(record, "exact_token_rate") for record in successful]
        ),
        "exact_line_rate": _safe_mean(
            [1.0 if _field(record, "exact_line") else 0.0 for record in successful]
        ),
        "mean_faithfulness_precision": _safe_mean(
            [_field(record, "faithfulness.precision") for record in successful]
        ),
        "mean_faithfulness_recall": _safe_mean(
            [_field(record, "faithfulness.recall") for record in successful]
        ),
        "mean_faithfulness_f1": _safe_mean(
            [_field(record, "faithfulness.f1") for record in successful]
        ),
        "mean_bag_of_words_WER": _safe_mean(
            [_field(record, "reading_order.bag_of_words_wer") for record in successful]
        ),
        "mean_reading_order_gap": _safe_mean(
            [_field(record, "reading_order.order_gap") for record in successful]
        ),
        "micro_corpus_CER_canonical": (
            round(edit_distance_total / ref_total, 6) if ref_total else None
        ),
        "page_bootstrap_95ci": bootstrap_ci(valid_values),
    }


def metadata_breakdowns(records: Sequence[Any]) -> dict[str, dict[str, object]]:
    breakdowns: dict[str, dict[str, object]] = {}
    for key in (
        "font",
        "font_size",
        "background",
        "capture",
        "layout",
        "degradations",
        "mixed_language",
        "handwriting_support",
    ):
        values: dict[str, list[Any]] = {}
        for record in records:
            metadata = _field(record, "page_metadata", {}) or _field(record, "metadata", {})
            raw = metadata.get(key) if isinstance(metadata, Mapping) else None
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if item is not None:
                    values.setdefault(str(item), []).append(record)
        if values:
            breakdowns[key] = {
                value: summarize_records(group) for value, group in sorted(values.items())
            }
    return breakdowns


__all__ = [
    "bootstrap_ci",
    "bag_of_words_wer",
    "cer",
    "edit_distance",
    "edit_statistics",
    "exact_token_rate",
    "metadata_breakdowns",
    "orthographic_diagnostics",
    "percentile",
    "punctuation_diagnostics",
    "score_text",
    "summarize_records",
    "token_faithfulness",
    "unicode_variant_diagnostics",
    "wer",
]
