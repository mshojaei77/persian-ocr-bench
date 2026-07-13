"""Validated reporting for Persian OCR screening and final benchmarks.

The reporting boundary is intentionally stricter than the model adapters.  A
JSON file is never comparable merely because it contains a CER value: its run,
protocol, dataset, model, sample coverage, and explicit slice labels must all
validate first.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

try:
    from .paths import REPO_ROOT
except ImportError:  # Temporary compatibility while the package is bootstrapped.
    REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from .artifacts import (
        ArtifactValidationError as CoreArtifactValidationError,
        validate_artifact as validate_core_artifact,
    )
except ImportError:  # The reporting module remains inspectable during bootstrap.
    CoreArtifactValidationError = ValueError  # type: ignore[assignment,misc]
    validate_core_artifact = None


BENCHMARK_SCHEMA = "persian_ocr_benchmark_v2"
RUN_IDENTITY_SCHEMA = "persian_ocr_run_identity_v2"
PROTOCOL_IDENTITY_SCHEMA = "persian_ocr_protocol_identity_v2"
SCREENING_REPORT_SCHEMA = "persian_ocr_screening_report_v1"
FINAL_REPORT_SCHEMA = "persian_ocr_final_leaderboard_v1"
DECISIONS = ("Advance", "Hold", "Reject", "Blocked")
OK_STATUSES = {"ok", "success", "completed"}
ERROR_STATUSES = {"error", "failed", "timeout", "blocked", "skipped"}
SCREENING_PHASES = {"phase1", "phase_1", "phase1_screening", "screening", "smoke20"}
FINAL_PHASES = {"final", "phase2", "phase_2", "large_benchmark", "held_out"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class ValidatedArtifact:
    path: Path
    payload: dict[str, Any]
    issues: list[ValidationIssue] = field(default_factory=list)
    model_id: str | None = None
    model_class: str | None = None
    phase: str | None = None
    benchmark_id: str | None = None
    protocol_track: str | None = None
    protocol_version: str | None = None
    protocol_hash: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    run_hash: str | None = None
    config_hash: str | None = None
    environment_hash: str | None = None
    expected: int = 0
    total: int = 0
    ok: int = 0
    error: int = 0
    coverage: float = 0.0
    attempt_coverage: float = 0.0
    cer: float | None = None
    wer: float | None = None
    micro_cer: float | None = None
    ci95: list[float] | None = None
    mean_seconds: float | None = None
    median_seconds: float | None = None

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def complete(self) -> bool:
        return (
            self.valid
            and self.expected > 0
            and self.total == self.expected
            and self.ok == self.expected
            and self.error == 0
        )


def _issue(record: ValidatedArtifact, severity: str, code: str, message: str) -> None:
    record.issues.append(ValidationIssue(severity, code, message))


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value.lower()))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = mapping
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def _metric(mapping: Mapping[str, Any], *paths: str) -> float | None:
    return finite_number(_nested(mapping, *paths))


def _first_number(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _logical_path_ok(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if re.match(r"^[A-Za-z]:", value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def safe_source(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def bootstrap_ci(
    values: Sequence[float], *, iterations: int = 10_000, seed: int = 20260713
) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iterations)
    ]
    return [round(percentile(samples, 0.025), 6), round(percentile(samples, 0.975), 6)]


def _result_status(result: Mapping[str, Any]) -> str:
    explicit = str(result.get("status") or "").strip().lower()
    if explicit:
        return explicit
    return "error" if result.get("error") else "ok"


def _result_metric(result: Mapping[str, Any], metric: str) -> float | None:
    metrics = _mapping(result.get("metrics"))
    if metric == "cer":
        return _first_number(
            _metric(
                metrics,
                "cer_canonical",
                "cer_grapheme_canonical",
                "canonical.cer",
                "canonical.macro_page_cer",
            ),
            _metric(result, "cer_grapheme_canonical", "cer_canonical"),
        )
    if metric == "wer":
        return _first_number(
            _metric(metrics, "wer_canonical", "canonical.wer"),
            _metric(result, "wer_canonical"),
        )
    return None


def _summary_metric(summary: Mapping[str, Any], metric: str) -> float | None:
    metrics = _mapping(summary.get("metrics"))
    legacy = _mapping(summary.get("primary_results"))
    if metric == "cer":
        return _first_number(
            _metric(
                metrics,
                "macro_page_cer_canonical",
                "macro_page_CER_canonical",
                "canonical.macro_page_cer",
            ),
            _metric(legacy, "macro_page_CER_canonical", "macro_CER_canonical"),
        )
    if metric == "wer":
        return _first_number(
            _metric(
                metrics,
                "mean_wer_canonical",
                "mean_WER_canonical",
                "canonical.mean_wer",
            ),
            _metric(legacy, "mean_WER_canonical", "mean_WER"),
        )
    if metric == "micro_cer":
        return _first_number(
            _metric(
                metrics, "micro_corpus_cer_canonical", "micro_corpus_CER_canonical"
            ),
            _metric(legacy, "micro_corpus_CER_canonical"),
        )
    return None


def _validate_hash_mapping(
    record: ValidatedArtifact,
    mapping: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(mapping, dict) or (not mapping and not allow_empty):
        _issue(
            record,
            "error",
            "dataset_hashes_missing",
            f"run_identity.dataset.{field_name} must be a non-empty mapping",
        )
        return
    for logical_path, digest in mapping.items():
        if not _logical_path_ok(logical_path):
            _issue(
                record,
                "error",
                "dataset_path_invalid",
                f"{field_name} contains unsafe logical path {logical_path!r}",
            )
        if not is_sha256(digest):
            _issue(
                record,
                "error",
                "dataset_hash_invalid",
                f"{field_name}[{logical_path!r}] is not SHA-256",
            )


def validate_artifact(
    path: Path,
    *,
    expected_model_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> ValidatedArtifact:
    """Load and validate one v2 benchmark artifact without trusting its summary."""
    if payload is None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record = ValidatedArtifact(path=path, payload={})
            _issue(record, "error", "artifact_invalid_json", str(exc))
            return record
        payload = loaded if isinstance(loaded, dict) else {}
    record = ValidatedArtifact(path=path, payload=payload)
    if not isinstance(payload, dict) or not payload:
        _issue(
            record,
            "error",
            "artifact_invalid_root",
            "artifact root must be a JSON object",
        )
        return record

    schema = payload.get("schema")
    if schema != BENCHMARK_SCHEMA:
        untrusted_identity = _mapping(payload.get("run_identity"))
        untrusted_model = _mapping(untrusted_identity.get("model"))
        record.model_id = (
            untrusted_model.get("id")
            if isinstance(untrusted_model.get("id"), str)
            else None
        )
        record.model_class = (
            untrusted_model.get("class")
            if isinstance(untrusted_model.get("class"), str)
            else None
        )
        _issue(
            record,
            "error",
            "artifact_schema_incompatible",
            f"schema={schema!r}; expected {BENCHMARK_SCHEMA!r}",
        )
        # Legacy envelopes have materially different count and metric scopes.
        # Stop here rather than producing a noisy cascade or accidentally
        # treating a recognizable legacy field as v2 evidence.
        return record
    if validate_core_artifact is not None:
        try:
            validate_core_artifact(payload)
        except CoreArtifactValidationError as exc:
            errors = getattr(exc, "errors", [str(exc)])
            for message in errors:
                _issue(record, "error", "core_contract_invalid", str(message))

    identity = _mapping(payload.get("run_identity"))
    if identity.get("schema") != RUN_IDENTITY_SCHEMA:
        _issue(
            record,
            "error",
            "run_identity_schema_incompatible",
            f"run_identity.schema={identity.get('schema')!r}",
        )

    record.run_hash = (
        payload.get("run_id") if isinstance(payload.get("run_id"), str) else None
    )
    identity_digest = identity.get("digest")
    if not is_sha256(record.run_hash) or record.run_hash != identity_digest:
        _issue(
            record,
            "error",
            "run_hash_invalid",
            "run_id must be a SHA-256 equal to run_identity.digest",
        )

    protocol = _mapping(identity.get("protocol"))
    record.benchmark_id = (
        protocol.get("id") if isinstance(protocol.get("id"), str) else None
    )
    record.phase = str(protocol.get("phase") or "").strip().lower() or None
    record.protocol_track = str(protocol.get("track") or "").strip() or None
    record.protocol_version = str(protocol.get("version") or "").strip() or None
    if not all(
        (
            record.benchmark_id,
            record.phase,
            record.protocol_track,
            record.protocol_version,
        )
    ):
        _issue(
            record,
            "error",
            "protocol_identity_incomplete",
            "protocol requires id, phase, track, and version",
        )
    explicit_protocol_hash = protocol.get("digest") or protocol.get("sha256")
    if explicit_protocol_hash is not None and not is_sha256(explicit_protocol_hash):
        _issue(
            record, "error", "protocol_hash_invalid", "protocol digest is not SHA-256"
        )
    protocol_for_hash = {
        key: value for key, value in protocol.items() if key not in {"digest", "sha256"}
    }
    computed_protocol_hash = canonical_sha256(protocol_for_hash)
    if (
        is_sha256(explicit_protocol_hash)
        and explicit_protocol_hash != computed_protocol_hash
    ):
        _issue(
            record,
            "error",
            "protocol_hash_mismatch",
            "protocol digest does not match canonical protocol identity",
        )
    record.protocol_hash = explicit_protocol_hash or computed_protocol_hash

    dataset = _mapping(identity.get("dataset"))
    record.dataset_id = (
        dataset.get("id") if isinstance(dataset.get("id"), str) else None
    )
    try:
        record.expected = int(dataset.get("n_samples") or 0)
    except (TypeError, ValueError):
        record.expected = 0
    if not record.dataset_id or record.expected <= 0:
        _issue(
            record,
            "error",
            "dataset_identity_incomplete",
            "dataset requires id and positive n_samples",
        )
    if not is_sha256(dataset.get("manifest_sha256")):
        _issue(
            record,
            "error",
            "dataset_manifest_hash_invalid",
            "dataset.manifest_sha256 is required",
        )
    _validate_hash_mapping(
        record, dataset.get("reference_corpora_sha256"), "reference_corpora_sha256"
    )
    _validate_hash_mapping(record, dataset.get("images_sha256"), "images_sha256")
    images_sha = dataset.get("images_sha256")
    if (
        isinstance(images_sha, dict)
        and record.expected
        and len(images_sha) != record.expected
    ):
        _issue(
            record,
            "error",
            "dataset_image_count_mismatch",
            f"dataset has {len(images_sha)} image hashes, expected {record.expected}",
        )
    explicit_dataset_hash = dataset.get("digest")
    if explicit_dataset_hash is not None and not is_sha256(explicit_dataset_hash):
        _issue(record, "error", "dataset_hash_invalid", "dataset digest is not SHA-256")
    if dataset.get("dataset_sha256") is not None and not is_sha256(
        dataset.get("dataset_sha256")
    ):
        _issue(
            record,
            "error",
            "dataset_content_hash_invalid",
            "dataset.dataset_sha256 is not SHA-256",
        )
    dataset_for_hash = {key: value for key, value in dataset.items() if key != "digest"}
    computed_dataset_hash = canonical_sha256(dataset_for_hash)
    record.dataset_hash = explicit_dataset_hash or computed_dataset_hash

    model = _mapping(identity.get("model"))
    record.model_id = model.get("id") if isinstance(model.get("id"), str) else None
    record.model_class = (
        model.get("class") if isinstance(model.get("class"), str) else None
    )
    if not record.model_id or not record.model_class:
        _issue(
            record, "error", "model_identity_incomplete", "model requires id and class"
        )
    if expected_model_id is not None and record.model_id != expected_model_id:
        _issue(
            record,
            "error",
            "artifact_model_mismatch",
            f"model.id={record.model_id!r}; expected {expected_model_id!r}",
        )
    record.config_hash = canonical_sha256(_mapping(identity.get("config")))
    record.environment_hash = canonical_sha256(_mapping(identity.get("environment")))

    results = payload.get("results")
    if not isinstance(results, list):
        _issue(record, "error", "artifact_results_missing", "results must be a list")
        results = []
    sample_ids: set[str] = set()
    successful_cer: list[float] = []
    successful_wer: list[float] = []
    successful_seconds: list[float] = []
    status_counts: Counter[str] = Counter()
    ok = error = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            _issue(
                record, "error", "result_invalid", f"results[{index}] must be an object"
            )
            error += 1
            continue
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            _issue(
                record,
                "error",
                "sample_id_missing",
                f"results[{index}] has no sample_id",
            )
        elif sample_id in sample_ids:
            _issue(
                record,
                "error",
                "sample_id_duplicate",
                f"duplicate sample_id {sample_id!r}",
            )
        else:
            sample_ids.add(sample_id)
        for field_name in ("split", "content_type"):
            if not isinstance(result.get(field_name), str) or not result.get(
                field_name
            ):
                _issue(
                    record,
                    "error",
                    "slice_label_missing",
                    f"results[{index}].{field_name} is required",
                )
        if not _logical_path_ok(result.get("image")):
            _issue(
                record,
                "error",
                "result_image_path_invalid",
                f"results[{index}].image must be a relative POSIX logical path",
            )
        seconds = finite_number(result.get("seconds"))
        if seconds is None or seconds < 0:
            _issue(
                record,
                "error",
                "result_seconds_invalid",
                f"results[{index}].seconds must be finite and non-negative",
            )
        status = _result_status(result)
        if status in OK_STATUSES:
            status_counts["ok"] += 1
            ok += 1
            if not isinstance(result.get("reference"), str) or not isinstance(
                result.get("prediction"), str
            ):
                _issue(
                    record,
                    "error",
                    "result_text_missing",
                    f"successful results[{index}] requires reference and prediction",
                )
            cer_value = _result_metric(result, "cer")
            if cer_value is None or cer_value < 0:
                _issue(
                    record,
                    "error",
                    "result_cer_missing",
                    f"successful results[{index}] requires non-negative canonical CER",
                )
            else:
                successful_cer.append(cer_value)
            wer_value = _result_metric(result, "wer")
            if wer_value is not None and wer_value >= 0:
                successful_wer.append(wer_value)
            if seconds is not None and seconds >= 0:
                successful_seconds.append(seconds)
        elif status in ERROR_STATUSES:
            status_counts["skipped" if status == "skipped" else "error"] += 1
            error += 1
            if not result.get("error"):
                _issue(
                    record,
                    "error",
                    "result_error_missing",
                    f"failed results[{index}] requires error details",
                )
        else:
            status_counts["error"] += 1
            error += 1
            _issue(
                record,
                "error",
                "result_status_invalid",
                f"results[{index}].status={status!r}",
            )

    record.total = len(results)
    record.ok = ok
    record.error = error
    record.coverage = round(ok / record.expected, 6) if record.expected else 0.0
    record.attempt_coverage = (
        round(record.total / record.expected, 6) if record.expected else 0.0
    )
    if record.total != record.expected:
        _issue(
            record,
            "warning",
            "artifact_incomplete",
            f"artifact contains {record.total}/{record.expected} expected sample rows",
        )

    summary = _mapping(payload.get("summary"))
    summary_counts = _mapping(summary.get("counts"))
    expected_counts = {
        "total": record.total,
        "ok": status_counts["ok"],
        "error": status_counts["error"],
        "skipped": status_counts["skipped"],
    }
    for name, derived in expected_counts.items():
        if summary_counts.get(name) != derived:
            _issue(
                record,
                "error",
                "summary_count_mismatch",
                f"summary.counts.{name}={summary_counts.get(name)!r}; derived {derived}",
            )
    summary_model = _mapping(summary.get("model"))
    if (
        summary_model.get("id") != record.model_id
        or summary_model.get("class") != record.model_class
    ):
        _issue(
            record,
            "error",
            "summary_model_mismatch",
            "summary.model differs from run_identity.model",
        )
    summary_protocol = _mapping(summary.get("protocol"))
    for name, value in (
        ("id", record.benchmark_id),
        ("phase", record.phase),
        ("track", record.protocol_track),
    ):
        actual = (
            str(summary_protocol.get(name) or "").strip().lower()
            if name == "phase"
            else summary_protocol.get(name)
        )
        if actual != value:
            _issue(
                record,
                "error",
                "summary_protocol_mismatch",
                f"summary.protocol.{name} differs from run identity",
            )
    summary_dataset = _mapping(summary.get("dataset"))
    if (
        summary_dataset.get("id") != record.dataset_id
        or summary_dataset.get("n_samples") != record.expected
    ):
        _issue(
            record,
            "error",
            "summary_dataset_mismatch",
            "summary.dataset differs from run_identity.dataset",
        )
    if not isinstance(summary.get("metrics"), dict) or not isinstance(
        summary.get("operations"), dict
    ):
        _issue(
            record,
            "error",
            "summary_sections_missing",
            "summary requires metrics and operations objects",
        )

    record.cer = (
        round(statistics.fmean(successful_cer), 6)
        if successful_cer
        else _summary_metric(summary, "cer")
    )
    record.wer = (
        round(statistics.fmean(successful_wer), 6)
        if successful_wer
        else _summary_metric(summary, "wer")
    )
    record.micro_cer = _summary_metric(summary, "micro_cer")
    record.ci95 = bootstrap_ci(successful_cer)
    if record.ci95 is None:
        ci = _nested(
            summary,
            "metrics.page_bootstrap_95ci",
            "metrics.canonical.page_bootstrap_95ci",
        )
        if (
            isinstance(ci, list)
            and len(ci) == 2
            and all(finite_number(value) is not None for value in ci)
        ):
            record.ci95 = [float(ci[0]), float(ci[1])]
    record.mean_seconds = (
        round(statistics.fmean(successful_seconds), 6) if successful_seconds else None
    )
    record.median_seconds = (
        round(statistics.median(successful_seconds), 6) if successful_seconds else None
    )
    return record


def compatibility_key(record: ValidatedArtifact) -> tuple[str, ...]:
    return (
        BENCHMARK_SCHEMA,
        record.phase or "",
        record.benchmark_id or "",
        record.protocol_version or "",
        record.protocol_track or "",
        record.protocol_hash or "",
        record.dataset_id or "",
        record.dataset_hash or "",
    )


def _target_contract(
    records: Sequence[ValidatedArtifact], mode: str
) -> tuple[str, ...] | None:
    phases = SCREENING_PHASES if mode == "screening" else FINAL_PHASES
    keys = [
        compatibility_key(record)
        for record in records
        if record.valid and record.phase in phases
    ]
    if not keys:
        return None
    counts = Counter(keys)
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


def _artifact_row(record: ValidatedArtifact) -> dict[str, Any]:
    ok_results = [
        result
        for result in record.payload.get("results", [])
        if isinstance(result, dict) and _result_status(result) in OK_STATUSES
    ]
    page_cers = sorted(
        value
        for result in ok_results
        if (value := _result_metric(result, "cer")) is not None
    )
    orthographic = [
        result["metrics"]["orthographic"]
        for result in ok_results
        if isinstance(result.get("metrics"), dict)
        and isinstance(result["metrics"].get("orthographic"), dict)
    ]

    def mean_orthographic(name: str) -> float | None:
        values = [item[name] for item in orthographic if isinstance(item.get(name), (int, float))]
        return round(statistics.fmean(values), 6) if values else None

    return {
        "model_id": record.model_id or record.path.stem,
        "capability_class": record.model_class,
        "artifact": safe_source(record.path),
        "benchmark_id": record.benchmark_id,
        "phase": record.phase,
        "protocol_track": record.protocol_track,
        "dataset_id": record.dataset_id,
        "dataset_hash": record.dataset_hash,
        "protocol_hash": record.protocol_hash,
        "run_hash": record.run_hash,
        "config_hash": record.config_hash,
        "environment_hash": record.environment_hash,
        "expected": record.expected,
        "attempted": record.total,
        "ok": record.ok,
        "error": record.error,
        "coverage": record.coverage,
        "attempt_coverage": record.attempt_coverage,
        "macro_page_cer_canonical": record.cer,
        "micro_corpus_cer_canonical": record.micro_cer,
        "mean_wer_canonical": record.wer,
        "page_cer_ci95_low": record.ci95[0] if record.ci95 else None,
        "page_cer_ci95_high": record.ci95[1] if record.ci95 else None,
        "mean_seconds_per_image": record.mean_seconds,
        "median_seconds_per_image": record.median_seconds,
        "p95_seconds_per_image": _nested(record.payload, "operations.p95_seconds_per_page"),
        "p90_page_cer_canonical": (
            round(page_cers[max(0, math.ceil(len(page_cers) * 0.90) - 1)], 6)
            if page_cers
            else None
        ),
        "worst_quartile_page_cer_canonical": (
            round(statistics.fmean(page_cers[math.floor(len(page_cers) * 0.75) :]), 6)
            if page_cers
            else None
        ),
        "exact_page_rate": (
            round(sum(value == 0 for value in page_cers) / len(page_cers), 6)
            if page_cers
            else None
        ),
        "yeh_recall": mean_orthographic("yeh_recall"),
        "kaf_recall": mean_orthographic("kaf_recall"),
        "zwnj_f1": mean_orthographic("zwnj_f1"),
    }


def _explicit_decision(
    record: ValidatedArtifact, overrides: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    model_id = record.model_id or record.path.stem
    raw = overrides.get(model_id)
    reason = None
    if isinstance(raw, dict):
        reason = str(raw.get("reason") or "").strip() or None
        raw = raw.get("decision")
    if raw is None:
        raw = _nested(
            record.payload,
            "screening.decision",
            "summary.screening.decision",
            "decision",
        )
        reason = reason or _nested(
            record.payload,
            "screening.reason",
            "summary.screening.reason",
            "decision_reason",
        )
    if raw is None:
        return None, reason
    normalized = str(raw).strip().lower()
    decision = next((item for item in DECISIONS if item.lower() == normalized), None)
    return decision, str(reason) if reason else None


def _pareto_frontier(records: Sequence[ValidatedArtifact]) -> set[int]:
    candidates = [
        record
        for record in records
        if record.complete
        and record.cer is not None
        and record.mean_seconds is not None
    ]
    frontier: set[int] = set()
    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other.cer is not None
            and other.mean_seconds is not None
            and other.cer <= candidate.cer
            and other.mean_seconds <= candidate.mean_seconds
            and (
                other.cer < candidate.cer or other.mean_seconds < candidate.mean_seconds
            )
            for other in candidates
        )
        if not dominated:
            frontier.add(id(candidate))
    return frontier


def paired_difference(
    baseline: ValidatedArtifact,
    candidate: ValidatedArtifact,
    *,
    practical_threshold: float = 0.01,
) -> dict[str, Any]:
    """Compare candidate-minus-baseline CER on the same successful pages."""
    baseline_values = {
        str(result.get("sample_id")): value
        for result in baseline.payload.get("results", [])
        if isinstance(result, dict)
        and _result_status(result) in OK_STATUSES
        and (value := _result_metric(result, "cer")) is not None
    }
    candidate_values = {
        str(result.get("sample_id")): value
        for result in candidate.payload.get("results", [])
        if isinstance(result, dict)
        and _result_status(result) in OK_STATUSES
        and (value := _result_metric(result, "cer")) is not None
    }
    sample_ids = sorted(baseline_values.keys() & candidate_values.keys())
    differences = [
        candidate_values[sample_id] - baseline_values[sample_id]
        for sample_id in sample_ids
    ]
    mean = round(statistics.fmean(differences), 6) if differences else None
    ci = bootstrap_ci(differences)
    if ci is None:
        conclusion = "insufficient_paired_evidence"
    elif ci[0] > practical_threshold:
        conclusion = "worse"
    elif ci[1] < -practical_threshold:
        conclusion = "better"
    elif ci[0] >= -practical_threshold and ci[1] <= practical_threshold:
        conclusion = "practical_tie"
    else:
        conclusion = "inconclusive"
    return {
        "baseline_model_id": baseline.model_id,
        "n_paired": len(differences),
        "mean_cer_difference": mean,
        "ci95": ci,
        "practical_threshold": practical_threshold,
        "conclusion": conclusion,
    }


def slice_rows(records: Iterable[ValidatedArtifact]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.valid:
            continue
        for result in record.payload.get("results", []):
            if not isinstance(result, dict):
                continue
            grouped[
                (
                    record.model_id or record.path.stem,
                    str(result["split"]),
                    str(result["content_type"]),
                )
            ].append(result)
    rows: list[dict[str, Any]] = []
    for (model_id, split, content_type), results in sorted(grouped.items()):
        ok_results = [
            result for result in results if _result_status(result) in OK_STATUSES
        ]
        cers = [
            value
            for result in ok_results
            if (value := _result_metric(result, "cer")) is not None
        ]
        rows.append(
            {
                "model_id": model_id,
                "split": split,
                "content_type": content_type,
                "expected": len(results),
                "ok": len(ok_results),
                "error": len(results) - len(ok_results),
                "coverage": round(len(ok_results) / len(results), 6)
                if results
                else 0.0,
                "macro_page_cer_canonical": round(statistics.fmean(cers), 6)
                if cers
                else None,
                "page_cer_ci95": bootstrap_ci(cers),
            }
        )
    return rows


def load_decisions(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load decisions {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Decisions file must be a JSON object keyed by model id")
    return payload


def build_report(
    artifact_paths: Sequence[Path],
    *,
    mode: str,
    decision_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"screening", "final"}:
        raise ValueError(f"unknown report mode: {mode}")
    overrides = decision_overrides or {}
    records = [validate_artifact(path) for path in sorted(artifact_paths)]
    target = _target_contract(records, mode)
    allowed_phases = SCREENING_PHASES if mode == "screening" else FINAL_PHASES
    eligible = [
        record
        for record in records
        if record.valid
        and record.phase in allowed_phases
        and target is not None
        and compatibility_key(record) == target
    ]

    duplicates = Counter((record.model_class, record.model_id) for record in eligible)
    duplicate_ids = {key for key, count in duplicates.items() if count > 1}
    eligible_unique = [
        record
        for record in eligible
        if (record.model_class, record.model_id) not in duplicate_ids
    ]
    frontier_by_group: dict[tuple[str | None, str | None], set[int]] = {}
    for group in {
        (record.model_class, record.environment_hash) for record in eligible_unique
    }:
        peers = [
            record
            for record in eligible_unique
            if (record.model_class, record.environment_hash) == group
        ]
        frontier_by_group[group] = _pareto_frontier(peers) if len(peers) >= 2 else set()

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        row = _artifact_row(record)
        reasons: list[str] = []
        if not record.valid:
            reasons.append("artifact validation failed")
        elif record.phase not in allowed_phases:
            reasons.append(f"phase {record.phase!r} is not eligible for {mode}")
        elif target is not None and compatibility_key(record) != target:
            reasons.append(
                "dataset or protocol is incompatible with the selected report contract"
            )
        elif (record.model_class, record.model_id) in duplicate_ids:
            reasons.append(
                "duplicate model runs require explicit run selection; automatic best-run selection is forbidden"
            )

        if mode == "screening":
            explicit, explicit_reason = _explicit_decision(record, overrides)
            if reasons:
                decision = "Blocked"
            elif record.ok == 0 and record.error > 0:
                decision = "Reject"
                reasons.append("all attempted pages failed")
            elif not record.complete:
                decision = "Hold"
                reasons.append("incomplete coverage prevents advancement")
            elif record.cer is None:
                decision = "Hold"
                reasons.append("no comparable canonical CER")
            elif explicit:
                decision = explicit
            elif id(record) in frontier_by_group.get(
                (record.model_class, record.environment_hash), set()
            ):
                decision = "Advance"
                reasons.append(
                    "complete run is on the quality/latency Pareto frontier within its capability class and environment"
                )
            else:
                decision = "Hold"
                reasons.append(
                    "complete evidence retained without a forced small-sample rank"
                )
            if explicit_reason:
                reasons.append(explicit_reason)
            if decision == "Advance" and not record.complete:
                decision = "Hold"
                reasons.append("Advance was downgraded because coverage is incomplete")
            row["decision"] = decision
            row["decision_reasons"] = reasons
            row["validation_issues"] = [asdict(issue) for issue in record.issues]
            rows.append(row)
        else:
            if not reasons and not record.complete:
                reasons.append(
                    "final leaderboard requires complete, failure-free coverage"
                )
            if not reasons and record.cer is None:
                reasons.append("final leaderboard requires canonical CER")
            if reasons:
                excluded.append(
                    {
                        **row,
                        "eligibility": "Excluded",
                        "reasons": reasons,
                        "validation_issues": [asdict(issue) for issue in record.issues],
                    }
                )
            else:
                row["comparison_group"] = record.model_class
                rows.append(row)

    if mode == "screening":
        decision_order = {decision: index for index, decision in enumerate(DECISIONS)}
        rows.sort(
            key=lambda row: (
                decision_order[row["decision"]],
                row.get("capability_class") or "",
                row["model_id"],
            )
        )
    else:
        ranked: list[dict[str, Any]] = []
        record_lookup = {
            (record.model_class, record.model_id): record
            for record in eligible_unique
            if record.complete and record.cer is not None
        }
        for group in sorted({row["comparison_group"] for row in rows}):
            group_rows = [row for row in rows if row["comparison_group"] == group]
            group_rows.sort(
                key=lambda row: (
                    row["macro_page_cer_canonical"] is None,
                    row["macro_page_cer_canonical"]
                    if row["macro_page_cer_canonical"] is not None
                    else math.inf,
                    row["model_id"],
                )
            )
            rank = 1
            previous_record: ValidatedArtifact | None = None
            for row in group_rows:
                current_record = record_lookup[(group, row["model_id"])]
                if previous_record is None:
                    row["paired_comparison"] = {
                        "baseline_model_id": row["model_id"],
                        "n_paired": current_record.expected,
                        "mean_cer_difference": 0.0,
                        "ci95": [0.0, 0.0],
                        "practical_threshold": 0.01,
                        "conclusion": "reference",
                    }
                else:
                    comparison = paired_difference(previous_record, current_record)
                    row["paired_comparison"] = comparison
                    if comparison["conclusion"] == "worse":
                        rank += 1
                row["rank"] = rank
                ranked.append(row)
                previous_record = current_record
        rows = ranked

    compatible_records = [
        record for record in eligible_unique if compatibility_key(record) == target
    ]
    report: dict[str, Any] = {
        "schema": SCREENING_REPORT_SCHEMA
        if mode == "screening"
        else FINAL_REPORT_SCHEMA,
        "mode": mode,
        "report_identity": {
            "target_contract": {
                "schema": target[0],
                "phase": target[1],
                "benchmark_id": target[2],
                "protocol_version": target[3],
                "protocol_track": target[4],
                "protocol_hash": target[5],
                "dataset_id": target[6],
                "dataset_hash": target[7],
            }
            if target
            else None,
            "run_hashes": sorted(
                record.run_hash for record in compatible_records if record.run_hash
            ),
            "artifact_hashes": {
                safe_source(record.path): file_sha256(record.path)
                for record in records
                if record.path.is_file()
            },
        },
        "rows": rows,
        "slices": slice_rows(compatible_records),
        "excluded": excluded,
        "ranking_policy": (
            None
            if mode == "screening"
            else {
                "primary_metric": "macro_page_cer_canonical",
                "paired_unit": "sample_id",
                "practical_cer_threshold": 0.01,
                "tie_rule": "Ranks advance only when the paired 95% interval is wholly worse than the practical threshold; otherwise the result is a practical tie or inconclusive.",
            }
        ),
        "validation_summary": {
            "artifacts_seen": len(records),
            "artifacts_valid": sum(record.valid for record in records),
            "artifacts_compatible": len(eligible),
            "artifacts_unique": len(eligible_unique),
            "artifacts_complete": sum(record.complete for record in eligible),
            "duplicate_runs": len(eligible) - len(eligible_unique),
            "errors": sum(
                issue.severity == "error"
                for record in records
                for issue in record.issues
            ),
            "warnings": sum(
                issue.severity == "warning"
                for record in records
                for issue in record.issues
            ),
        },
    }
    report["report_identity"]["report_sha256"] = canonical_sha256(report)
    return report


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in fields:
                fields.append(field_name)
    if not fields:
        fields = ["model_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in fields} for row in rows
        )


def _markdown(report: Mapping[str, Any]) -> str:
    mode = report["mode"]
    lines = [
        "# Phase 1 screening report"
        if mode == "screening"
        else "# Final compatible leaderboard",
        "",
    ]
    identity = report["report_identity"].get("target_contract")
    if identity:
        lines.extend(
            [
                f"Dataset: `{identity['dataset_id']}` (`{identity['dataset_hash']}`)",
                "",
                f"Protocol: `{identity['benchmark_id']}` / `{identity['protocol_track']}` (`{identity['protocol_hash']}`)",
                "",
            ]
        )
    if mode == "screening":
        lines.extend(
            [
                "This 20-image phase is a viability screen, not a general Persian OCR ranking.",
                "",
                "| Decision | Model | Coverage | Macro CER (95% CI) | P90 CER | Worst-quartile CER | Exact pages | Mean sec/image |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["rows"]:
            ci = (
                "-"
                if row["page_cer_ci95_low"] is None
                else f"{row['page_cer_ci95_low']:.4f}-{row['page_cer_ci95_high']:.4f}"
            )
            cer = (
                "-"
                if row["macro_page_cer_canonical"] is None
                else f"{row['macro_page_cer_canonical']:.4f} ({ci})"
            )
            seconds = (
                "-"
                if row["mean_seconds_per_image"] is None
                else f"{row['mean_seconds_per_image']:.3f}"
            )
            lines.append(
                f"| {row['decision']} | `{row['model_id']}` | {row['ok']}/{row['expected']} | {cer} | {row.get('p90_page_cer_canonical') or '-'} | {row.get('worst_quartile_page_cer_canonical') or '-'} | {row.get('exact_page_rate') or '-'} | {seconds} |"
            )
    else:
        lines.extend(
            [
                "Only complete, failure-free artifacts with the exact selected dataset and protocol identity are included.",
                "",
                "| Group | Rank | Model | Macro CER | P90 CER | WER | Exact pages | Mean sec/image |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["rows"]:
            ci = (
                "-"
                if row["page_cer_ci95_low"] is None
                else f"{row['page_cer_ci95_low']:.4f}-{row['page_cer_ci95_high']:.4f}"
            )
            cer = f"{row['macro_page_cer_canonical']:.4f} ({ci})"
            wer = (
                "-"
                if row["mean_wer_canonical"] is None
                else f"{row['mean_wer_canonical']:.4f}"
            )
            seconds = (
                "-"
                if row["mean_seconds_per_image"] is None
                else f"{row['mean_seconds_per_image']:.3f}"
            )
            lines.append(
                f"| {row['comparison_group']} | {row['rank']} | `{row['model_id']}` | {cer} | {row.get('p90_page_cer_canonical') or '-'} | {wer} | {row.get('exact_page_rate') or '-'} | {seconds} |"
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            json.dumps(
                report["validation_summary"], ensure_ascii=False, sort_keys=True
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_charts(report: Mapping[str, Any], output_dir: Path) -> list[str]:
    rows = report["rows"]
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Plotting is an optional dependency; JSON/CSV/Markdown remain the
        # canonical report and the manifest makes the absence explicit.
        return []

    generated: list[str] = []
    if report["mode"] == "screening":
        labels = [row["model_id"] for row in rows]
        coverage = [row["coverage"] for row in rows]
        colors = {
            "Advance": "#2a9d8f",
            "Hold": "#e9c46a",
            "Reject": "#e76f51",
            "Blocked": "#8d99ae",
        }
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(rows) * 0.55)))
        ax.barh(
            labels[::-1],
            coverage[::-1],
            color=[colors[row["decision"]] for row in rows][::-1],
        )
        ax.set(
            title="Phase 1 screening coverage",
            xlabel="Successful page coverage",
            xlim=(0, 1.02),
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        name = "screening_coverage.png"
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)
        generated.append(name)

        plotted = [
            row
            for row in rows
            if row["macro_page_cer_canonical"] is not None
            and row["mean_seconds_per_image"] is not None
        ]
        by_environment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in plotted:
            by_environment[str(row.get("environment_hash") or "unknown")].append(row)
        for environment_hash, environment_rows in sorted(by_environment.items()):
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for row in environment_rows:
                low, high = row["page_cer_ci95_low"], row["page_cer_ci95_high"]
                yerr = (
                    None
                    if low is None
                    else [
                        [row["macro_page_cer_canonical"] - low],
                        [high - row["macro_page_cer_canonical"]],
                    ]
                )
                ax.errorbar(
                    row["mean_seconds_per_image"],
                    row["macro_page_cer_canonical"],
                    yerr=yerr,
                    fmt="o",
                    color=colors[row["decision"]],
                    capsize=3,
                )
                ax.annotate(
                    row["model_id"],
                    (row["mean_seconds_per_image"], row["macro_page_cer_canonical"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set(
                title=f"Phase 1 quality and latency (environment {environment_hash[:8]})",
                xlabel="Mean seconds per successful image",
                ylabel="Macro page CER (lower is better)",
            )
            ax.grid(alpha=0.25)
            fig.tight_layout()
            name = f"screening_quality_latency_{environment_hash[:8]}.png"
            fig.savefig(output_dir / name, dpi=160)
            plt.close(fig)
            generated.append(name)
    else:
        valid = [row for row in rows if row["macro_page_cer_canonical"] is not None]
        if valid:
            fig, ax = plt.subplots(figsize=(10, max(3.5, len(valid) * 0.55)))
            ordered = sorted(
                valid,
                key=lambda row: (
                    row["comparison_group"],
                    -row["macro_page_cer_canonical"],
                ),
            )
            ax.barh(
                [f"{row['comparison_group']} / {row['model_id']}" for row in ordered],
                [row["macro_page_cer_canonical"] for row in ordered],
                color="#2f6f9f",
            )
            ax.set(
                title="Final benchmark: compatible complete runs",
                xlabel="Macro page CER (lower is better)",
            )
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            name = "final_cer.png"
            fig.savefig(output_dir / name, dpi=160)
            plt.close(fig)
            generated.append(name)
    return generated


LEGACY_GENERATED = {
    "leaderboard.csv",
    "leaderboard.json",
    "leaderboard_by_type.csv",
    "leaderboard_by_type.json",
    "leaderboard_accuracy_latency.png",
    "leaderboard_cer.png",
    "leaderboard_hand_written.png",
    "leaderboard_latency.png",
    "leaderboard_typed.png",
}


def _clean_generated(output_dir: Path) -> None:
    candidates = set(LEGACY_GENERATED)
    manifest = output_dir / "report_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            candidates.update(
                name
                for name in payload.get("generated_files", [])
                if isinstance(name, str)
            )
        except (OSError, json.JSONDecodeError):
            pass
    candidates.update(
        {
            "screening_report.json",
            "screening_report.csv",
            "screening_report.md",
            "screening_slices.csv",
            "screening_coverage.png",
            "screening_quality_latency.png",
            "final_leaderboard.json",
            "final_leaderboard.csv",
            "final_leaderboard.md",
            "final_slices.csv",
            "final_cer.png",
            "report_manifest.json",
        }
    )
    root = output_dir.resolve()
    for name in candidates:
        target = (output_dir / name).resolve()
        if target.parent == root and target.is_file():
            target.unlink()


def write_report(report: Mapping[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_generated(output_dir)
    prefix = (
        "screening_report" if report["mode"] == "screening" else "final_leaderboard"
    )
    slice_name = (
        "screening_slices.csv" if report["mode"] == "screening" else "final_slices.csv"
    )
    json_name, csv_name, md_name = f"{prefix}.json", f"{prefix}.csv", f"{prefix}.md"
    (output_dir / json_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / csv_name, report["rows"])
    _write_csv(output_dir / slice_name, report["slices"])
    (output_dir / md_name).write_text(_markdown(report), encoding="utf-8")
    generated = [json_name, csv_name, slice_name, md_name]
    generated.extend(_write_charts(report, output_dir))
    if report["mode"] == "screening":
        # Keep the historical filenames usable by dashboards and notebooks while
        # making their contents come from the current v2 screening report.
        aliases = {
            "leaderboard.json": json_name,
            "leaderboard.csv": csv_name,
            "leaderboard_by_type.csv": slice_name,
            "leaderboard_cer.png": "screening_coverage.png",
            "leaderboard_accuracy_latency.png": next(
                (name for name in generated if name.startswith("screening_quality_latency_")),
                None,
            ),
        }
        for alias, source in aliases.items():
            if source and (output_dir / source).is_file():
                shutil.copyfile(output_dir / source, output_dir / alias)
                generated.append(alias)
    manifest_payload = {
        "schema": "persian_ocr_report_manifest_v1",
        "mode": report["mode"],
        "report_sha256": report["report_identity"]["report_sha256"],
        "generated_files": sorted(generated),
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sorted(generated + ["report_manifest.json"])


def discover_artifacts(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise RuntimeError(f"Input does not exist: {input_path}")
    artifacts: list[Path] = []
    for path in sorted(input_path.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # An explicitly selected malformed file is an error; unrelated
            # malformed JSON below a run root is not presumed to be an artifact.
            continue
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("results"), list)
            and isinstance(payload.get("summary"), dict)
        ):
            artifacts.append(path)
    return artifacts


def cli(argv: Sequence[str] | None = None, *, default_mode: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("screening", "final"), default=default_mode or "screening"
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "bench_runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--decisions",
        type=Path,
        help="Optional JSON decisions keyed by model id (screening only).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any artifact is invalid or incompatible.",
    )
    args = parser.parse_args(argv)
    output = args.output or (
        REPO_ROOT
        / "bench_runs"
        / ("screening_report" if args.mode == "screening" else "leaderboard")
    )
    try:
        paths = discover_artifacts(args.input)
        decisions = load_decisions(args.decisions) if args.mode == "screening" else {}
        report = build_report(paths, mode=args.mode, decision_overrides=decisions)
        generated = write_report(report, output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = report["validation_summary"]
    print(f"Mode:       {args.mode}")
    print(
        f"Artifacts:  {summary['artifacts_seen']} seen, {summary['artifacts_valid']} valid, {summary['artifacts_compatible']} compatible"
    )
    print(f"Rows:       {len(report['rows'])}")
    print(f"Output:     {output.resolve()}")
    if report["rows"]:
        print("Metrics:")
        print("  model_id                              CER     P90 CER  Worst Q   Exact  Mean sec")
        for row in report["rows"]:
            def _display(name: str, digits: int = 4) -> str:
                value = row.get(name)
                return "-" if value is None else f"{value:.{digits}f}"

            print(
                f"  {str(row['model_id']):<36} "
                f"{_display('macro_page_cer_canonical'):>7} "
                f"{_display('p90_page_cer_canonical'):>8} "
                f"{_display('worst_quartile_page_cer_canonical'):>8} "
                f"{_display('exact_page_rate'):>7} "
                f"{_display('mean_seconds_per_image', 3):>9}"
            )
    print(f"Generated:  {', '.join(generated)}")
    invalid = (
        summary["errors"] > 0
        or summary["artifacts_compatible"] < summary["artifacts_valid"]
        or summary["duplicate_runs"] > 0
    )
    no_final_rows = args.mode == "final" and not report["rows"]
    return 1 if no_final_rows or (args.strict and invalid) else 0


__all__ = [
    "BENCHMARK_SCHEMA",
    "DECISIONS",
    "FINAL_REPORT_SCHEMA",
    "PROTOCOL_IDENTITY_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "SCREENING_REPORT_SCHEMA",
    "ValidatedArtifact",
    "ValidationIssue",
    "build_report",
    "cli",
    "compatibility_key",
    "discover_artifacts",
    "slice_rows",
    "validate_artifact",
    "write_report",
]
