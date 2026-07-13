"""Validate models.yaml and report local implementation/benchmark status."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persian_ocr.reporting import validate_artifact  # noqa: E402


DEFAULT_CATALOG = REPO_ROOT / "models.yaml"
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


def add_issue(issues: list[Issue], severity: str, code: str, message: str) -> None:
    issues.append(Issue(severity=severity, code=code, message=message))


def load_catalog(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Catalog root must be a mapping: {path}")
    return payload


def resolve_repo_path(value: str | None, pattern: str, model_id: str) -> Path:
    rendered = value or pattern.format(id=model_id)
    path = Path(rendered)
    return path if path.is_absolute() else REPO_ROOT / path


def valid_url(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def inspect_artifact(
    path: Path, model_id: str, issues: list[Issue]
) -> tuple[bool, bool]:
    """Return ``(selection_valid, complete)`` after strict v2 validation."""
    validation = validate_artifact(path, expected_model_id=model_id)
    for validation_issue in validation.issues:
        add_issue(
            issues,
            validation_issue.severity,
            validation_issue.code,
            f"{path.name}: {validation_issue.message}",
        )

    # Keep a targeted diagnostic for historical artifacts.  Absolute paths are
    # permitted only while inspecting them; v2 output never serializes them.
    results = validation.payload.get("results")
    stale = 0
    if not isinstance(results, list):
        return False, False
    missing_sources = 0
    for result in results:
        source = result.get("reference_source") if isinstance(result, dict) else None
        if not source:
            continue
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if not source_path.is_file():
            missing_sources += 1
    if missing_sources:
        add_issue(
            issues,
            "error",
            "artifact_reference_stale",
            f"{path.name}: {missing_sources}/{len(results)} reference_source paths are missing",
        )
        stale = missing_sources
    return validation.valid and not stale, validation.complete and not stale


def validate_catalog(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Issue]]:
    issues: list[Issue] = []
    catalog = payload.get("catalog")
    protocol = payload.get("benchmark_protocol")
    registry = payload.get("execution_registry")
    models = payload.get("models")
    excluded = payload.get("excluded_or_redundant", [])

    if not isinstance(catalog, dict):
        add_issue(issues, "error", "catalog_missing", "catalog must be a mapping")
        catalog = {}
    if not isinstance(protocol, dict):
        add_issue(
            issues, "error", "protocol_missing", "benchmark_protocol must be a mapping"
        )
        protocol = {}
    if not isinstance(registry, dict):
        add_issue(
            issues, "error", "registry_missing", "execution_registry must be a mapping"
        )
        registry = {}
    if not isinstance(models, list) or not models:
        add_issue(issues, "error", "models_missing", "models must be a non-empty list")
        return [], issues
    if not isinstance(excluded, list):
        add_issue(
            issues, "error", "excluded_invalid", "excluded_or_redundant must be a list"
        )
        excluded = []

    model_ids = [model.get("id") for model in models if isinstance(model, dict)]
    if len(model_ids) != len(models) or any(
        not isinstance(item, str) for item in model_ids
    ):
        add_issue(
            issues, "error", "model_id_invalid", "every model requires a string id"
        )
    duplicates = [item for item, count in Counter(model_ids).items() if count > 1]
    if duplicates:
        add_issue(
            issues, "error", "model_id_duplicate", f"duplicate model ids: {duplicates}"
        )

    ranks = [model.get("rank") for model in models if isinstance(model, dict)]
    if ranks != list(range(1, len(models) + 1)):
        add_issue(
            issues, "error", "rank_not_contiguous", f"ranks must be 1..{len(models)}"
        )

    track_names = {
        track.get("name")
        for track in protocol.get("leaderboards", [])
        if isinstance(track, dict) and isinstance(track.get("name"), str)
    }
    allowed_statuses = set(registry.get("status_values", {}))
    default_status = registry.get("default_status", "catalog_only")
    entrypoint_pattern = registry.get("entrypoint_pattern", "src/{id}.py")
    artifact_pattern = registry.get("artifact_pattern", "bench_runs/{id}.json")
    if default_status not in allowed_statuses:
        add_issue(
            issues,
            "error",
            "default_status_invalid",
            f"unknown default status: {default_status}",
        )

    rows: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        model_id = model["id"]
        priority = model.get("priority")
        if priority not in PRIORITY_ORDER:
            add_issue(
                issues,
                "error",
                "priority_invalid",
                f"{model_id}: invalid priority {priority!r}",
            )
        support_status = model.get("persian_support", {}).get("status")
        if support_status not in catalog.get("persian_support_status", {}):
            add_issue(
                issues,
                "error",
                "persian_status_invalid",
                f"{model_id}: invalid Persian support status {support_status!r}",
            )
        unknown_tracks = sorted(set(model.get("recommended_tracks", [])) - track_names)
        if unknown_tracks:
            add_issue(
                issues,
                "error",
                "track_invalid",
                f"{model_id}: unknown tracks {unknown_tracks}",
            )
        for name, value in model.get("links", {}).items():
            if name.endswith("_scope"):
                continue
            if not valid_url(value):
                add_issue(
                    issues,
                    "error",
                    "url_invalid",
                    f"{model_id}: links.{name}={value!r}",
                )

        execution = model.get("execution") or {}
        declared_status = execution.get("status", default_status)
        if declared_status not in allowed_statuses:
            add_issue(
                issues,
                "error",
                "execution_status_invalid",
                f"{model_id}: unknown execution status {declared_status!r}",
            )
        entrypoint = resolve_repo_path(
            execution.get("entrypoint"), entrypoint_pattern, model_id
        )
        artifact = resolve_repo_path(
            execution.get("artifact"), artifact_pattern, model_id
        )
        entrypoint_exists = entrypoint.is_file()
        artifact_exists = artifact.is_file()
        artifact_valid = artifact_complete = False
        if artifact_exists:
            artifact_valid, artifact_complete = inspect_artifact(
                artifact, model_id, issues
            )
        if entrypoint_exists and artifact_exists and artifact_complete:
            actual_status = "benchmarked"
        elif entrypoint_exists and artifact_exists and artifact_valid:
            actual_status = "benchmark_partial"
        elif artifact_exists:
            actual_status = "artifact_invalid"
        elif entrypoint_exists:
            actual_status = "adapter_ready"
        else:
            actual_status = "catalog_only"
        if (
            declared_status not in {"blocked", "rejected"}
            and declared_status != actual_status
        ):
            add_issue(
                issues,
                "error"
                if declared_status == "benchmarked"
                or actual_status == "artifact_invalid"
                else "warning",
                "execution_status_drift",
                f"{model_id}: declares {declared_status}, filesystem implies {actual_status}",
            )
        rows.append(
            {
                "rank": model.get("rank"),
                "id": model_id,
                "priority": priority,
                "persian_support": support_status,
                "declared_status": declared_status,
                "actual_status": actual_status,
                "entrypoint": str(entrypoint.relative_to(REPO_ROOT)),
                "artifact": str(artifact.relative_to(REPO_ROOT)),
            }
        )

    active_ids = set(model_ids)
    excluded_ids = [item.get("id") for item in excluded if isinstance(item, dict)]
    overlap = sorted(active_ids & set(excluded_ids))
    if overlap:
        add_issue(
            issues,
            "error",
            "active_excluded_overlap",
            f"active and excluded ids overlap: {overlap}",
        )
    for item in excluded:
        if not isinstance(item, dict):
            continue
        replacement = item.get("replacement")
        if replacement and replacement not in active_ids:
            add_issue(
                issues,
                "error",
                "replacement_missing",
                f"{item.get('id')}: replacement {replacement!r} is not active",
            )
    return rows, issues


def queue_key(row: dict[str, Any]) -> tuple[int, int]:
    return PRIORITY_ORDER.get(row["priority"], 99), int(row["rank"] or 9999)


def print_report(
    rows: list[dict[str, Any]], issues: list[Issue], show_all: bool
) -> None:
    counts = Counter(row["actual_status"] for row in rows)
    print(f"Catalog: {len(rows)} active models")
    print(
        "Status:  "
        + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    )
    if show_all:
        print("\nRank Priority Status             Model")
        for row in rows:
            print(
                f"{row['rank']:>4} {row['priority']:<8} "
                f"{row['actual_status']:<18} {row['id']}"
            )
    ready = sorted(
        (row for row in rows if row["actual_status"] == "adapter_ready"), key=queue_key
    )
    planned = sorted(
        (row for row in rows if row["actual_status"] == "catalog_only"), key=queue_key
    )
    print(f"\nNext benchmark:      {ready[0]['id'] if ready else '-'}")
    print(f"Next implementation: {planned[0]['id'] if planned else '-'}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  [{issue.severity.upper():7}] {issue.code}: {issue.message}")
    else:
        print("\nIssues: none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--all", action="store_true", help="List every active model.")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures."
    )
    args = parser.parse_args()
    try:
        payload = load_catalog(args.catalog)
        rows, issues = validate_catalog(payload)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {"models": rows, "issues": [asdict(item) for item in issues]}, indent=2
            )
        )
    else:
        print_report(rows, issues, args.all)
    has_errors = any(issue.severity == "error" for issue in issues)
    has_warnings = any(issue.severity == "warning" for issue in issues)
    return 1 if has_errors or (args.strict and has_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
