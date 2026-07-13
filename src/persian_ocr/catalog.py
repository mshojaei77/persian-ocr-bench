"""Lightweight models.yaml loading and declared-status validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import workspace_paths


class CatalogError(RuntimeError):
    """Raised when the model catalog cannot be used safely."""


@dataclass(frozen=True)
class ModelRecord:
    id: str
    rank: int | None
    priority: str
    category: str
    declared_status: str
    execution: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class CatalogValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def load_catalog(
    path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    catalog_path = Path(path).expanduser().resolve() if path else workspace_paths(workspace).catalog
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"Could not load {catalog_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogError(f"Catalog root must be an object: {catalog_path}")
    report = validate_catalog(payload)
    if not report.ok:
        raise CatalogError("; ".join(report.errors))
    return payload


def validate_catalog(payload: Mapping[str, Any]) -> CatalogValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    registry = payload.get("execution_registry")
    models = payload.get("models")
    if not isinstance(registry, Mapping):
        errors.append("execution_registry must be an object")
        registry = {}
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty array")
        models = []
    status_values = registry.get("status_values", {})
    allowed_statuses = set(status_values) if isinstance(status_values, Mapping) else set()
    default_status = str(registry.get("default_status") or "catalog_only")
    ids: list[str] = []
    ranks: list[int] = []
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            errors.append(f"models[{index}] must be an object")
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(f"models[{index}].id is required")
            continue
        ids.append(model_id)
        rank = model.get("rank")
        if isinstance(rank, int):
            ranks.append(rank)
        execution = model.get("execution") or {}
        if not isinstance(execution, Mapping):
            errors.append(f"{model_id}.execution must be an object")
            continue
        status = str(execution.get("status") or default_status)
        if allowed_statuses and status not in allowed_statuses:
            errors.append(f"{model_id}.execution.status is invalid: {status}")
        if status in {"adapter_ready", "benchmarked"} and not execution.get("entrypoint"):
            warnings.append(f"{model_id}: {status} without an entrypoint")
        if status == "benchmarked" and not execution.get("artifact"):
            warnings.append(f"{model_id}: benchmarked without an artifact")
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    duplicate_ranks = sorted(value for value, count in Counter(ranks).items() if count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate model ids: {duplicate_ids}")
    if duplicate_ranks:
        warnings.append(f"Duplicate catalog ranks: {duplicate_ranks}")
    return CatalogValidationReport(tuple(errors), tuple(warnings))


def model_records(payload: Mapping[str, Any]) -> list[ModelRecord]:
    registry = payload.get("execution_registry", {})
    default_status = str(registry.get("default_status") or "catalog_only")
    records: list[ModelRecord] = []
    for raw in payload.get("models", []):
        execution = dict(raw.get("execution") or {})
        records.append(
            ModelRecord(
                id=str(raw["id"]),
                rank=raw.get("rank") if isinstance(raw.get("rank"), int) else None,
                priority=str(raw.get("priority") or ""),
                category=str(raw.get("category") or ""),
                declared_status=str(execution.get("status") or default_status),
                execution=execution,
                raw=dict(raw),
            )
        )
    return sorted(records, key=lambda item: (item.rank is None, item.rank or 0, item.id))


def get_model(payload: Mapping[str, Any], model_id: str) -> ModelRecord:
    try:
        return next(record for record in model_records(payload) if record.id == model_id)
    except StopIteration as exc:
        raise CatalogError(f"Unknown model id: {model_id}") from exc


def catalog_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = model_records(payload)
    return {
        "n_models": len(records),
        "status_counts": dict(
            sorted(Counter(record.declared_status for record in records).items())
        ),
        "models": [
            {
                "rank": record.rank,
                "id": record.id,
                "priority": record.priority,
                "category": record.category,
                "status": record.declared_status,
                "entrypoint": record.execution.get("entrypoint"),
                "artifact": record.execution.get("artifact"),
            }
            for record in records
        ],
    }


__all__ = [
    "CatalogError",
    "CatalogValidationReport",
    "ModelRecord",
    "catalog_status",
    "get_model",
    "load_catalog",
    "model_records",
    "validate_catalog",
]
