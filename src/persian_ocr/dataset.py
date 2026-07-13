"""Manifest-backed benchmark loading, validation, and dataset identity."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from .artifacts import (
    DATASET_IDENTITY_SCHEMA,
    sha256_bytes,
    sha256_file,
    validate_dataset_identity,
)
from .paths import discover_workspace, ensure_logical_path, logical_path, resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
HUMAN_REVIEW_STATUSES = {"human_reviewed", "human_verified"}


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be loaded or does not satisfy its contract."""


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    image: str
    reference: str
    reference_source: str
    reference_key: str
    split: str
    track: str
    content_type: str
    condition: tuple[str, ...]
    review_status: str
    provenance_status: str
    image_sha256: str
    reference_sha256: str
    dataset_id: str
    dataset_version: str
    dataset_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_quality(self) -> str:
        return "reviewed" if self.review_status in HUMAN_REVIEW_STATUSES else self.review_status

    @property
    def page_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "condition": list(self.condition),
            "review_status": self.review_status,
            "provenance_status": self.provenance_status,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        payload["condition"] = list(self.condition)
        return payload


@dataclass(frozen=True)
class DatasetValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    stats: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "stats": self.stats,
        }


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"Manifest row must be an object at {path}:{line_number}")
        if not row.get("image"):
            raise DatasetError(f"Manifest row is missing image at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise DatasetError(f"Manifest contains no samples: {path}")
    return rows


def _workspace_for_manifest(path: Path, workspace_root: str | Path | None) -> Path:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve()
    discovered = discover_workspace(start=path, required=False)
    return discovered or path.parent.resolve()


def load_manifest(
    path: str | Path,
    workspace_root: str | Path | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Compatibility loader returning resolved image paths and row metadata."""
    manifest = Path(path).expanduser().resolve()
    workspace = _workspace_for_manifest(manifest, workspace_root)
    entries: list[tuple[Path, dict[str, Any]]] = []
    for row in _read_manifest_rows(manifest):
        image = resolve_path(row["image"], base=workspace)
        metadata = dict(row)
        metadata.pop("image", None)
        entries.append((image, metadata))
    return entries


def _reference_indexes(
    rows: Iterable[Mapping[str, Any]], workspace: Path
) -> dict[Path, dict[str, str]]:
    indexes: dict[Path, dict[str, str]] = {}
    for row in rows:
        corpus_value = row.get("reference_corpus")
        if not corpus_value:
            image = resolve_path(str(row["image"]), base=workspace)
            corpus = image.parent.parent / f"{image.parent.name}.json"
        else:
            corpus = resolve_path(str(corpus_value), base=workspace)
        if corpus in indexes:
            continue
        try:
            payload = json.loads(corpus.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Invalid reference corpus {corpus}: {exc}") from exc
        if not isinstance(payload, list):
            raise DatasetError(f"Reference corpus must contain a JSON array: {corpus}")
        index: dict[str, str] = {}
        for item in payload:
            if not isinstance(item, dict) or not isinstance(item.get("image"), str):
                raise DatasetError(f"Invalid reference record in {corpus}")
            text = item.get("text")
            if not isinstance(text, str):
                raise DatasetError(f"Reference text must be a string: {corpus}:{item['image']}")
            index[item["image"]] = text
        indexes[corpus] = index
    return indexes


def load_dataset(
    manifest_path: str | Path,
    workspace_root: str | Path | None = None,
) -> list[DatasetSample]:
    """Load the explicit manifest and its split-level reference corpora."""
    manifest = Path(manifest_path).expanduser().resolve()
    workspace = _workspace_for_manifest(manifest, workspace_root)
    rows = _read_manifest_rows(manifest)
    indexes = _reference_indexes(rows, workspace)
    samples: list[DatasetSample] = []
    for index, row in enumerate(rows, 1):
        image_path = resolve_path(str(row["image"]), base=workspace)
        image_logical = ensure_logical_path(str(row["image"]).replace("\\", "/"))
        corpus_value = row.get("reference_corpus")
        if corpus_value:
            corpus = resolve_path(str(corpus_value), base=workspace)
            corpus_logical = ensure_logical_path(str(corpus_value).replace("\\", "/"))
        else:
            corpus = image_path.parent.parent / f"{image_path.parent.name}.json"
            corpus_logical = logical_path(corpus, base=workspace)
        reference_key = str(
            row.get("reference_key") or f"{image_path.parent.name}/{image_path.name}"
        )
        try:
            reference = indexes[corpus][reference_key]
        except KeyError as exc:
            raise DatasetError(f"Missing reference {reference_key!r} in {corpus}") from exc
        condition = row.get("condition", [])
        if isinstance(condition, str):
            condition = [condition]
        samples.append(
            DatasetSample(
                sample_id=str(row.get("sample_id") or image_logical),
                image_path=image_path,
                image=image_logical,
                reference=reference,
                reference_source=corpus_logical,
                reference_key=reference_key,
                split=str(row.get("split") or image_path.parent.name),
                track=str(row.get("track") or track_for_subdir(image_path.parent.name)),
                content_type=str(
                    row.get("content_type")
                    or ("handwritten" if "hand" in image_path.parent.name else "printed")
                ),
                condition=tuple(str(item) for item in condition),
                review_status=str(row.get("review_status") or "unreviewed"),
                provenance_status=str(row.get("provenance_status") or "unknown"),
                image_sha256=str(row.get("image_sha256") or sha256_file(image_path)),
                reference_sha256=str(
                    row.get("reference_sha256")
                    or sha256_bytes(reference.encode("utf-8"))
                ),
                dataset_id=str(row.get("dataset_id") or manifest.parent.name),
                dataset_version=str(row.get("dataset_version") or "1"),
                dataset_sha256=str(row.get("dataset_sha256") or ""),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "sample_id",
                        "image",
                        "reference_corpus",
                        "reference_key",
                        "split",
                        "track",
                        "content_type",
                        "condition",
                        "review_status",
                        "provenance_status",
                        "image_sha256",
                        "reference_sha256",
                        "dataset_id",
                        "dataset_version",
                        "dataset_sha256",
                    }
                },
            )
        )
    return samples


def load_ground_truth(image_path: str | Path) -> tuple[str, str, str, dict[str, Any]]:
    """Load a legacy split-corpus reference for one image."""
    image = Path(image_path).expanduser().resolve()
    corpus = image.parent.parent / f"{image.parent.name}.json"
    try:
        payload = json.loads(corpus.read_text(encoding="utf-8"))
        record = next(
            item
            for item in payload
            if item.get("image") == f"{image.parent.name}/{image.name}"
        )
        text = record["text"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, StopIteration) as exc:
        raise DatasetError(f"Invalid benchmark reference {corpus}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"Empty reference in {corpus} for {image.name}")
    return text, str(corpus), "unreviewed", {}


def track_for_subdir(subdir: str) -> str:
    return {
        "typed": "printed_smoke",
        "hand-written": "handwriting_smoke",
    }.get(subdir, "unclassified")


def dataset_content_digest(
    dataset_id: str,
    dataset_version: str,
    samples: Iterable[DatasetSample],
) -> str:
    lines = [f"{dataset_id}\n{dataset_version}"]
    lines.extend(
        f"{sample.sample_id}|{sample.image_sha256}|{sample.reference_sha256}"
        for sample in samples
    )
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def validate_dataset(
    samples: Iterable[DatasetSample],
    *,
    require_reviewed: bool = False,
    verify_images: bool = True,
) -> DatasetValidationReport:
    selected = list(samples)
    errors: list[str] = []
    warnings: list[str] = []
    ids = [sample.sample_id for sample in selected]
    images = [sample.image for sample in selected]
    for label, values in (("sample_id", ids), ("image", images)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            errors.append(f"Duplicate {label} values: {duplicates}")
    for sample in selected:
        if not sample.image_path.is_file():
            errors.append(f"{sample.sample_id}: image does not exist: {sample.image_path}")
            continue
        actual_image_hash = sha256_file(sample.image_path)
        if actual_image_hash != sample.image_sha256:
            errors.append(f"{sample.sample_id}: image_sha256 mismatch")
        actual_reference_hash = sha256_bytes(sample.reference.encode("utf-8"))
        if actual_reference_hash != sample.reference_sha256:
            errors.append(f"{sample.sample_id}: reference_sha256 mismatch")
        if not sample.reference.strip():
            errors.append(f"{sample.sample_id}: reference is empty")
        if require_reviewed and sample.review_status not in HUMAN_REVIEW_STATUSES:
            errors.append(
                f"{sample.sample_id}: review_status={sample.review_status!r} is not human-reviewed"
            )
        if verify_images:
            try:
                with Image.open(sample.image_path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                errors.append(f"{sample.sample_id}: unreadable image: {exc}")
        if sample.provenance_status != "verified":
            warnings.append(
                f"{sample.sample_id}: provenance_status={sample.provenance_status}"
            )
    dataset_ids = {sample.dataset_id for sample in selected}
    versions = {sample.dataset_version for sample in selected}
    declared_digests = {sample.dataset_sha256 for sample in selected if sample.dataset_sha256}
    if len(dataset_ids) != 1:
        errors.append(f"Manifest mixes dataset_id values: {sorted(dataset_ids)}")
    if len(versions) != 1:
        errors.append(f"Manifest mixes dataset_version values: {sorted(versions)}")
    if len(declared_digests) != 1:
        errors.append(f"Manifest must declare one dataset_sha256: {sorted(declared_digests)}")
    if len(dataset_ids) == len(versions) == len(declared_digests) == 1:
        computed = dataset_content_digest(next(iter(dataset_ids)), next(iter(versions)), selected)
        if computed != next(iter(declared_digests)):
            errors.append("dataset_sha256 does not match ordered sample content")
    stats = {
        "n_samples": len(selected),
        "dataset_ids": sorted(dataset_ids),
        "dataset_versions": sorted(versions),
        "dataset_sha256": next(iter(declared_digests)) if len(declared_digests) == 1 else None,
        "splits": dict(sorted(Counter(sample.split for sample in selected).items())),
        "content_types": dict(
            sorted(Counter(sample.content_type for sample in selected).items())
        ),
        "review_statuses": dict(
            sorted(Counter(sample.review_status for sample in selected).items())
        ),
    }
    return DatasetValidationReport(tuple(errors), tuple(warnings), stats)


def require_valid_dataset(
    samples: Iterable[DatasetSample],
    *,
    require_reviewed: bool = False,
    verify_images: bool = True,
) -> DatasetValidationReport:
    report = validate_dataset(
        samples, require_reviewed=require_reviewed, verify_images=verify_images
    )
    if not report.ok:
        raise DatasetError("; ".join(report.errors))
    return report


def build_dataset_identity(
    manifest_path: str | Path,
    samples: Iterable[DatasetSample],
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create and validate the ordered content identity used for run gating."""
    manifest = Path(manifest_path).expanduser().resolve()
    workspace = _workspace_for_manifest(manifest, workspace_root)
    selected = list(samples)
    report = require_valid_dataset(selected, verify_images=False)
    dataset_id = report.stats["dataset_ids"][0]
    dataset_version = report.stats["dataset_versions"][0]
    digest = dataset_content_digest(dataset_id, dataset_version, selected)
    corpora = sorted({sample.reference_source for sample in selected})
    identity = {
        "schema": DATASET_IDENTITY_SCHEMA,
        "id": dataset_id,
        "version": dataset_version,
        "digest": digest,
        "dataset_sha256": digest,
        "manifest": logical_path(manifest, base=workspace),
        "manifest_sha256": sha256_file(manifest),
        "reference_corpora_sha256": {
            corpus: sha256_file(resolve_path(corpus, base=workspace)) for corpus in corpora
        },
        "images_sha256": {
            sample.image: sample.image_sha256 for sample in selected
        },
        "samples": [
            {
                "sample_id": sample.sample_id,
                "image_sha256": sample.image_sha256,
                "reference_sha256": sample.reference_sha256,
            }
            for sample in selected
        ],
        "n_samples": len(selected),
        "splits": report.stats["splits"],
        "content_types": report.stats["content_types"],
    }
    validate_dataset_identity(identity)
    return identity


__all__ = [
    "HUMAN_REVIEW_STATUSES",
    "IMAGE_EXTENSIONS",
    "DatasetError",
    "DatasetSample",
    "DatasetValidationReport",
    "build_dataset_identity",
    "dataset_content_digest",
    "load_dataset",
    "load_ground_truth",
    "load_manifest",
    "require_valid_dataset",
    "track_for_subdir",
    "validate_dataset",
]
