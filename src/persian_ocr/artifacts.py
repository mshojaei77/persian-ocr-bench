"""Content-addressed benchmark artifact identities and v2 validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from . import __version__
from .paths import PACKAGE_ROOT, ensure_logical_path


ARTIFACT_SCHEMA = "persian_ocr_benchmark_v2"
RUN_IDENTITY_SCHEMA = "persian_ocr_run_identity_v2"
DATASET_IDENTITY_SCHEMA = "persian_ocr_dataset_identity_v2"
PROTOCOL_IDENTITY_SCHEMA = "persian_ocr_protocol_identity_v2"
RESULT_STATUS_VALUES = {"ok", "error", "skipped"}


class ArtifactValidationError(ValueError):
    """Raised when an identity or benchmark artifact violates the v2 contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data deterministically for cross-platform hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_digest(payload: Mapping[str, Any], key: str = "digest") -> dict[str, Any]:
    identity = dict(payload)
    identity[key] = sha256_json(identity)
    return identity


def build_protocol_identity(
    protocol_id: str,
    *,
    phase: str,
    track: str,
    version: str = "1",
    **metadata: Any,
) -> dict[str, Any]:
    """Create the comparison boundary used by artifact eligibility checks."""
    payload = {
        "schema": PROTOCOL_IDENTITY_SCHEMA,
        "id": protocol_id,
        "phase": phase,
        "track": track,
        "version": version,
        **metadata,
    }
    return _with_digest(payload)


def source_identity(
    files: Iterable[str | Path] | None = None,
    *,
    root: str | Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    """Hash every scoring/runner source file, not only an adapter entry point."""
    source_root = Path(root).resolve()
    candidates = (
        sorted(source_root.rglob("*.py"))
        if files is None
        else sorted(Path(path).resolve() for path in files)
    )
    file_hashes: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            logical = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Runner source is outside {source_root}: {path}") from exc
        file_hashes[logical] = sha256_file(path)
    return {"files": file_hashes, "source_sha256": sha256_json(file_hashes)}


def git_identity(workspace: str | Path | None) -> dict[str, Any]:
    """Return best-effort VCS provenance without making Git a runtime requirement."""
    if workspace is None:
        return {"commit": None, "dirty": None}
    root = Path(workspace).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def build_runner_identity(
    *,
    source_files: Iterable[str | Path] | None = None,
    source_root: str | Path = PACKAGE_ROOT,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    sources = source_identity(source_files, root=source_root)
    return {
        "package": "persian-ocr",
        "version": __version__,
        **sources,
        "git": git_identity(workspace),
    }


def installed_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_environment_identity(packages: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "packages": installed_versions(packages),
    }


def build_run_identity(
    *,
    protocol: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    runner: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed identity over every result-affecting input."""
    payload = {
        "schema": RUN_IDENTITY_SCHEMA,
        "protocol": dict(protocol),
        "dataset": dict(dataset),
        "model": dict(model),
        "runner": dict(runner or build_runner_identity()),
        "config": dict(config),
        "environment": dict(environment or build_environment_identity()),
    }
    identity = _with_digest(payload)
    validate_run_identity(identity)
    return identity


def build_result_record(
    *,
    sample_id: str,
    image: str,
    split: str,
    track: str,
    content_type: str,
    seconds: float,
    reference: str | None = None,
    prediction: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
    reference_source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    raw_output: Any = None,
) -> dict[str, Any]:
    """Create one portable result record with an explicit success state."""
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "image": ensure_logical_path(image),
        "split": split,
        "track": track,
        "content_type": content_type,
        "status": "error" if error else "ok",
        "seconds": round(float(seconds), 6),
        "metadata": dict(metadata or {}),
    }
    if reference_source is not None:
        record["reference_source"] = ensure_logical_path(reference_source)
    if error:
        record["error"] = error
    else:
        if reference is None or prediction is None or metrics is None:
            raise ValueError("Successful results require reference, prediction, and metrics")
        record.update(
            reference=reference,
            prediction=prediction,
            metrics=dict(metrics),
        )
    if raw_output is not None:
        record["raw_output"] = raw_output
    return record


def build_summary(
    *,
    model: Mapping[str, Any],
    protocol: Mapping[str, Any],
    dataset: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(result.get("status")) for result in results)
    if metrics is None:
        from .metrics import summarize_records

        metrics = summarize_records(results)
    return {
        "model": {"id": model.get("id"), "class": model.get("class")},
        "protocol": {
            key: protocol.get(key) for key in ("id", "phase", "track", "version", "digest")
        },
        "dataset": {
            "id": dataset.get("id"),
            "digest": dataset.get("digest"),
            "n_samples": dataset.get("n_samples"),
        },
        "counts": {
            "total": len(results),
            "ok": counts["ok"],
            "error": counts["error"],
            "skipped": counts["skipped"],
        },
        "metrics": dict(metrics),
        "operations": dict(operations or {}),
    }


def build_artifact(
    *,
    run_identity: Mapping[str, Any],
    summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "run_id": run_identity.get("digest"),
        "run_identity": dict(run_identity),
        "summary": dict(summary),
        "results": [dict(result) for result in results],
    }
    validate_artifact(artifact)
    return artifact


def artifact_relative_path(run_identity: Mapping[str, Any]) -> str:
    protocol = run_identity["protocol"]
    dataset = run_identity["dataset"]
    model = run_identity["model"]
    return ensure_logical_path(
        "/".join(
            (
                str(protocol["id"]),
                f"{dataset['id']}@{dataset['version']}-{str(dataset['digest'])[:12]}",
                str(model["id"]),
                str(run_identity["digest"]),
                "result.json",
            )
        )
    )


def _validate_digest(
    identity: Mapping[str, Any], *, key: str, label: str, errors: list[str]
) -> None:
    claimed = identity.get(key)
    payload = dict(identity)
    payload.pop(key, None)
    try:
        expected = sha256_json(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label} is not canonical JSON: {exc}")
        return
    if claimed != expected:
        errors.append(f"{label}.{key} does not match its canonical content")


def validate_protocol_identity(identity: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if identity.get("schema") != PROTOCOL_IDENTITY_SCHEMA:
        errors.append(f"protocol.schema must be {PROTOCOL_IDENTITY_SCHEMA}")
    for key in ("id", "phase", "track", "version", "digest"):
        if not identity.get(key):
            errors.append(f"protocol.{key} is required")
    _validate_digest(identity, key="digest", label="protocol", errors=errors)
    if errors:
        raise ArtifactValidationError(errors)


def validate_dataset_identity(identity: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if identity.get("schema") != DATASET_IDENTITY_SCHEMA:
        errors.append(f"dataset.schema must be {DATASET_IDENTITY_SCHEMA}")
    for key in (
        "id",
        "version",
        "digest",
        "dataset_sha256",
        "manifest_sha256",
        "reference_corpora_sha256",
        "images_sha256",
        "n_samples",
        "splits",
        "content_types",
        "samples",
    ):
        if key not in identity:
            errors.append(f"dataset.{key} is required")
    claimed = identity.get("digest")
    if identity.get("dataset_sha256") != claimed:
        errors.append("dataset.digest and dataset.dataset_sha256 must match")
    samples = identity.get("samples")
    if isinstance(samples, list) and identity.get("id") and identity.get("version"):
        try:
            if all("metadata_sha256" in sample for sample in samples):
                material = {
                    "dataset_id": identity["id"],
                    "dataset_version": identity["version"],
                    "samples": [
                        {
                            "sample_id": sample["sample_id"],
                            "image_sha256": sample["image_sha256"],
                            "reference_scorable_sha256": sample.get(
                                "reference_scorable_sha256", sample["reference_sha256"]
                            ),
                            "metadata_sha256": sample["metadata_sha256"],
                        }
                        for sample in samples
                    ],
                }
                expected = sha256_json(material)
            else:
                lines = [f"{identity['id']}\n{identity['version']}"]
                for sample in samples:
                    lines.append(
                        f"{sample['sample_id']}|{sample['image_sha256']}|"
                        f"{sample['reference_sha256']}"
                    )
                expected = sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))
            if claimed != expected:
                errors.append("dataset.digest does not match ordered sample content")
        except (KeyError, TypeError) as exc:
            errors.append(f"dataset.samples cannot reconstruct the digest: {exc}")
    if errors:
        raise ArtifactValidationError(errors)


def validate_run_identity(identity: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if identity.get("schema") != RUN_IDENTITY_SCHEMA:
        errors.append(f"run_identity.schema must be {RUN_IDENTITY_SCHEMA}")
    for key in ("protocol", "dataset", "model", "runner", "config", "environment"):
        if not isinstance(identity.get(key), Mapping):
            errors.append(f"run_identity.{key} must be an object")
    model = identity.get("model")
    if isinstance(model, Mapping):
        for key in ("id", "class", "checkpoint_type", "identity"):
            if key not in model:
                errors.append(f"run_identity.model.{key} is required")
    if isinstance(identity.get("protocol"), Mapping):
        try:
            validate_protocol_identity(identity["protocol"])
        except ArtifactValidationError as exc:
            errors.extend(exc.errors)
    if isinstance(identity.get("dataset"), Mapping):
        try:
            validate_dataset_identity(identity["dataset"])
        except ArtifactValidationError as exc:
            errors.extend(exc.errors)
    _validate_digest(identity, key="digest", label="run_identity", errors=errors)
    if errors:
        raise ArtifactValidationError(errors)


def _validate_result(result: Mapping[str, Any], index: int, errors: list[str]) -> None:
    prefix = f"results[{index}]"
    for key in ("sample_id", "image", "split", "track", "content_type", "status", "seconds"):
        if key not in result:
            errors.append(f"{prefix}.{key} is required")
    try:
        ensure_logical_path(result.get("image", ""))
    except ValueError as exc:
        errors.append(f"{prefix}.image: {exc}")
    if "reference_source" in result:
        try:
            ensure_logical_path(result["reference_source"])
        except ValueError as exc:
            errors.append(f"{prefix}.reference_source: {exc}")
    status = result.get("status")
    if status not in RESULT_STATUS_VALUES:
        errors.append(f"{prefix}.status must be one of {sorted(RESULT_STATUS_VALUES)}")
    if status == "ok":
        for key in ("reference", "prediction", "metrics"):
            if key not in result:
                errors.append(f"{prefix}.{key} is required for status=ok")
    elif status == "error" and not result.get("error"):
        errors.append(f"{prefix}.error is required for status=error")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        errors.append(f"artifact.schema must be {ARTIFACT_SCHEMA}")
    run_identity = artifact.get("run_identity")
    if not isinstance(run_identity, Mapping):
        errors.append("artifact.run_identity must be an object")
    else:
        try:
            validate_run_identity(run_identity)
        except ArtifactValidationError as exc:
            errors.extend(exc.errors)
        if artifact.get("run_id") != run_identity.get("digest"):
            errors.append("artifact.run_id must equal run_identity.digest")
    summary = artifact.get("summary")
    results = artifact.get("results")
    if not isinstance(summary, Mapping):
        errors.append("artifact.summary must be an object")
    if not isinstance(results, list):
        errors.append("artifact.results must be an array")
        results = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            errors.append(f"results[{index}] must be an object")
        else:
            _validate_result(result, index, errors)
    if isinstance(summary, Mapping):
        for key in ("model", "protocol", "dataset", "counts", "metrics", "operations"):
            if not isinstance(summary.get(key), Mapping):
                errors.append(f"summary.{key} must be an object")
        counts = summary.get("counts")
        if isinstance(counts, Mapping) and counts.get("total") != len(results):
            errors.append("summary.counts.total must equal len(results)")
        if isinstance(run_identity, Mapping):
            summary_model = summary.get("model", {})
            summary_protocol = summary.get("protocol", {})
            summary_dataset = summary.get("dataset", {})
            if summary_model.get("id") != run_identity.get("model", {}).get("id"):
                errors.append("summary.model.id must match run_identity.model.id")
            if summary_protocol.get("digest") != run_identity.get("protocol", {}).get("digest"):
                errors.append("summary.protocol.digest must match run_identity.protocol.digest")
            if summary_dataset.get("digest") != run_identity.get("dataset", {}).get("digest"):
                errors.append("summary.dataset.digest must match run_identity.dataset.digest")
    if errors:
        raise ArtifactValidationError(errors)


def write_artifact(
    artifact: Mapping[str, Any],
    *,
    output_root: str | Path,
    relative_path: str | None = None,
) -> Path:
    """Validate and atomically write a portable v2 artifact."""
    validate_artifact(artifact)
    logical = relative_path or artifact_relative_path(artifact["run_identity"])
    logical = ensure_logical_path(logical)
    destination = Path(output_root).resolve() / Path(*logical.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_artifact(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactValidationError(["Artifact root must be an object"])
    if validate:
        validate_artifact(payload)
    return payload


__all__ = [
    "ARTIFACT_SCHEMA",
    "DATASET_IDENTITY_SCHEMA",
    "PROTOCOL_IDENTITY_SCHEMA",
    "RESULT_STATUS_VALUES",
    "RUN_IDENTITY_SCHEMA",
    "ArtifactValidationError",
    "artifact_relative_path",
    "build_artifact",
    "build_environment_identity",
    "build_protocol_identity",
    "build_result_record",
    "build_run_identity",
    "build_runner_identity",
    "build_summary",
    "canonical_json_bytes",
    "git_identity",
    "installed_versions",
    "read_artifact",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "source_identity",
    "validate_artifact",
    "validate_dataset_identity",
    "validate_protocol_identity",
    "validate_run_identity",
    "write_artifact",
]
