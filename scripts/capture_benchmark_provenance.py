"""Capture reproducibility metadata for a completed smoke20 benchmark run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "bench_runs" / "smoke20-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def capture(run_root: Path) -> dict[str, Any]:
    state = load_json(run_root / "state.json")
    catalog = yaml.safe_load((REPO_ROOT / "models.yaml").read_text(encoding="utf-8"))
    catalog_models = {
        item["id"]: item
        for item in catalog.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    models: dict[str, Any] = {}
    for model_id, model_state in state.get("models", {}).items():
        artifact_path = REPO_ROOT / model_state["output"]
        artifact = load_json(artifact_path)
        identity = artifact.get("run_identity", {})
        models[model_id] = {
            "command": model_state.get("command"),
            "status": model_state.get("status"),
            "return_code": model_state.get("return_code"),
            "artifact": model_state.get("output"),
            "artifact_sha256": sha256(artifact_path),
            "catalog_identity": {
                key: catalog_models.get(model_id, {}).get(key)
                for key in ("id", "name", "category", "priority", "license")
            },
            "run_identity": {
                "model": identity.get("model"),
                "protocol": identity.get("protocol"),
                "dataset": identity.get("dataset"),
                "runner": identity.get("runner"),
                "config": identity.get("config"),
                "environment": identity.get("environment"),
            },
        }
    return {
        "schema": "persian_ocr_benchmark_provenance_v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": state.get("dataset"),
        "run_root": str(run_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "git": {"commit": git_value("rev-parse", "HEAD"), "dirty": git_value("status", "--porcelain") != ""},
        "dependencies": {
            "lockfile": "uv.lock",
            "lockfile_sha256": sha256(REPO_ROOT / "uv.lock"),
            "project": "pyproject.toml",
            "project_sha256": sha256(REPO_ROOT / "pyproject.toml"),
        },
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else REPO_ROOT / args.run_root
    output = args.output or run_root / "provenance.json"
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.write_text(json.dumps(capture(run_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
