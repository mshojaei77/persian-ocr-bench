"""
Unified model downloader.

Usage:

    # List available models
    uv run python scripts/pull.py --list

    # Download one model
    uv run python scripts/pull.py --model surya-ocr-2

    # Download several models (comma-separated)
    uv run python scripts/pull.py --model deepseek-ocr,deepseek-ocr-2

    # Download all models sequentially
    uv run python scripts/pull.py --model all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

from persian_ocr.adapters import resolve_adapter
from persian_ocr.registry import MODELS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "huggingface"
DEFAULT_MANIFEST = ROOT / "models" / "manifest.json"


def parse_model_ids(value: str) -> list[str]:
    if value == "all":
        return list(MODELS)
    return [s.strip() for s in value.split(",") if s.strip()]


def validate_model_ids(ids: Iterable[str]) -> None:
    unknown = sorted(set(ids) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")


def pull_huggingface(
    spec,
    cache_dir: Path,
    token: str | None,
) -> Path:
    if not spec.repo_id:
        raise ValueError(f"{spec.id} has no repo_id")
    path = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=list(spec.allow_patterns) or None,
        ignore_patterns=list(spec.ignore_patterns) or None,
        max_workers=8,
    )
    return Path(path)


def pull_model(spec, cache_dir: Path, token: str | None) -> Path | None:
    if spec.download_kind == "none":
        print(f"  [skip] model has no separate download")
        return None
    if spec.download_kind == "custom":
        adapter_cls = resolve_adapter(spec.adapter)
        return adapter_cls.pull(spec=spec, cache_dir=cache_dir, token=token)
    return pull_huggingface(spec, cache_dir, token)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Persian OCR benchmark models.")
    parser.add_argument(
        "--model", "--models", dest="models",
        default="all",
        help="Model ID, comma-separated IDs, or 'all'.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list", action="store_true", help="List registered models and exit.")
    args = parser.parse_args()

    if args.list:
        print(f"{'Model ID':35} {'Profile':15} {'Repo':45} {'Persian':>8}")
        print("-" * 103)
        for mid, spec in MODELS.items():
            print(f"{mid:35} {spec.profile:15} {(spec.repo_id or '-'):45} {'YES' if spec.official_persian else '':>8}")
        return

    model_ids = parse_model_ids(args.models)
    validate_model_ids(model_ids)

    token = os.environ.get("HF_TOKEN")
    manifest = load_manifest(args.manifest)

    for mid in model_ids:
        spec = MODELS[mid]
        print(f"\n=== Pulling {spec.display_name} ===")

        try:
            path = pull_model(spec, args.cache_dir, token)
            manifest[mid] = {
                "path": str(path) if path else None,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "status": "ready",
            }
            save_manifest(args.manifest, manifest)
            print(f"  Ready: {path or '(no download needed)'}")
        except Exception as exc:
            manifest[mid] = {
                "path": None,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            save_manifest(args.manifest, manifest)
            print(f"  FAILED: {exc}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
