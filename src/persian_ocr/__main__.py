"""Lightweight command line entry point for shared benchmark operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__


def _models_status(args: argparse.Namespace) -> int:
    from .catalog import catalog_status, load_catalog, validate_catalog

    payload = load_catalog(args.catalog, workspace=args.workspace)
    report = validate_catalog(payload)
    status = catalog_status(payload)
    if args.json:
        print(json.dumps({**status, "validation": report.to_dict()}, ensure_ascii=False, indent=2))
    else:
        counts = ", ".join(f"{key}={value}" for key, value in status["status_counts"].items())
        print(f"Models: {status['n_models']} ({counts})")
        print("\nRank Priority Status         Model")
        for model in status["models"]:
            rank = "-" if model["rank"] is None else str(model["rank"])
            print(
                f"{rank:>4} {model['priority']:<8} {model['status']:<14} {model['id']}"
            )
        for warning in report.warnings:
            print(f"[WARNING] {warning}")
    return 0


def _dataset_validate(args: argparse.Namespace) -> int:
    from .dataset import load_dataset, validate_dataset
    from .paths import workspace_paths

    workspace = workspace_paths(args.workspace)
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else workspace.manifest
    samples = load_dataset(manifest, workspace.root)
    report = validate_dataset(
        samples,
        require_reviewed=args.require_reviewed,
        verify_images=not args.no_image_verify,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        state = "PASS" if report.ok else "FAIL"
        print(f"Dataset validation: {state}")
        for key, value in report.stats.items():
            print(f"  {key}: {value}")
        for warning in report.warnings:
            print(f"[WARNING] {warning}")
        for error in report.errors:
            print(f"[ERROR] {error}")
    return 0 if report.ok else 1


def _dataset_identity(args: argparse.Namespace) -> int:
    from .dataset import build_dataset_identity, load_dataset
    from .paths import workspace_paths

    workspace = workspace_paths(args.workspace)
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else workspace.manifest
    samples = load_dataset(manifest, workspace.root)
    identity = build_dataset_identity(manifest, samples, workspace_root=workspace.root)
    print(json.dumps(identity, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="persian-ocr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="Inspect model catalog metadata.")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    status = model_commands.add_parser("status", help="Show declared adapter/artifact status.")
    status.add_argument("--workspace", type=Path)
    status.add_argument("--catalog", type=Path)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_models_status)

    dataset = commands.add_parser("dataset", help="Validate or identify benchmark data.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate", help="Validate manifest, hashes, and images.")
    validate.add_argument("--workspace", type=Path)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--require-reviewed", action="store_true")
    validate.add_argument("--no-image-verify", action="store_true")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=_dataset_validate)
    identity = dataset_commands.add_parser("identity", help="Print the v2 dataset identity.")
    identity.add_argument("--workspace", type=Path)
    identity.add_argument("--manifest", type=Path)
    identity.set_defaults(handler=_dataset_identity)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
