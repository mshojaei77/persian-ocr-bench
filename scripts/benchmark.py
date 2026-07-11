"""
Unified benchmark runner.

Usage:

    # Benchmark one model (auto-download if needed)
    uv run python scripts/benchmark.py --model surya-ocr-2 --auto-pull --resume

    # Benchmark multiple models (sequential, isolated subprocess each)
    uv run python scripts/benchmark.py --model deepseek-ocr,deepseek-ocr-2 --auto-pull

    # Benchmark all models
    uv run python scripts/benchmark.py --model all --auto-pull

    # Run only handwritten samples
    uv run python scripts/benchmark.py --model surya-ocr-2 --split hand-written

Each model runs in its own subprocess so GPU memory is fully released
between models.  Predictions are written immediately as JSONL for
resumability; ``scores.csv`` and ``summary.csv`` are generated after.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from persian_ocr.adapters import resolve_adapter
from persian_ocr.dataset import load_dataset
from persian_ocr.registry import MODELS
from persian_ocr.results import (
    append_record,
    build_record,
    read_completed,
    write_scores_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "small_bench"
DEFAULT_OUTPUT = ROOT / "bench_runs"
DEFAULT_CACHE = ROOT / ".cache" / "huggingface"
DEFAULT_MANIFEST = ROOT / "models" / "manifest.json"


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def parse_model_ids(value: str) -> list[str]:
    if value == "all":
        return list(MODELS)
    return [s.strip() for s in value.split(",") if s.strip()]


def validate_model_ids(ids: list[str]) -> None:
    unknown = sorted(set(ids) - set(MODELS))
    if unknown:
        raise SystemExit(f"Unknown models: {', '.join(unknown)}")


def ensure_model_downloaded(
    model_id: str,
    cache_dir: Path,
    manifest_path: Path,
) -> Path | None:
    """Delegate to ``pull.py`` in a subprocess."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "pull.py"),
        "--model", model_id,
        "--cache-dir", str(cache_dir),
        "--manifest", str(manifest_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull model {model_id}")

    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest.get(model_id, {}).get("path")
    return Path(path) if path else None


# ═══════════════════════════════════════════════════════════════════════
# Worker — runs for one model in a subprocess
# ═══════════════════════════════════════════════════════════════════════

def run_worker(args: argparse.Namespace) -> None:
    """Benchmark a single model and write JSONL results."""
    model_id = args.model
    spec = MODELS[model_id]

    # 1. Ensure model is downloaded
    model_path: Path | None = None
    if args.auto_pull:
        print(f"Ensuring {model_id} is downloaded ...")
        model_path = ensure_model_downloaded(
            model_id, args.cache_dir, args.manifest
        )
    else:
        # Check manifest for existing download
        if args.manifest.exists():
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            mp = manifest.get(model_id, {}).get("path")
            if mp:
                model_path = Path(mp)

    # 2. Load adapter
    adapter_cls = resolve_adapter(spec.adapter)
    adapter = adapter_cls(
        spec=spec,
        model_path=model_path,
        device=args.device,
    )

    # 3. Load dataset
    samples = load_dataset(args.dataset)
    if args.split != "all":
        samples = [s for s in samples if s.split == args.split]

    # 4. Setup output
    out_dir = args.output / spec.id
    predictions_path = out_dir / "predictions.jsonl"
    completed = read_completed(predictions_path) if args.resume else set()

    # 5. Load model once
    print(f"Loading {spec.display_name} ...")
    adapter.load()
    print(f"  Running {len(samples)} samples ({'resume' if args.resume else 'fresh'})")

    try:
        for sample in samples:
            if sample.sample_id in completed:
                print(f"  SKIP {sample.sample_id}")
                continue

            print(f"  RUN  {sample.sample_id} ...", end=" ", flush=True)
            started = time.perf_counter()

            try:
                image = Image.open(sample.image_path).convert("RGB")
                prediction_raw = adapter.predict(image)

                record = build_record(
                    model_id=model_id,
                    sample_id=sample.sample_id,
                    split=sample.split,
                    reference=sample.reference,
                    prediction_raw=prediction_raw,
                    latency=time.perf_counter() - started,
                )
            except Exception as exc:
                record = build_record(
                    model_id=model_id,
                    sample_id=sample.sample_id,
                    split=sample.split,
                    reference=sample.reference,
                    prediction_raw="",
                    latency=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )

            append_record(predictions_path, record)
            dur = time.perf_counter() - started
            cerr = record.get("cer_norm", "?")
            print(f"{dur:.1f}s  CER={cerr}")
    finally:
        adapter.close()
        del adapter
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # 6. Generate CSV outputs
    write_scores_csv(predictions_path)
    print(f"\nResults written to {out_dir}/")
    print(f"  predictions.jsonl, scores.csv, summary.csv")


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator — spawns a subprocess per model
# ═══════════════════════════════════════════════════════════════════════

def run_isolated(args: argparse.Namespace, model_id: str) -> int:
    """Run one model in a fresh subprocess for clean GPU isolation."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model", model_id,
        "--dataset", str(args.dataset),
        "--output", str(args.output),
        "--cache-dir", str(args.cache_dir),
        "--manifest", str(args.manifest),
        "--device", args.device,
        "--split", args.split,
    ]
    if args.auto_pull:
        cmd.append("--auto-pull")
    if args.resume:
        cmd.append("--resume")

    print(f"\n{'=' * 70}")
    print(f"Benchmarking {model_id}")
    print(f"{'=' * 70}")

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Persian OCR benchmark."
    )
    parser.add_argument(
        "--model", "--models", dest="models",
        required=True,
        help="Model ID, comma-separated IDs, or 'all'.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--split", choices=["all", "typed", "hand-written"], default="all"
    )
    parser.add_argument("--auto-pull", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        args.model = args.models
        run_worker(args)
        return

    model_ids = parse_model_ids(args.models)
    validate_model_ids(model_ids)

    failed: list[str] = []
    for mid in model_ids:
        rc = run_isolated(args, mid)
        if rc != 0:
            failed.append(mid)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print("All models completed.")


if __name__ == "__main__":
    main()
