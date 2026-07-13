"""Shared Phase 1 adapter runner.

The model adapters own downloads, runtime construction, and prediction parsing.
This module owns the invariant dataset, scoring, failure-accounting, and artifact
contract so every Phase 1 result is directly auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from persian_ocr.artifacts import (
    build_artifact,
    build_environment_identity,
    build_protocol_identity,
    build_result_record,
    build_run_identity,
    build_runner_identity,
    build_summary,
    source_identity,
    write_artifact,
)
from persian_ocr.dataset import (
    HUMAN_REVIEW_STATUSES,
    DatasetSample,
    build_dataset_identity,
    load_dataset,
    require_valid_dataset,
)
from persian_ocr.metrics import metadata_breakdowns, score_text, summarize_records
from persian_ocr.paths import PACKAGE_ROOT, REPO_ROOT, logical_path
from persian_ocr.preprocessing import preprocess_image


AI_ASSISTED_REVIEW_STATUS = "ai_assisted_recovered_not_human_reviewed"
PHASE1_REVIEW_STATUSES = {*HUMAN_REVIEW_STATUSES, AI_ASSISTED_REVIEW_STATUS}


@dataclass(frozen=True)
class Phase1Adapter:
    """Model-owned callbacks and identities consumed by the shared runner."""

    model: Mapping[str, Any]
    prepare: Callable[[], tuple[object, Mapping[str, Any], float]]
    predict: Callable[[object, Image.Image], tuple[str, float, Mapping[str, Any]]]
    config: Mapping[str, Any]
    packages: Sequence[str]
    latency_scope: str


def _manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _output_path(value: str | Path) -> tuple[Path, str]:
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    return path, logical_path(path, base=REPO_ROOT)


def _select_samples(
    samples: Sequence[DatasetSample], args: argparse.Namespace
) -> list[DatasetSample]:
    selected = list(samples)
    if args.subdir:
        wanted = set(args.subdir)
        selected = [sample for sample in selected if sample.image_path.parent.name in wanted]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("No benchmark images matched the selected inputs")
    return selected


def _validate_review_statuses(samples: Sequence[DatasetSample]) -> None:
    """Gate Phase 1 without falsely relabeling AI-assisted text as human-reviewed."""
    rejected = [
        f"{sample.sample_id}={sample.review_status}"
        for sample in samples
        if sample.review_status not in PHASE1_REVIEW_STATUSES
    ]
    if rejected:
        raise RuntimeError(
            "--require-reviewed accepts human review or the explicitly disclosed "
            f"Phase 1 AI-assisted recovery status; rejected {len(rejected)} sample(s), "
            f"first: {rejected[0]}"
        )


def _shared_runner_identity() -> dict[str, Any]:
    runner = build_runner_identity(workspace=REPO_ROOT)
    scorer_files = [
        PACKAGE_ROOT / "artifacts.py",
        PACKAGE_ROOT / "dataset.py",
        PACKAGE_ROOT / "metrics.py",
        PACKAGE_ROOT / "normalization.py",
        PACKAGE_ROOT / "preprocessing.py",
        Path(__file__).resolve(),
    ]
    runner["shared_scorer"] = source_identity(scorer_files, root=PACKAGE_ROOT)
    return runner


def _operations(
    results: Sequence[Mapping[str, Any]],
    *,
    initialization_seconds: float,
    latency_scope: str,
) -> dict[str, Any]:
    timings = sorted(
        float(result["seconds"])
        for result in results
        if result.get("status") == "ok"
    )
    return {
        "latency_scope": latency_scope,
        "initialization_seconds": round(initialization_seconds, 6),
        "mean_seconds_per_page": (
            round(statistics.mean(timings), 6) if timings else None
        ),
        "median_seconds_per_page": (
            round(statistics.median(timings), 6) if timings else None
        ),
        "p95_seconds_per_page": (
            round(timings[round((len(timings) - 1) * 0.95)], 6) if timings else None
        ),
        "peak_vram_gb": None,
        "peak_ram_gb": None,
    }


def _print_failures(results: Sequence[Mapping[str, Any]]) -> None:
    successful = [result for result in results if result.get("status") == "ok"]
    worst = sorted(
        successful,
        key=lambda result: float(
            result.get("metrics", {}).get("cer_grapheme_canonical", -1)
        ),
        reverse=True,
    )[:5]
    if not worst:
        return
    print("\nWorst successful runs:")
    for result in worst:
        metrics = result["metrics"]
        print(
            f"  {result['sample_id']}  "
            f"CER={metrics['cer_grapheme_canonical']:.3f} "
            f"WER={metrics['wer_canonical']:.3f}"
        )


def run_phase1(args: argparse.Namespace, adapter: Phase1Adapter) -> int:
    """Run one model over smoke20 and always account for every selected sample."""
    manifest = _manifest_path(args.manifest)
    if not manifest.is_file():
        raise RuntimeError(f"Manifest does not exist: {manifest}")
    all_samples = load_dataset(manifest, workspace_root=REPO_ROOT)
    require_valid_dataset(all_samples, require_reviewed=False, verify_images=True)
    selected = _select_samples(all_samples, args)
    if args.require_reviewed:
        _validate_review_statuses(selected)

    dataset_identity = build_dataset_identity(
        manifest, all_samples, workspace_root=REPO_ROOT
    )
    protocol = build_protocol_identity(
        "phase1_screening",
        phase="phase1_screening",
        track="full_page_ocr",
        version="2.1",
        purpose="viability_screen_only",
        ranking_policy="none",
        dataset="smoke20-v1",
        metric_contract="fa_ir_phase1_metrics_v1",
        metric_scope=(
            "raw_unicode_cer,persian_normalized_cer,wer,faithfulness,"
            "reading_order,exactness,orthographic,operations"
        ),
    )
    model = dict(adapter.model)
    runtime: object | None = None
    initialization_seconds = 0.0
    preparation_error: str | None = None
    preparation_started = time.perf_counter()
    try:
        runtime, prepared_identity, initialization_seconds = adapter.prepare()
        model["identity"] = dict(prepared_identity)
    except Exception as exc:  # noqa: BLE001 - recorded for every sample below
        initialization_seconds = time.perf_counter() - preparation_started
        preparation_error = f"model initialization failed: {type(exc).__name__}: {exc}"

    config = {
        **dict(adapter.config),
        "manifest": logical_path(manifest, base=REPO_ROOT),
        "selected_sample_ids": [sample.sample_id for sample in selected],
        "selected_count": len(selected),
        "dataset_count": len(all_samples),
        "subdir": list(args.subdir) if args.subdir else None,
        "limit": args.limit,
        "require_reviewed": bool(args.require_reviewed),
        "accepted_review_statuses": sorted(PHASE1_REVIEW_STATUSES),
    }
    run_identity = build_run_identity(
        protocol=protocol,
        dataset=dataset_identity,
        model=model,
        config=config,
        runner=_shared_runner_identity(),
        environment=build_environment_identity(adapter.packages),
    )

    output_path, output_logical = _output_path(args.output)
    print(f"Model:              {model['id']}")
    print(f"Comparison class:   {model['class']}")
    print(f"Protocol:           phase1_screening (no ranks)")
    print(f"Dataset:            {dataset_identity['id']}")
    print(f"Selected images:    {len(selected)}/{len(all_samples)}")
    print(f"Preprocessing:      {args.preprocess}")
    print(f"Results output:     {output_path}\n")

    results: list[dict[str, Any]] = []
    for sample in selected:
        error = preparation_error
        prediction: str | None = None
        raw_output: Mapping[str, Any] | None = None
        processed: Image.Image | None = None
        elapsed = 0.0
        if error is None:
            started = time.perf_counter()
            try:
                with Image.open(sample.image_path) as source:
                    processed = preprocess_image(source, args.preprocess)
                assert runtime is not None
                prediction, elapsed, raw_output = adapter.predict(runtime, processed)
                if not isinstance(prediction, str):
                    raise TypeError("Adapter prediction must be text")
                if not isinstance(raw_output, Mapping):
                    raise TypeError("Adapter raw output must be an object")
            except Exception as exc:  # noqa: BLE001 - per-page failure accounting
                if elapsed <= 0:
                    elapsed = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"

        metadata = {
            **sample.page_metadata,
            "comparison_class": model["class"],
            "reference_quality": sample.reference_quality,
        }
        if error:
            if getattr(args, "save_failure_images", False) and processed is not None:
                failure_path = output_path.parent / "failure_images" / (
                    f"{model['id']}_{sample.sample_id}_{args.preprocess}.png"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                processed.save(failure_path)
                metadata["failure_image"] = logical_path(failure_path, base=REPO_ROOT)
            result = build_result_record(
                sample_id=sample.sample_id,
                image=sample.image,
                split=sample.split,
                track=sample.track,
                content_type=sample.content_type,
                seconds=elapsed,
                error=error,
                reference_source=sample.reference_source,
                metadata=metadata,
            )
            print(f"  [ERR] {sample.sample_id}: {error}")
        else:
            assert prediction is not None
            metrics = score_text(sample.reference, prediction)
            if (
                getattr(args, "save_failure_images", False)
                and processed is not None
                and metrics["cer_grapheme_canonical"]
                >= getattr(args, "failure_cer_threshold", 0.5)
            ):
                failure_path = output_path.parent / "failure_images" / (
                    f"{model['id']}_{sample.sample_id}_{args.preprocess}.png"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                processed.save(failure_path)
                metadata["failure_image"] = logical_path(failure_path, base=REPO_ROOT)
            result = build_result_record(
                sample_id=sample.sample_id,
                image=sample.image,
                split=sample.split,
                track=sample.track,
                content_type=sample.content_type,
                seconds=elapsed,
                reference=sample.reference,
                prediction=prediction,
                metrics=metrics,
                reference_source=sample.reference_source,
                metadata=metadata,
                raw_output=dict(raw_output or {}),
            )
            print(
                f"  [OK ] {sample.sample_id}  "
                f"CER={metrics['cer_grapheme_canonical']:.3f} t={elapsed:.2f}s"
            )
        results.append(result)

    operations = _operations(
        results,
        initialization_seconds=initialization_seconds,
        latency_scope=adapter.latency_scope,
    )
    summary_metrics = summarize_records(results)
    summary_metrics["metadata_breakdowns"] = metadata_breakdowns(results)
    summary = build_summary(
        model=model,
        protocol=protocol,
        dataset=dataset_identity,
        results=results,
        metrics=summary_metrics,
        operations=operations,
    )
    artifact = build_artifact(
        run_identity=run_identity,
        summary=summary,
        results=results,
    )
    written = write_artifact(
        artifact,
        output_root=REPO_ROOT,
        relative_path=output_logical,
    )
    print(f"\nResults saved to: {written}")
    if args.show_failures:
        _print_failures(results)
    return 2 if results and not any(r["status"] == "ok" for r in results) else 0


__all__ = [
    "AI_ASSISTED_REVIEW_STATUS",
    "PHASE1_REVIEW_STATUSES",
    "Phase1Adapter",
    "run_phase1",
]
