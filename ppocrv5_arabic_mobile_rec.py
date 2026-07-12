#!/usr/bin/env python3
"""Pull and benchmark PP-OCRv5 Arabic Mobile Recognition on smoke20.

Full pages are processed by an explicit PP-OCRv5 mobile detector followed by
the Arabic-script recognizer. Detector and recognizer identity are recorded
separately; reported latency is end-to-end pipeline latency.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

from tesseract_fas import (
    IMAGE_EXTENSIONS,
    REPO_ROOT,
    edit_statistics,
    load_ground_truth,
    load_manifest,
    metadata_breakdowns,
    normalize_fa,
    normalize_transport,
    orthographic_diagnostics,
    punctuation_diagnostics,
    sha256_file,
    summarize_records,
    track_for_subdir,
    unicode_variant_diagnostics,
    cer,
    wer,
)
from tesseract_preprocess import PROFILES, preprocess_image


MODEL_ID = "ppocrv5_arabic_mobile_rec"
RECOGNIZER_REPO = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec"
DETECTOR_REPO = "PaddlePaddle/PP-OCRv5_mobile_det"
MODEL_ROOT = REPO_ROOT / "models" / MODEL_ID
RECOGNIZER_DIR = MODEL_ROOT / "recognizer"
DETECTOR_DIR = MODEL_ROOT / "detector"
DEFAULT_OUTPUT = REPO_ROOT / "bench_runs" / f"{MODEL_ID}.json"
REQUIRED_MODEL_FILES = {"config.json", "inference.json", "inference.pdiparams"}
LEADERBOARD_SCHEMA = "persian_ocr_benchmark_v1"


@dataclass
class PaddleResult:
    subdir: str
    track: str
    image: str
    reference_source: str
    reference_quality: str
    page_metadata: dict[str, object]
    device: str
    preprocess: str
    seconds: float
    image_sha256: str
    reference_sha256: str
    text_raw: str
    text_canonical: str
    recognized_lines: list[str]
    line_confidences: list[float]
    boxes: list[object]
    mean_confidence: Optional[float]
    detected_regions: int
    cer_codepoint_strict: Optional[float]
    cer_grapheme_strict: Optional[float]
    cer_grapheme_canonical: Optional[float]
    cer_grapheme_search: Optional[float]
    wer_canonical: Optional[float]
    canonical_ref_graphemes: Optional[int]
    canonical_hyp_graphemes: Optional[int]
    canonical_insertions: Optional[int]
    canonical_deletions: Optional[int]
    canonical_substitutions: Optional[int]
    canonical_edit_distance: Optional[int]
    diagnostics: dict[str, Optional[float] | int]
    unicode_form_diagnostics: dict[str, Optional[float] | int]
    punctuation_diagnostics: dict[str, dict[str, int]]
    raw_pipeline_result: dict[str, object]
    failure_image_path: Optional[str]
    error: Optional[str] = None


def package_versions() -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {}
    for name in ("paddleocr", "paddlepaddle", "paddlex", "huggingface-hub"):
        try:
            result[name] = package_version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def directory_identity(path: Path, repo_id: str, revision: str) -> dict[str, object]:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": str(path),
        "size_bytes": sum(file.stat().st_size for file in files),
        "files": {
            str(file.relative_to(path)).replace("\\", "/"): {
                "size_bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        },
    }


def pull_repo(repo_id: str, local_dir: Path, force: bool) -> dict[str, object]:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=local_dir,
        force_download=force,
    )
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file())
    if missing:
        raise RuntimeError(f"{repo_id} is missing required files: {missing}")
    return directory_identity(local_dir, repo_id, info.sha)


def cmd_pull(args: argparse.Namespace) -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    identities = {
        "recognizer": pull_repo(RECOGNIZER_REPO, RECOGNIZER_DIR, args.force),
        "detector": pull_repo(DETECTOR_REPO, DETECTOR_DIR, args.force),
    }
    identity_path = MODEL_ROOT / "model_identity.json"
    identity_path.write_text(
        json.dumps(identities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Models ready: {MODEL_ROOT}")
    print(f"Identity:     {identity_path}")
    return 0


def ensure_models() -> dict[str, object]:
    identity_path = MODEL_ROOT / "model_identity.json"
    if not all((RECOGNIZER_DIR / name).is_file() for name in REQUIRED_MODEL_FILES):
        cmd_pull(argparse.Namespace(force=False))
    if not all((DETECTOR_DIR / name).is_file() for name in REQUIRED_MODEL_FILES):
        cmd_pull(argparse.Namespace(force=False))
    if identity_path.is_file():
        return json.loads(identity_path.read_text(encoding="utf-8"))
    return {
        "recognizer": directory_identity(RECOGNIZER_DIR, RECOGNIZER_REPO, "unknown"),
        "detector": directory_identity(DETECTOR_DIR, DETECTOR_REPO, "unknown"),
    }


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def result_payload(result) -> dict[str, object]:
    if isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            payload = getattr(result, "res", None)
        if payload is None and hasattr(result, "to_dict"):
            payload = result.to_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(result)!r}")
    payload = json_safe(payload)
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def compact_pipeline_result(payload: dict[str, object]) -> dict[str, object]:
    """Keep reproducible OCR outputs without embedding image tensors in JSON."""
    keep = {
        "input_path",
        "page_index",
        "dt_polys",
        "model_settings",
        "text_det_params",
        "text_type",
        "text_rec_score_thresh",
        "return_word_box",
        "rec_texts",
        "rec_scores",
        "rec_polys",
        "textline_orientation_angles",
        "rec_boxes",
    }
    return {key: value for key, value in payload.items() if key in keep}


def create_pipeline(device: str):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("Run `uv sync` to install pinned PaddleOCR dependencies") from exc
    return PaddleOCR(
        # A local model directory does not override PaddleOCR's default model
        # identity. Keep both names explicit so PaddleOCR validates the
        # downloaded PP-OCRv5 checkpoints against their actual configs.
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(DETECTOR_DIR),
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(RECOGNIZER_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        # PaddlePaddle 3.3.1's CPU oneDNN/PIR path cannot load this model's
        # graph attributes; plain CPU inference is the compatible fallback.
        enable_mkldnn=False if device.startswith("cpu") else None,
        device=device,
    )


def predict_page(pipeline, image: Image.Image) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    outputs = list(pipeline.predict(np.asarray(image)))
    elapsed = time.perf_counter() - started
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one page result, received {len(outputs)}")
    return result_payload(outputs[0]), elapsed


def discover_inputs(args: argparse.Namespace) -> tuple[Path, list[tuple[Path, dict[str, object]]]]:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if manifest_path.exists():
        entries = load_manifest(manifest_path)
    else:
        root = REPO_ROOT / "small_bench"
        entries = [
            (path, {})
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    if args.subdir:
        wanted = set(args.subdir)
        entries = [item for item in entries if item[0].parent.name in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise RuntimeError("No benchmark images matched the selected inputs")
    return manifest_path, entries


def validate_reviewed(entries: list[tuple[Path, dict[str, object]]]) -> None:
    invalid = []
    for image_path, _ in entries:
        sidecar = image_path.with_suffix(".reference.json")
        if not sidecar.is_file():
            invalid.append(f"{sidecar} (missing)")
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if payload.get("quality") not in {"reviewed", "double_reviewed"}:
                invalid.append(f"{sidecar} (quality is not reviewed)")
        except (OSError, json.JSONDecodeError, AttributeError):
            invalid.append(f"{sidecar} (invalid JSON)")
    if invalid:
        raise RuntimeError(
            f"Reviewed references required; invalid {len(invalid)}, first: {invalid[0]}"
        )


def cmd_small_bench(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")
    model_identity = ensure_models()
    manifest_path, entries = discover_inputs(args)
    if args.require_reviewed:
        validate_reviewed(entries)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    print(f"Model:              {RECOGNIZER_REPO}")
    print(f"Detector:           {DETECTOR_REPO}")
    print(f"Device:             {args.device}")
    print(f"Preprocessing:      {args.preprocess}")
    print(f"Images:             {len(entries)}")
    print(f"Results output:     {output_path}")
    print()

    init_started = time.perf_counter()
    pipeline = create_pipeline(args.device)
    initialization_seconds = time.perf_counter() - init_started
    results: list[PaddleResult] = []

    for image_path, manifest_metadata in entries:
        try:
            ref_raw, reference_source, reference_quality, sidecar_metadata = load_ground_truth(
                image_path
            )
            metadata = {**manifest_metadata, **sidecar_metadata}
            with Image.open(image_path) as source:
                processed = preprocess_image(source, PROFILES[args.preprocess])
            raw_result, seconds = predict_page(pipeline, processed)
            lines = [str(text) for text in raw_result.get("rec_texts", [])]
            scores = [float(score) for score in raw_result.get("rec_scores", [])]
            boxes = json_safe(raw_result.get("rec_boxes", raw_result.get("rec_polys", [])))
            hyp_raw = "\n".join(line for line in lines if line.strip())
            error = None
        except Exception as exc:  # noqa: BLE001
            reference_source = str(image_path.with_suffix(".reference.json"))
            reference_quality = "unknown"
            metadata = manifest_metadata
            ref_raw = ""
            raw_result, lines, scores, boxes, hyp_raw = {}, [], [], [], ""
            seconds = 0.0
            error = f"{type(exc).__name__}: {exc}"

        ref_strict = normalize_fa(ref_raw, "strict")
        ref_canonical = normalize_fa(ref_raw, "canonical")
        ref_search = normalize_fa(ref_raw, "search")
        hyp_strict = normalize_fa(hyp_raw, "strict")
        hyp_canonical = normalize_fa(hyp_raw, "canonical")
        hyp_search = normalize_fa(hyp_raw, "search")
        edits = {} if error else edit_statistics(ref_canonical, hyp_canonical)
        result = PaddleResult(
            subdir=image_path.parent.name,
            track=str(metadata.get("track", track_for_subdir(image_path.parent.name))),
            image=image_path.name,
            reference_source=reference_source,
            reference_quality=reference_quality,
            page_metadata=metadata,
            device=args.device,
            preprocess=args.preprocess,
            seconds=round(seconds, 4),
            image_sha256=sha256_file(image_path),
            reference_sha256=(
                sha256_file(Path(reference_source))
                if Path(reference_source).is_file()
                else ""
            ),
            text_raw=hyp_raw,
            text_canonical=hyp_canonical,
            recognized_lines=lines,
            line_confidences=scores,
            boxes=boxes,
            mean_confidence=round(statistics.mean(scores), 4) if scores else None,
            detected_regions=len(lines),
            cer_codepoint_strict=(
                None if error else round(cer(ref_strict, hyp_strict, unit="codepoint"), 4)
            ),
            cer_grapheme_strict=(
                None if error else round(cer(ref_strict, hyp_strict), 4)
            ),
            cer_grapheme_canonical=(
                None if error else round(cer(ref_canonical, hyp_canonical), 4)
            ),
            cer_grapheme_search=(
                None if error else round(cer(ref_search, hyp_search), 4)
            ),
            wer_canonical=(
                None if error else round(wer(ref_canonical, hyp_canonical), 4)
            ),
            canonical_ref_graphemes=edits.get("ref_graphemes"),
            canonical_hyp_graphemes=edits.get("hyp_graphemes"),
            canonical_insertions=edits.get("insertions"),
            canonical_deletions=edits.get("deletions"),
            canonical_substitutions=edits.get("substitutions"),
            canonical_edit_distance=edits.get("edit_distance"),
            diagnostics=(
                {} if error else orthographic_diagnostics(ref_canonical, hyp_canonical)
            ),
            unicode_form_diagnostics=(
                {}
                if error
                else unicode_variant_diagnostics(
                    normalize_transport(ref_raw), normalize_transport(hyp_raw)
                )
            ),
            punctuation_diagnostics=(
                {} if error else punctuation_diagnostics(ref_canonical, hyp_canonical)
            ),
            raw_pipeline_result=compact_pipeline_result(raw_result),
            failure_image_path=None,
            error=error,
        )
        results.append(result)
        if error:
            print(f"  [ERR] {result.subdir}/{result.image}: {error}")
        else:
            print(
                f"  [OK ] {result.subdir}/{result.image}  "
                f"CER_canonical={result.cer_grapheme_canonical:.3f}  "
                f"lines={result.detected_regions} conf={result.mean_confidence} "
                f"t={result.seconds:.2f}s"
            )

    successful = [result for result in results if not result.error]
    seconds = sorted(result.seconds for result in successful)
    summary = {
        "benchmark": {
            "schema": LEADERBOARD_SCHEMA,
            "name": "persian_ocr_smoke20",
            "scope": "full-page detector+recognizer benchmark; leaderboard-comparable",
            "reference_quality_counts": dict(
                Counter(result.reference_quality for result in results)
            ),
        },
        "model": {
            "id": MODEL_ID,
            "recognizer": RECOGNIZER_REPO,
            "detector": DETECTOR_REPO,
            "checkpoint_type": "recognition_checkpoint_with_explicit_detector",
            "ordering_policy": "PaddleOCR pipeline output order; RTL reading order not separately scored",
            "device": args.device,
            "preprocess": PROFILES[args.preprocess].to_dict(),
            "identity": model_identity,
        },
        "config": {
            "manifest": str(manifest_path) if manifest_path.exists() else None,
            "require_reviewed": args.require_reviewed,
            "limit": args.limit,
        },
        "n_images": len(results),
        "n_ok": len(successful),
        "n_err": len(results) - len(successful),
        "primary_results": summarize_records(results),
        "track_breakdowns": {
            track: summarize_records([result for result in results if result.track == track])
            for track in sorted({result.track for result in results})
        },
        "metadata_breakdowns": metadata_breakdowns(results),
        "operations": {
            "latency_scope": "end-to-end detector+recognizer pipeline",
            "initialization_seconds": round(initialization_seconds, 4),
            "mean_seconds_per_run": round(statistics.mean(seconds), 4) if seconds else None,
            "median_seconds_per_run": round(statistics.median(seconds), 4) if seconds else None,
            "p95_seconds_per_run": (
                round(seconds[round((len(seconds) - 1) * 0.95)], 4) if seconds else None
            ),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(result) for result in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")
    if args.show_failures and successful:
        print("\nWorst primary-configuration runs:")
        for result in sorted(
            successful,
            key=lambda item: item.cer_grapheme_canonical or -1,
            reverse=True,
        )[:5]:
            print(
                f"  {result.subdir}/{result.image}  "
                f"CER={result.cer_grapheme_canonical:.3f} "
                f"WER={result.wer_canonical:.3f}"
            )
    return 2 if results and not successful else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull and benchmark PP-OCRv5 Arabic Mobile Recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pull", action="store_true")
    mode.add_argument("--small_bench", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force model re-download.")
    parser.add_argument("--device", default="cpu", help="PaddleOCR device, e.g. cpu or gpu:0.")
    parser.add_argument("--preprocess", choices=sorted(PROFILES), default="raw")
    parser.add_argument("--manifest", default="small_bench/manifest.jsonl")
    parser.add_argument("--subdir", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT))
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_pull(args) if args.pull else cmd_small_bench(args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
