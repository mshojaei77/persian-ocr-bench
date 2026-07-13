#!/usr/bin/env python3
"""Pull and benchmark the EasyOCR Persian detector/recognizer pipeline."""

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

try:
    from .tesseract_fas import (
        IMAGE_EXTENSIONS,
        PROFILES,
        REPO_ROOT,
        cer,
        edit_statistics,
        load_ground_truth,
        load_manifest,
        metadata_breakdowns,
        normalize_fa,
        normalize_transport,
        orthographic_diagnostics,
        preprocess_image,
        punctuation_diagnostics,
        sha256_file,
        summarize_records,
        track_for_subdir,
        unicode_variant_diagnostics,
        wer,
    )
except ImportError:
    from tesseract_fas import (
        IMAGE_EXTENSIONS,
        PROFILES,
        REPO_ROOT,
        cer,
        edit_statistics,
        load_ground_truth,
        load_manifest,
        metadata_breakdowns,
        normalize_fa,
        normalize_transport,
        orthographic_diagnostics,
        preprocess_image,
        punctuation_diagnostics,
        sha256_file,
        summarize_records,
        track_for_subdir,
        unicode_variant_diagnostics,
        wer,
    )


MODEL_ID = "easyocr_fa"
MODEL_ROOT = REPO_ROOT / "models" / MODEL_ID
IDENTITY_PATH = MODEL_ROOT / "model_identity.json"
DEFAULT_OUTPUT = REPO_ROOT / "bench_runs" / f"{MODEL_ID}.json"
DEFAULT_MANIFEST = "small_bench/manifest.jsonl"
LEADERBOARD_SCHEMA = "persian_ocr_benchmark_v1"


@dataclass
class EasyOCRResult:
    subdir: str
    track: str
    image: str
    reference_source: str
    reference_quality: str
    page_metadata: dict[str, object]
    device: str
    languages: list[str]
    detector: str
    decoder: str
    ordering: str
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


def installed_version(name: str) -> Optional[str]:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def parse_languages(value: str) -> list[str]:
    languages = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if "fa" not in languages:
        raise ValueError("EasyOCR Persian runs must include the fa language code")
    return languages


def device_argument(device: str) -> bool | str:
    if device == "cpu":
        return False
    if device == "auto":
        return True
    return device


def create_reader(
    *, languages: list[str], device: str, detector: str, download_enabled: bool
):
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "Install the adapter with `uv sync --extra easyocr` before running EasyOCR"
        ) from exc
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        languages,
        gpu=device_argument(device),
        model_storage_directory=str(MODEL_ROOT),
        download_enabled=download_enabled,
        detect_network=detector,
        detector=True,
        recognizer=True,
        verbose=True,
    )


def model_identity(reader, languages: list[str], detector: str) -> dict[str, object]:
    files = sorted(
        path
        for path in MODEL_ROOT.rglob("*")
        if path.is_file() and path != IDENTITY_PATH
    )
    return {
        "library": "easyocr",
        "library_version": installed_version("easyocr"),
        "languages": languages,
        "model_language": getattr(reader, "model_lang", None),
        "detector": detector,
        "model_root": str(MODEL_ROOT),
        "files": {
            str(path.relative_to(MODEL_ROOT)).replace("\\", "/"): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }


def write_identity(identity: dict[str, object]) -> None:
    IDENTITY_PATH.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_pull(args: argparse.Namespace) -> int:
    languages = parse_languages(args.langs)
    reader = create_reader(
        languages=languages,
        device=args.device,
        detector=args.detector,
        download_enabled=True,
    )
    identity = model_identity(reader, languages, args.detector)
    write_identity(identity)
    print(f"Models ready: {MODEL_ROOT}")
    print(f"Identity:     {IDENTITY_PATH}")
    return 0


def ensure_models(args: argparse.Namespace, languages: list[str]) -> dict[str, object]:
    if not list(MODEL_ROOT.glob("*.pth")):
        cmd_pull(args)
    if IDENTITY_PATH.is_file():
        try:
            return json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    reader = create_reader(
        languages=languages,
        device=args.device,
        detector=args.detector,
        download_enabled=False,
    )
    identity = model_identity(reader, languages, args.detector)
    write_identity(identity)
    return identity


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def box_geometry(box: object) -> tuple[float, float, float]:
    points = json_safe(box)
    if not isinstance(points, list) or not points:
        return 0.0, 0.0, 1.0
    xs = [float(point[0]) for point in points if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in points if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return 0.0, 0.0, 1.0
    return statistics.mean(xs), statistics.mean(ys), max(max(ys) - min(ys), 1.0)


def rtl_row_order(detections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group nearby boxes into rows, then order each row right-to-left."""
    positioned = [(*box_geometry(item["box"]), item) for item in detections]
    positioned.sort(key=lambda item: item[1])
    rows: list[list[tuple[float, float, float, dict[str, object]]]] = []
    for item in positioned:
        if not rows:
            rows.append([item])
            continue
        row_center = statistics.mean(existing[1] for existing in rows[-1])
        row_height = statistics.median(existing[2] for existing in rows[-1])
        if abs(item[1] - row_center) <= 0.6 * max(row_height, item[2]):
            rows[-1].append(item)
        else:
            rows.append([item])
    ordered: list[dict[str, object]] = []
    for row in rows:
        row.sort(key=lambda item: item[0], reverse=True)
        ordered.extend(item[3] for item in row)
    return ordered


def predict_page(reader, image: Image.Image, args: argparse.Namespace) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    raw = reader.readtext(
        np.asarray(image),
        decoder=args.decoder,
        beamWidth=args.beam_width,
        batch_size=args.batch_size,
        workers=args.workers,
        detail=1,
        paragraph=False,
    )
    elapsed = time.perf_counter() - started
    detections = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise TypeError(f"Unexpected EasyOCR detection: {item!r}")
        detections.append(
            {
                "box": json_safe(item[0]),
                "text": str(item[1]),
                "confidence": float(item[2]),
            }
        )
    if args.ordering == "rtl_rows":
        detections = rtl_row_order(detections)
    return detections, elapsed


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
        entries = [entry for entry in entries if entry[0].parent.name in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise RuntimeError("No benchmark images matched the selected inputs")
    return manifest_path, entries


def validate_reviewed(entries: list[tuple[Path, dict[str, object]]]) -> None:
    unreviewed = [
        str(path)
        for path, metadata in entries
        if metadata.get("reference_quality") != "reviewed"
    ]
    if unreviewed:
        raise RuntimeError(
            "--require-reviewed requested, but the manifest has no reviewed provenance "
            f"for {len(unreviewed)} selected images"
        )
    for path, _ in entries:
        load_ground_truth(path)


def dataset_identity(
    manifest_path: Path, entries: list[tuple[Path, dict[str, object]]]
) -> dict[str, object]:
    corpora = sorted({path.parent.parent / f"{path.parent.name}.json" for path, _ in entries})
    return {
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "reference_corpora": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in corpora
            if path.is_file()
        },
        "n_selected_images": len(entries),
    }


def reference_quality(metadata: dict[str, object]) -> str:
    return str(metadata.get("reference_quality") or "unreviewed_migrated_text")


def cmd_small_bench(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero")
    languages = parse_languages(args.langs)
    identity = ensure_models(args, languages)
    manifest_path, entries = discover_inputs(args)
    if args.require_reviewed:
        validate_reviewed(entries)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    print(f"Model:              EasyOCR {installed_version('easyocr')}")
    print(f"Languages:          {','.join(languages)}")
    print(f"Detector:           {args.detector}")
    print(f"Device:             {args.device}")
    print(f"Ordering:           {args.ordering}")
    print(f"Preprocessing:      {args.preprocess}")
    print(f"Images:             {len(entries)}")
    print(f"Results output:     {output_path}\n")

    init_started = time.perf_counter()
    reader = create_reader(
        languages=languages,
        device=args.device,
        detector=args.detector,
        download_enabled=False,
    )
    initialization_seconds = time.perf_counter() - init_started
    results: list[EasyOCRResult] = []

    for image_path, manifest_metadata in entries:
        try:
            ref_raw, ref_source, _, sidecar_metadata = load_ground_truth(image_path)
            metadata = {**manifest_metadata, **sidecar_metadata}
            with Image.open(image_path) as source:
                processed = preprocess_image(source, PROFILES[args.preprocess])
            detections, seconds = predict_page(reader, processed, args)
            lines = [str(item["text"]) for item in detections if str(item["text"]).strip()]
            scores = [float(item["confidence"]) for item in detections]
            boxes = [item["box"] for item in detections]
            hyp_raw = "\n".join(lines)
            error = None
        except Exception as exc:  # noqa: BLE001
            ref_source = str(image_path.parent.parent / f"{image_path.parent.name}.json")
            metadata = manifest_metadata
            ref_raw = ""
            detections, lines, scores, boxes, hyp_raw = [], [], [], [], ""
            seconds = 0.0
            error = f"{type(exc).__name__}: {exc}"

        ref_strict = normalize_fa(ref_raw, "strict")
        ref_canonical = normalize_fa(ref_raw, "canonical")
        ref_search = normalize_fa(ref_raw, "search")
        hyp_strict = normalize_fa(hyp_raw, "strict")
        hyp_canonical = normalize_fa(hyp_raw, "canonical")
        hyp_search = normalize_fa(hyp_raw, "search")
        edits = {} if error else edit_statistics(ref_canonical, hyp_canonical)
        result = EasyOCRResult(
            subdir=image_path.parent.name,
            track=str(metadata.get("track", track_for_subdir(image_path.parent.name))),
            image=image_path.name,
            reference_source=ref_source,
            reference_quality=reference_quality(metadata),
            page_metadata=metadata,
            device=args.device,
            languages=languages,
            detector=args.detector,
            decoder=args.decoder,
            ordering=args.ordering,
            preprocess=args.preprocess,
            seconds=round(seconds, 4),
            image_sha256=sha256_file(image_path),
            reference_sha256=sha256_file(Path(ref_source)) if Path(ref_source).is_file() else "",
            text_raw=hyp_raw,
            text_canonical=hyp_canonical,
            recognized_lines=lines,
            line_confidences=scores,
            boxes=boxes,
            mean_confidence=round(statistics.mean(scores), 4) if scores else None,
            detected_regions=len(lines),
            cer_codepoint_strict=None if error else round(cer(ref_strict, hyp_strict, unit="codepoint"), 4),
            cer_grapheme_strict=None if error else round(cer(ref_strict, hyp_strict), 4),
            cer_grapheme_canonical=None if error else round(cer(ref_canonical, hyp_canonical), 4),
            cer_grapheme_search=None if error else round(cer(ref_search, hyp_search), 4),
            wer_canonical=None if error else round(wer(ref_canonical, hyp_canonical), 4),
            canonical_ref_graphemes=edits.get("ref_graphemes"),
            canonical_hyp_graphemes=edits.get("hyp_graphemes"),
            canonical_insertions=edits.get("insertions"),
            canonical_deletions=edits.get("deletions"),
            canonical_substitutions=edits.get("substitutions"),
            canonical_edit_distance=edits.get("edit_distance"),
            diagnostics={} if error else orthographic_diagnostics(ref_canonical, hyp_canonical),
            unicode_form_diagnostics=(
                {} if error else unicode_variant_diagnostics(
                    normalize_transport(ref_raw), normalize_transport(hyp_raw)
                )
            ),
            punctuation_diagnostics=(
                {} if error else punctuation_diagnostics(ref_canonical, hyp_canonical)
            ),
            raw_pipeline_result={"detections": detections},
            failure_image_path=None,
            error=error,
        )
        results.append(result)
        if error:
            print(f"  [ERR] {result.subdir}/{result.image}: {error}")
        else:
            print(
                f"  [OK ] {result.subdir}/{result.image}  "
                f"CER={result.cer_grapheme_canonical:.3f} "
                f"regions={result.detected_regions} conf={result.mean_confidence} "
                f"t={result.seconds:.2f}s"
            )

    successful = [result for result in results if not result.error]
    timings = sorted(result.seconds for result in successful)
    summary = {
        "benchmark": {
            "schema": LEADERBOARD_SCHEMA,
            "name": "persian_ocr_smoke20",
            "scope": "full-page detector+recognizer baseline; leaderboard-comparable",
            "reference_quality_counts": dict(
                Counter(result.reference_quality for result in results)
            ),
        },
        "model": {
            "id": MODEL_ID,
            "library": "easyocr",
            "languages": languages,
            "detector": args.detector,
            "checkpoint_type": "detector_plus_recognizer_pipeline",
            "ordering_policy": args.ordering,
            "device": args.device,
            "preprocess": PROFILES[args.preprocess].to_dict(),
            "identity": identity,
        },
        "run_identity": {
            "catalog_sha256": sha256_file(REPO_ROOT / "models.yaml"),
            "runner_sha256": sha256_file(Path(__file__)),
            "dataset": dataset_identity(manifest_path, entries),
        },
        "config": {
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
            "require_reviewed": args.require_reviewed,
            "limit": args.limit,
            "decoder": args.decoder,
            "beam_width": args.beam_width,
            "batch_size": args.batch_size,
            "workers": args.workers,
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
            "latency_scope": "end-to-end EasyOCR detector+recognizer pipeline",
            "initialization_seconds": round(initialization_seconds, 4),
            "mean_seconds_per_run": round(statistics.mean(timings), 4) if timings else None,
            "median_seconds_per_run": round(statistics.median(timings), 4) if timings else None,
            "p95_seconds_per_run": (
                round(timings[round((len(timings) - 1) * 0.95)], 4) if timings else None
            ),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: installed_version(name)
                for name in ("easyocr", "torch", "torchvision", "numpy", "pillow")
            },
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
        print("\nWorst runs:")
        for result in sorted(
            successful,
            key=lambda item: item.cer_grapheme_canonical
            if item.cer_grapheme_canonical is not None
            else -1,
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
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pull", action="store_true", help="Download and verify EasyOCR weights.")
    mode.add_argument("--small_bench", action="store_true", help="Run the small benchmark.")
    parser.add_argument("--langs", default="fa,en", help="Comma-separated EasyOCR language codes.")
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda, cuda:0, or mps.")
    parser.add_argument("--detector", choices=["craft", "dbnet18"], default="craft")
    parser.add_argument(
        "--decoder", choices=["greedy", "beamsearch", "wordbeamsearch"], default="greedy"
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ordering", choices=["engine", "rtl_rows"], default="engine")
    parser.add_argument("--preprocess", choices=sorted(PROFILES), default="raw")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--subdir", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(REPO_ROOT)))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_pull(args) if args.pull else cmd_small_bench(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
