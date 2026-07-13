"""Pull and benchmark the EasyOCR Persian full-page pipeline."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import statistics
import time
from typing import Any, Optional

from PIL import Image

from persian_ocr.artifacts import sha256_file
from persian_ocr.devices import DEFAULT_DEVICE, resolve_torch_device
from persian_ocr.paths import REPO_ROOT
from persian_ocr.preprocessing import PROFILES

from ._shared import Phase1Adapter, run_phase1


MODEL_ID = "easyocr_fa"
MODEL_ROOT = REPO_ROOT / "models" / MODEL_ID
IDENTITY_PATH = MODEL_ROOT / "model_identity.json"
DEFAULT_MANIFEST = "small_bench/manifest.jsonl"
DEFAULT_OUTPUT = f"bench_runs/smoke20-v1/{MODEL_ID}.json"


def installed_version(name: str) -> Optional[str]:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def parse_languages(value: str) -> list[str]:
    languages = list(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
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
    resolved_device = resolve_torch_device(device)
    return easyocr.Reader(
        languages,
        gpu=device_argument(resolved_device),
        model_storage_directory=str(MODEL_ROOT),
        download_enabled=download_enabled,
        detect_network=detector,
        detector=True,
        recognizer=True,
        verbose=True,
    )


def _component_files() -> dict[str, dict[str, object]]:
    if not MODEL_ROOT.exists():
        return {}
    return {
        path.relative_to(MODEL_ROOT).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(MODEL_ROOT.rglob("*"))
        if path.is_file() and path != IDENTITY_PATH
    }


def model_identity(
    reader: object | None, languages: list[str], detector: str
) -> dict[str, object]:
    """Describe both EasyOCR components without embedding machine-local paths."""
    return {
        "library": "easyocr",
        "library_version": installed_version("easyocr"),
        "languages": languages,
        "recognizer": {
            "model_language": getattr(reader, "model_lang", None),
            "component": "EasyOCR recognition network",
        },
        "detector": {
            "name": detector,
            "component": "EasyOCR text detector",
        },
        "files": _component_files(),
    }


def write_identity(identity: dict[str, object]) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    IDENTITY_PATH.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
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
    xs = [
        float(point[0])
        for point in points
        if isinstance(point, list) and len(point) >= 2
    ]
    ys = [
        float(point[1])
        for point in points
        if isinstance(point, list) and len(point) >= 2
    ]
    if not xs or not ys:
        return 0.0, 0.0, 1.0
    return statistics.mean(xs), statistics.mean(ys), max(max(ys) - min(ys), 1.0)


def rtl_row_order(
    detections: list[dict[str, object]],
) -> list[dict[str, object]]:
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


def prepare_runtime(args: argparse.Namespace) -> tuple[object, dict[str, object], float]:
    languages = parse_languages(args.langs)
    download_enabled = not list(MODEL_ROOT.glob("*.pth"))
    started = time.perf_counter()
    resolved_device = resolve_torch_device(args.device)
    reader = create_reader(
        languages=languages,
        device=resolved_device,
        detector=args.detector,
        download_enabled=download_enabled,
    )
    elapsed = time.perf_counter() - started
    identity = model_identity(reader, languages, args.detector)
    identity["runtime_device"] = resolved_device
    write_identity(identity)
    return reader, identity, elapsed


def predict_page(
    reader: object, image: Image.Image, args: argparse.Namespace
) -> tuple[str, float, dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("EasyOCR requires NumPy; run `uv sync --extra easyocr`") from exc
    started = time.perf_counter()
    raw = reader.readtext(  # type: ignore[attr-defined]
        np.asarray(image),
        decoder=args.decoder,
        beamWidth=args.beam_width,
        batch_size=args.batch_size,
        workers=args.workers,
        detail=1,
        paragraph=False,
    )
    elapsed = time.perf_counter() - started
    detections: list[dict[str, object]] = []
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
    lines = [str(item["text"]) for item in detections if str(item["text"]).strip()]
    confidences = [float(item["confidence"]) for item in detections]
    return (
        "\n".join(lines),
        elapsed,
        {
            "detections": detections,
            "detected_regions": len(lines),
            "mean_confidence": (
                round(statistics.mean(confidences), 6) if confidences else None
            ),
            "ordering": args.ordering,
        },
    )


def declared_model(args: argparse.Namespace) -> dict[str, object]:
    languages = parse_languages(args.langs)
    return {
        "id": MODEL_ID,
        "class": "full_page_detector_recognizer_pipeline",
        "checkpoint_type": "detector_plus_recognizer_pipeline",
        "identity": model_identity(None, languages, args.detector),
    }


def adapter(args: argparse.Namespace) -> Phase1Adapter:
    languages = parse_languages(args.langs)
    return Phase1Adapter(
        model=declared_model(args),
        prepare=lambda: prepare_runtime(args),
        predict=lambda runtime, image: predict_page(runtime, image, args),
        config={
            "languages": languages,
            "device": args.device,
            "device_policy": "gpu_first_auto" if args.device == DEFAULT_DEVICE else "explicit",
            "detector": args.detector,
            "decoder": args.decoder,
            "beam_width": args.beam_width,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "ordering": args.ordering,
            "preprocess": PROFILES[args.preprocess].to_dict(),
        },
        packages=("easyocr", "torch", "torchvision", "numpy", "Pillow"),
        latency_scope="end-to-end EasyOCR detector+recognizer pipeline",
    )


def cmd_small_bench(args: argparse.Namespace) -> int:
    return run_phase1(args, adapter(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pull", action="store_true", help="Download and verify EasyOCR weights.")
    mode.add_argument("--small_bench", action="store_true", help="Run the Phase 1 smoke screen.")
    parser.add_argument("--langs", default="fa,en", help="Comma-separated EasyOCR language codes.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="auto (CUDA, MPS, then CPU), cpu, cuda, cuda:0, or mps.",
    )
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_pull(args) if args.pull else cmd_small_bench(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_ID",
    "build_parser",
    "cmd_pull",
    "cmd_small_bench",
    "create_reader",
    "device_argument",
    "main",
    "parse_languages",
    "predict_page",
    "rtl_row_order",
]
