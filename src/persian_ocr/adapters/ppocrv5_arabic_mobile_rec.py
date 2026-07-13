"""Pull and benchmark the PP-OCRv5 Arabic detector/recognizer pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Optional

from PIL import Image

from persian_ocr.artifacts import sha256_file
from persian_ocr.paths import REPO_ROOT, logical_path
from persian_ocr.preprocessing import PROFILES

from ._shared import Phase1Adapter, run_phase1


MODEL_ID = "ppocrv5_arabic_mobile_rec"
RECOGNIZER_REPO = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec"
DETECTOR_REPO = "PaddlePaddle/PP-OCRv5_mobile_det"
MODEL_ROOT = REPO_ROOT / "models" / MODEL_ID
RECOGNIZER_DIR = MODEL_ROOT / "recognizer"
DETECTOR_DIR = MODEL_ROOT / "detector"
IDENTITY_PATH = MODEL_ROOT / "model_identity.json"
DEFAULT_OUTPUT = f"bench_runs/smoke20-v1/{MODEL_ID}.json"
REQUIRED_MODEL_FILES = {"config.json", "inference.json", "inference.pdiparams"}


def directory_identity(path: Path, repo_id: str, revision: str) -> dict[str, object]:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": logical_path(path, base=REPO_ROOT),
        "size_bytes": sum(file.stat().st_size for file in files),
        "files": {
            file.relative_to(path).as_posix(): {
                "size_bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        },
    }


def pull_repo(repo_id: str, local_dir: Path, force: bool) -> dict[str, object]:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the adapter with `uv sync --extra paddle` before pulling models"
        ) from exc
    info = HfApi().model_info(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=local_dir,
        force_download=force,
    )
    missing = sorted(
        name for name in REQUIRED_MODEL_FILES if not (local_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"{repo_id} is missing required files: {missing}")
    return directory_identity(local_dir, repo_id, info.sha)


def cmd_pull(args: argparse.Namespace) -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    identities = {
        "recognizer": pull_repo(RECOGNIZER_REPO, RECOGNIZER_DIR, args.force),
        "detector": pull_repo(DETECTOR_REPO, DETECTOR_DIR, args.force),
    }
    IDENTITY_PATH.write_text(
        json.dumps(identities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Models ready: {MODEL_ROOT}")
    print(f"Identity:     {IDENTITY_PATH}")
    return 0


def _model_files_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_MODEL_FILES)


def ensure_models() -> dict[str, object]:
    if not _model_files_ready(RECOGNIZER_DIR) or not _model_files_ready(DETECTOR_DIR):
        cmd_pull(argparse.Namespace(force=False))
    saved: dict[str, Any] = {}
    if IDENTITY_PATH.is_file():
        try:
            payload = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                saved = payload
        except (OSError, json.JSONDecodeError):
            pass
    identity = {
        "recognizer": directory_identity(
            RECOGNIZER_DIR,
            RECOGNIZER_REPO,
            str(saved.get("recognizer", {}).get("revision") or "unknown"),
        ),
        "detector": directory_identity(
            DETECTOR_DIR,
            DETECTOR_REPO,
            str(saved.get("detector", {}).get("revision") or "unknown"),
        ),
    }
    IDENTITY_PATH.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return identity


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
    if isinstance(value, Path):
        return value.as_posix()
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def result_payload(result: object) -> dict[str, object]:
    if isinstance(result, dict):
        payload: Any = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if payload is None:
            payload = getattr(result, "res", None)
        if payload is None and hasattr(result, "to_dict"):
            payload = result.to_dict()  # type: ignore[attr-defined]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(result)!r}")
    payload = json_safe(payload)
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def compact_pipeline_result(payload: dict[str, object]) -> dict[str, object]:
    """Keep reproducible OCR outputs without embedding page image tensors."""
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
        raise RuntimeError(
            "Install the adapter with `uv sync --extra paddle` before running PaddleOCR"
        ) from exc
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(DETECTOR_DIR),
        text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(RECOGNIZER_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        # PaddlePaddle 3.3.1's CPU oneDNN/PIR path cannot load these graph
        # attributes. Plain CPU inference is the proven compatibility fallback.
        enable_mkldnn=False if device.startswith("cpu") else None,
        device=device,
    )


def prepare_runtime(args: argparse.Namespace) -> tuple[object, dict[str, object], float]:
    identity = ensure_models()
    started = time.perf_counter()
    pipeline = create_pipeline(args.device)
    return pipeline, identity, time.perf_counter() - started


def predict_page(
    pipeline: object, image: Image.Image
) -> tuple[str, float, dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PaddleOCR requires NumPy; run `uv sync --extra paddle`") from exc
    started = time.perf_counter()
    outputs = list(pipeline.predict(np.asarray(image)))  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - started
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one page result, received {len(outputs)}")
    payload = result_payload(outputs[0])
    lines = [str(text) for text in payload.get("rec_texts", [])]
    confidences = [float(score) for score in payload.get("rec_scores", [])]
    compact = compact_pipeline_result(payload)
    compact.update(
        detected_regions=len([line for line in lines if line.strip()]),
        mean_confidence=(
            round(statistics.mean(confidences), 6) if confidences else None
        ),
    )
    return "\n".join(line for line in lines if line.strip()), elapsed, compact


def declared_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "recognizer": {"repo_id": RECOGNIZER_REPO, "path": "models/ppocrv5_arabic_mobile_rec/recognizer"},
        "detector": {"repo_id": DETECTOR_REPO, "path": "models/ppocrv5_arabic_mobile_rec/detector"},
    }
    if _model_files_ready(RECOGNIZER_DIR):
        identity["recognizer"] = directory_identity(
            RECOGNIZER_DIR, RECOGNIZER_REPO, "unknown"
        )
    if _model_files_ready(DETECTOR_DIR):
        identity["detector"] = directory_identity(DETECTOR_DIR, DETECTOR_REPO, "unknown")
    return identity


def adapter(args: argparse.Namespace) -> Phase1Adapter:
    return Phase1Adapter(
        model={
            "id": MODEL_ID,
            "class": "full_page_detector_recognizer_pipeline",
            "checkpoint_type": "recognition_checkpoint_with_explicit_detector",
            "identity": declared_identity(),
        },
        prepare=lambda: prepare_runtime(args),
        predict=predict_page,
        config={
            "device": args.device,
            "preprocess": PROFILES[args.preprocess].to_dict(),
            "ordering_policy": "PaddleOCR pipeline output order",
            "recognizer_repo": RECOGNIZER_REPO,
            "detector_repo": DETECTOR_REPO,
        },
        packages=("paddleocr", "paddlepaddle", "paddlex", "huggingface-hub", "numpy"),
        latency_scope="end-to-end PP-OCRv5 detector+recognizer pipeline",
    )


def cmd_small_bench(args: argparse.Namespace) -> int:
    return run_phase1(args, adapter(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return cmd_pull(args) if args.pull else cmd_small_bench(args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_ID",
    "build_parser",
    "cmd_pull",
    "cmd_small_bench",
    "compact_pipeline_result",
    "create_pipeline",
    "main",
    "predict_page",
    "result_payload",
]
