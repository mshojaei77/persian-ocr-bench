"""
Surya OCR 2 benchmark — diagnostic, run, and score.

Uses ``scripts/ocr_bench.py`` for all scoring and normalization.
Adds model-source / backend / device / dtype logging so we can verify
that predictions come from the intended local weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from scripts.ocr_bench import (
    bench_items,
    normalize_text,
    normalize_text_strict,
    repo_path,
    score_predictions,
    serializable,
)

PROMPT_NOTE = "Surya OCR 2 page OCR via surya.recognition.RecognitionPredictor"


# ── HTML text extraction ────────────────────────────────────────────


def extract_surya_text(page_result) -> str:
    """
    Extract visible text from a Surya 2 ``PageResult``.

    Uses BeautifulSoup for robust HTML parsing (handles nested tags,
    tables, math elements correctly).
    """
    parts: list[str] = []

    # Surya 2 uses `blocks` (not the obsolete `text_lines`)
    blocks = getattr(page_result, "blocks", None) or (
        page_result.get("blocks") if isinstance(page_result, dict) else []
    )

    for block in sorted(
        blocks,
        key=lambda b: (
            getattr(b, "reading_order", None)
            if not isinstance(b, dict)
            else b.get("reading_order", 0)
        )
        or 0,
    ):
        skipped = (
            getattr(block, "skipped", False)
            if not isinstance(block, dict)
            else block.get("skipped", False)
        )
        error = (
            getattr(block, "error", False)
            if not isinstance(block, dict)
            else block.get("error", False)
        )
        if skipped or error:
            continue

        html = (
            getattr(block, "html", "")
            if not isinstance(block, dict)
            else block.get("html", "")
        )
        if not html:
            continue

        text = BeautifulSoup(html, "html.parser").get_text(
            separator="\n", strip=True
        )
        if text:
            parts.append(text)

    return clean_lines_("\n".join(parts))


def clean_lines_(text: str) -> str:
    """Collapse spaces per line, preserving line breaks."""
    import re

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.replace("\r\n", "\n").split("\n")
    ]
    return "\n".join(line for line in lines if line).strip()


# ── Diagnostics ─────────────────────────────────────────────────────


def _print_raw_diagnostics(prediction, image_path: Path) -> None:
    """Print one sample's raw prediction structure for debugging."""
    import sys

    print("=" * 60, file=sys.stderr)
    print(f"DIAGNOSTIC for: {image_path}", file=sys.stderr)
    print(f"prediction type: {type(prediction).__name__}", file=sys.stderr)

    blocks = getattr(prediction, "blocks", None)
    if blocks is None and isinstance(prediction, dict):
        blocks = prediction.get("blocks", [])

    print(f"num blocks: {len(blocks) if blocks else 0}", file=sys.stderr)

    if blocks:
        for i, block in enumerate(blocks[:5]):  # first 5 blocks
            label = (
                getattr(block, "label", "")
                if not isinstance(block, dict)
                else block.get("label", "")
            )
            html = (
                getattr(block, "html", "")
                if not isinstance(block, dict)
                else block.get("html", "")
            )
            confidence = (
                getattr(block, "confidence", None)
                if not isinstance(block, dict)
                else block.get("confidence", None)
            )
            skipped = (
                getattr(block, "skipped", None)
                if not isinstance(block, dict)
                else block.get("skipped", None)
            )
            error = (
                getattr(block, "error", None)
                if not isinstance(block, dict)
                else block.get("error", None)
            )
            print(
                f"  block[{i}]: label={label!r} skipped={skipped} error={error} "
                f"confidence={confidence} html_len={len(html or '')}",
                file=sys.stderr,
            )
            if html and html.strip():
                first_100 = html.strip()[:100]
                print(f"    html[:100]: {first_100!r}", file=sys.stderr)

    # Print extracted text for quick visual check
    extracted = extract_surya_text(prediction)
    print(f"extracted text ({len(extracted)} chars):", file=sys.stderr)
    print(extracted[:500], file=sys.stderr)
    print("=" * 60, file=sys.stderr)


# ── Backend configuration ───────────────────────────────────────────


def configure_surya_backend(backend: str | None, inference_url: str | None) -> None:
    if backend:
        os.environ["SURYA_INFERENCE_BACKEND"] = backend
    if inference_url:
        os.environ["SURYA_INFERENCE_URL"] = inference_url
    os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")
    os.environ.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "1")

    if os.environ.get("SURYA_INFERENCE_URL"):
        return
    backend = os.environ.get("SURYA_INFERENCE_BACKEND")
    if backend is None and not shutil.which("docker"):
        if shutil.which("llama-server"):
            os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
            return
        raise SystemExit(
            "Surya OCR needs a serving backend. Install Docker for vllm, install "
            "llama-server and run with --backend llamacpp, or pass --inference-url."
        )
    if backend == "vllm" and not shutil.which("docker"):
        raise SystemExit(
            "Surya vllm backend needs Docker. Install Docker or run with "
            "--backend llamacpp."
        )
    if backend == "llamacpp" and not shutil.which("llama-server"):
        raise SystemExit(
            "Surya llama.cpp backend needs llama-server. Install llama.cpp or use "
            "--inference-url."
        )


# ── Run ─────────────────────────────────────────────────────────────


def run_surya(
    image_paths: list[Path],
    output_root: Path,
    backend: str | None,
    inference_url: str | None,
    debug_images: list[Path] | None = None,
) -> None:
    configure_surya_backend(backend, inference_url)

    from PIL import Image
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor

    manager = SuryaInferenceManager()
    predictor = RecognitionPredictor(manager)

    # Log model source and backend so we know what was actually used.
    model_source = getattr(predictor, "model_name", None)
    model_revision = getattr(predictor, "model_revision", None)
    device = getattr(predictor, "device", None)
    dtype = getattr(predictor, "dtype", None)
    print(f"model_source={model_source}")
    print(f"model_revision={model_revision}")
    print(f"backend={os.environ.get('SURYA_INFERENCE_BACKEND', 'auto')}")
    print(f"device={device}")
    print(f"dtype={dtype}")

    debug_set = {p.resolve() for p in (debug_images or [])}

    for image_path in image_paths:
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        prediction = predictor([image])[0]
        elapsed = time.perf_counter() - start

        text = extract_surya_text(prediction)
        (pred_dir / f"{image_path.stem}.md").write_text(text, encoding="utf-8")

        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "prediction": serializable(prediction),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"{split}/{image_path.name}: {elapsed:.1f}s  →  {len(text)} chars")

        # Debug diagnostics for requested images
        if image_path.resolve() in debug_set:
            _print_raw_diagnostics(prediction, image_path)

            # Save debug artifacts
            ref_path = image_path.with_suffix(".md")
            ref_text = ref_path.read_text(encoding="utf-8")
            debug_dir = output_root / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem

            (debug_dir / f"{stem}_raw_model_output.json").write_text(
                json.dumps(
                    {"prediction": serializable(prediction)}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            (debug_dir / f"{stem}_extracted_text.txt").write_text(text, encoding="utf-8")
            (debug_dir / f"{stem}_normalized_prediction.txt").write_text(
                normalize_text(text), encoding="utf-8"
            )
            (debug_dir / f"{stem}_normalized_reference.txt").write_text(
                normalize_text(ref_text), encoding="utf-8"
            )
            print(f"  Debug artifacts saved to {debug_dir}/", file=None)


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and score Surya OCR 2 on small_bench."
    )
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/surya-ocr-2")
    parser.add_argument("--model-name", default="surya-ocr-2")
    parser.add_argument("--backend", choices=["vllm", "llamacpp"])
    parser.add_argument("--inference-url")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--debug",
        nargs="+",
        default=[],
        help="Paths to image(s) for which raw diagnostics + debug artifacts are saved",
    )
    args = parser.parse_args()

    bench_root = repo_path(args.bench_root)
    output_root = repo_path(args.output_root)
    images = bench_items(bench_root)

    if not images:
        raise SystemExit(f"No JPG files found under {bench_root}")

    debug_images = [repo_path(p) for p in args.debug]

    if not args.score_only:
        run_surya(images, output_root, args.backend, args.inference_url, debug_images)
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "note": PROMPT_NOTE,
                    "surya_inference_backend": os.environ.get(
                        "SURYA_INFERENCE_BACKEND", "auto"
                    ),
                    "surya_inference_url": os.environ.get("SURYA_INFERENCE_URL", ""),
                    "surya_inference_parallel": os.environ.get(
                        "SURYA_INFERENCE_PARALLEL", "1"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    score_predictions(bench_root, output_root, args.model_name)


if __name__ == "__main__":
    main()
