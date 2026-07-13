"""Pull and benchmark Tesseract Persian (``fas.traineddata``)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from PIL import Image

from persian_ocr.artifacts import sha256_file
from persian_ocr.paths import REPO_ROOT, logical_path
from persian_ocr.preprocessing import PROFILES

from ._shared import Phase1Adapter, run_phase1


MODEL_ID = "tesseract_fas"
TESSDATA_ROOT = REPO_ROOT / "models" / "tessdata"
DEFAULT_OUTPUT = f"bench_runs/smoke20-v1/{MODEL_ID}.json"
TESSDATA_URLS = {
    "best": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main",
    "fast": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main",
}


def require_tesseract_binary() -> str:
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise RuntimeError(f"TESSERACT_CMD does not exist: {configured_path}")
        return str(configured_path.resolve())

    command = shutil.which("tesseract")
    if not command and os.name == "nt":
        roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ]
        candidates = [
            Path(root) / "Tesseract-OCR" / "tesseract.exe"
            for root in roots
            if root
        ]
        command = next((str(path) for path in candidates if path.is_file()), None)
    if not command:
        raise RuntimeError(
            "'tesseract' binary not found. Install Tesseract and add it to PATH, "
            "or set TESSERACT_CMD to the executable."
        )
    return command


def _pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "Install the adapter with `uv sync --extra tesseract` before running it"
        ) from exc
    return pytesseract


def configure_pytesseract(binary: str) -> Any:
    pytesseract = _pytesseract()
    pytesseract.pytesseract.tesseract_cmd = binary
    return pytesseract


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    if len(data) < 1024:
        raise RuntimeError(
            f"{url} returned a suspiciously small payload ({len(data)} bytes)"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"  -> {destination} ({len(data) / 1024:.1f} KB)")


def tessdata_dir_for(variant: str) -> Path:
    return TESSDATA_ROOT / variant


def _language_codes(languages: list[str]) -> list[str]:
    return sorted(
        {
            code.strip()
            for language in languages
            for code in language.split("+")
            if code.strip()
        }
    )


def ensure_tessdata(languages: list[str], variant: str) -> dict[str, Path]:
    tessdata_dir = tessdata_dir_for(variant)
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for code in _language_codes(languages):
        destination = tessdata_dir / f"{code}.traineddata"
        if not destination.exists():
            download(f"{TESSDATA_URLS[variant]}/{destination.name}", destination)
        files[code] = destination
    return files


def verify_langs(binary: str, languages: list[str], tessdata_dir: Path) -> dict[str, bool]:
    try:
        output = subprocess.run(
            [binary, "--tessdata-dir", str(tessdata_dir), "--list-langs"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {language: False for language in languages}
    available = set(output.split())
    return {language: language in available for language in languages}


def tesseract_version(binary: str, *, full: bool = False) -> str:
    try:
        output = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        return output if full else output.splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cmd_pull(args: argparse.Namespace) -> int:
    binary = require_tesseract_binary()
    languages = ["fas"] + (["eng"] if args.with_eng else [])
    tessdata_dir = tessdata_dir_for(args.variant)
    for language in languages:
        destination = tessdata_dir / f"{language}.traineddata"
        if destination.exists() and not args.force:
            print(f"  already present: {destination}")
        else:
            download(f"{TESSDATA_URLS[args.variant]}/{destination.name}", destination)
    availability = verify_langs(binary, languages, tessdata_dir)
    print(f"tesseract version: {tesseract_version(binary)}")
    for language, available in availability.items():
        print(f"  {language:>4}: {'OK' if available else 'MISSING'}")
    if not all(availability.values()):
        raise RuntimeError("Tesseract could not load every downloaded language")
    return 0


def _single_config(args: argparse.Namespace) -> tuple[str, int]:
    languages = [item.strip() for item in args.langs.split(",") if item.strip()]
    try:
        psms = [int(item.strip()) for item in args.psm.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--psm must be a comma-separated list of integers") from exc
    if len(languages) != 1 or len(psms) != 1:
        raise ValueError(
            "Artifact v2 records one comparable configuration per run; pass one "
            "--langs value and one --psm value, then run other configurations separately"
        )
    if psms[0] < 0 or psms[0] > 13:
        raise ValueError("--psm must be an integer from 0 through 13")
    if args.oem != 1:
        raise ValueError("tessdata_best and tessdata_fast require --oem 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.failure_cer_threshold < 0:
        raise ValueError("--failure-cer-threshold must be non-negative")
    return languages[0], psms[0]


def prepare_runtime(args: argparse.Namespace) -> tuple[object, dict[str, object], float]:
    language, _ = _single_config(args)
    binary = require_tesseract_binary()
    started = time.perf_counter()
    pytesseract = configure_pytesseract(binary)
    files = ensure_tessdata([language], args.variant)
    tessdata_dir = tessdata_dir_for(args.variant)
    availability = verify_langs(binary, _language_codes([language]), tessdata_dir)
    missing = [code for code, available in availability.items() if not available]
    if missing:
        raise RuntimeError(f"Tesseract could not load languages: {missing}")
    identity = {
        "engine": {
            "name": Path(binary).name,
            "version": tesseract_version(binary, full=True),
        },
        "language_components": {
            code: {
                "variant": args.variant,
                "path": logical_path(path, base=REPO_ROOT),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for code, path in sorted(files.items())
        },
    }
    runtime = {
        "pytesseract": pytesseract,
        "tessdata_dir": tessdata_dir,
    }
    return runtime, identity, time.perf_counter() - started


def predict_page(
    runtime: object, image: Image.Image, args: argparse.Namespace
) -> tuple[str, float, dict[str, object]]:
    language, psm = _single_config(args)
    data = runtime  # type: ignore[assignment]
    pytesseract = data["pytesseract"]
    tessdata_dir: Path = data["tessdata_dir"]
    # pytesseract passes Windows quote characters through as part of the
    # tessdata directory argument. This workspace path contains no spaces, so
    # pass the resolved value directly and keep the command portable here.
    config = f"--tessdata-dir {tessdata_dir} --oem {args.oem} --psm {psm}"
    started = time.perf_counter()
    try:
        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=config,
            timeout=int(args.timeout),
        )
        raw: dict[str, object] = {
            "config": f"--oem {args.oem} --psm {psm}",
            "language": language,
        }
        if args.save_tsv:
            tsv = pytesseract.image_to_data(
                image,
                lang=language,
                config=config,
                timeout=int(args.timeout),
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                float(confidence)
                for confidence, token in zip(tsv["conf"], tsv["text"])
                if str(token).strip() and float(confidence) >= 0
            ]
            raw["tsv_diagnostics"] = {
                "mean_word_confidence": (
                    round(statistics.mean(confidences), 6) if confidences else None
                ),
                "recognized_words": len(confidences),
                "detected_blocks": len(
                    {
                        block
                        for block, token in zip(tsv["block_num"], tsv["text"])
                        if str(token).strip()
                    }
                ),
                "tsv": tsv,
                "extra_tesseract_pass": True,
            }
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"Tesseract error: {exc}") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Tesseract timeout: {exc}") from exc
    return text, time.perf_counter() - started, raw


def declared_identity(args: argparse.Namespace) -> dict[str, object]:
    language, _ = _single_config(args)
    components: dict[str, object] = {}
    for code in _language_codes([language]):
        path = tessdata_dir_for(args.variant) / f"{code}.traineddata"
        components[code] = (
            {
                "variant": args.variant,
                "path": logical_path(path, base=REPO_ROOT),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file()
            else {"variant": args.variant, "status": "not_present"}
        )
    return {
        "engine": {"name": "tesseract", "version": None},
        "language_components": components,
    }


def adapter(args: argparse.Namespace) -> Phase1Adapter:
    language, psm = _single_config(args)
    return Phase1Adapter(
        model={
            "id": MODEL_ID,
            "class": "full_page_detector_recognizer_pipeline",
            "checkpoint_type": "integrated_page_segmentation_recognizer",
            "identity": declared_identity(args),
        },
        prepare=lambda: prepare_runtime(args),
        predict=lambda runtime, image: predict_page(runtime, image, args),
        config={
            "language": language,
            "oem": args.oem,
            "psm": psm,
            "variant": args.variant,
            "timeout_seconds": args.timeout,
            "preprocess": PROFILES[args.preprocess].to_dict(),
            "save_tsv": args.save_tsv,
        },
        packages=("Pillow", "pytesseract", "numpy", "opencv-python-headless"),
        latency_scope=(
            "Tesseract recognition plus optional second TSV diagnostics pass"
            if args.save_tsv
            else "end-to-end Tesseract page recognition"
        ),
    )


def cmd_small_bench(args: argparse.Namespace) -> int:
    return run_phase1(args, adapter(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pull", action="store_true", help="Download Persian tessdata and verify it.")
    mode.add_argument("--small_bench", action="store_true", help="Run the Phase 1 smoke screen.")
    parser.add_argument("--with-eng", action="store_true")
    parser.add_argument("--variant", choices=["best", "fast"], default="best")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--langs", default="fas")
    parser.add_argument("--oem", type=int, default=1)
    parser.add_argument("--psm", default="6")
    parser.add_argument("--subdir", nargs="*", default=None)
    parser.add_argument("--manifest", default="small_bench/manifest.jsonl")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--preprocess", choices=sorted(PROFILES), default="raw")
    parser.add_argument("--save-tsv", action="store_true")
    parser.add_argument("--save-failure-images", action="store_true")
    parser.add_argument("--failure-cer-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-reviewed", action="store_true")
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
    "configure_pytesseract",
    "download",
    "ensure_tessdata",
    "main",
    "predict_page",
    "require_tesseract_binary",
    "tesseract_version",
    "verify_langs",
]
