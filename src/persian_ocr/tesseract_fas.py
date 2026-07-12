#!/usr/bin/env python3
"""
tesseract_fas.py — Pull and benchmark Tesseract Persian (fas.traineddata).

Mirrors the protocol in models.yaml:
  leaderboards -> raw_recognition, normalized_recognition,
                  persian_failure_slices, operations
  normalization -> Unicode NFC, Arabic ي->Persian ی, Arabic ك->Persian ک,
                   whitespace collapse

Usage:
    python tesseract_fas.py --pull
    python tesseract_fas.py --small_bench
    python tesseract_fas.py --small_bench --langs fas --psm 3
    python tesseract_fas.py --small_bench --show-failures
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import regex
from rapidfuzz.distance import Levenshtein
from .tesseract_preprocess import PROFILES, preprocess_image

try:
    import pytesseract
    from PIL import Image
except ImportError:  # validated by require_pytesseract before use
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


# ---------- paths ----------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESSDATA_ROOT = REPO_ROOT / "tessdata"
SMALL_BENCH = REPO_ROOT / "small_bench"
RESULTS_DIR = REPO_ROOT / "bench_runs"
LEADERBOARD_SCHEMA = "persian_ocr_benchmark_v1"
RESULTS_PATH = RESULTS_DIR / "tesseract_fas.json"

TESSDATA_URLS = {
    "best": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main",
    "fast": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------- Persian / Arabic constants ----------

PERSIAN_YEH = "ی"   # U+06CC
ARABIC_YEH = "ي"    # U+064A
PERSIAN_KAF = "ک"   # U+06A9
ARABIC_KAF = "ك"    # U+0643
ZWNJ = "‌"           # U+200C (zero-width non-joiner)
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"


# ---------- text utilities ----------

_MD_HEADING = re.compile(r"^#+\s*", flags=re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_MD_RULE = re.compile(r"^[-*_]{3,}\s*$", flags=re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def strip_markdown(text: str) -> str:
    """Remove formatting syntax, not bracketed annotation text.

    Bracketed descriptions are intentionally preserved because code cannot know
    whether brackets are visible in the source image. Such regions require a
    human-reviewed annotation sidecar.
    """
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC_STAR.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    text = _HTML_TAG.sub("", text).replace("`", "")
    lines = []
    for line in text.splitlines():
        if _TABLE_SEPARATOR.match(line):
            continue
        if line.count("|") >= 2:
            line = line.strip().strip("|").replace("|", " ")
        lines.append(line)
    return "\n".join(lines).strip()


_BIDI_CONTROLS = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def graphemes(text: str) -> list[str]:
    """Return user-perceived Unicode characters, including combining marks."""
    return regex.findall(r"\X", text or "")


def diagnostic_units(text: str) -> list[str]:
    """Base code points plus standalone ZWNJ for orthographic diagnostics."""
    text = unicodedata.normalize("NFC", text or "")
    return [
        ch for ch in text
        if ch == ZWNJ or not unicodedata.category(ch).startswith("M")
    ]


def normalize_transport(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "")
    return unicodedata.normalize("NFC", text).strip()


def normalize_fa(text: str, policy: str = "canonical") -> str:
    """Apply an explicit scoring policy; preserve ZWNJ in all policies."""
    text = normalize_transport(text)
    if policy == "strict":
        return text
    if policy not in {"canonical", "search"}:
        raise ValueError(f"unknown normalization policy: {policy}")
    text = _BIDI_CONTROLS.sub("", text)
    text = text.replace(ARABIC_YEH, PERSIAN_YEH)
    text = text.replace(ARABIC_KAF, PERSIAN_KAF)
    if policy == "search":
        text = normalize_digits_optional(text)
        text = text.replace("ـ", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_digits_optional(text: str) -> str:
    """Canonicalize Arabic-Indic, Persian and ASCII digits to ASCII."""
    return (text or "").translate(
        str.maketrans(ARABIC_DIGITS + PERSIAN_DIGITS, ASCII_DIGITS + ASCII_DIGITS)
    )


def edit_distance(a, b) -> int:
    """Levenshtein distance over strings or arbitrary hashable sequences."""
    return int(Levenshtein.distance(a, b))


def cer(ref: str, hyp: str, *, unit: str = "grapheme") -> float:
    ref_units = graphemes(ref) if unit == "grapheme" else list(ref)
    hyp_units = graphemes(hyp) if unit == "grapheme" else list(hyp)
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return edit_distance(ref_units, hyp_units) / len(ref_units)


def wer(ref: str, hyp: str) -> float:
    rw = (ref or "").split()
    hw = (hyp or "").split()
    if not rw:
        return 0.0 if not hw else 1.0
    return edit_distance(rw, hw) / len(rw)


def _aligned_units(
    ref: str,
    hyp: str,
    unit_fn=graphemes,
) -> list[tuple[Optional[str], Optional[str]]]:
    a, b = unit_fn(ref), unit_fn(hyp)
    aligned: list[tuple[Optional[str], Optional[str]]] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b):
        if tag == "equal" or tag == "replace":
            width = max(i2 - i1, j2 - j1)
            aligned.extend(
                zip(
                    a[i1:i2] + [None] * (width - (i2 - i1)),
                    b[j1:j2] + [None] * (width - (j2 - j1)),
                )
            )
        elif tag == "delete":
            aligned.extend((item, None) for item in a[i1:i2])
        else:
            aligned.extend((None, item) for item in b[j1:j2])
    return aligned


def orthographic_diagnostics(ref: str, hyp: str) -> dict[str, Optional[float] | int]:
    """Reference-grounded Yeh/Kaf substitution, deletion and insertion rates."""
    aligned = _aligned_units(ref, hyp, diagnostic_units)
    out: dict[str, Optional[float] | int] = {}
    for name, target, alternate in (
        ("yeh", PERSIAN_YEH, ARABIC_YEH),
        ("kaf", PERSIAN_KAF, ARABIC_KAF),
    ):
        ref_count = sum(1 for r, _ in aligned if r == target)
        correct = sum(1 for r, h in aligned if r == target and h == target)
        substitutions = sum(
            1 for r, h in aligned if r == target and h not in {target, None}
        )
        deletions = sum(1 for r, h in aligned if r == target and h is None)
        insertions = sum(1 for r, h in aligned if r is None and h in {target, alternate})
        out[f"{name}_ref_count"] = ref_count
        out[f"{name}_correct"] = correct
        out[f"{name}_substitutions"] = substitutions
        out[f"{name}_deletions"] = deletions
        out[f"{name}_insertions"] = insertions
        out[f"{name}_recall"] = correct / ref_count if ref_count else None
    tp = sum(1 for r, h in aligned if r == ZWNJ and h == ZWNJ)
    fp = sum(1 for r, h in aligned if r != ZWNJ and h == ZWNJ)
    fn = sum(1 for r, h in aligned if r == ZWNJ and h != ZWNJ)
    out.update(
        zwnj_ref_count=tp + fn,
        zwnj_pred_count=tp + fp,
        zwnj_precision=tp / (tp + fp) if tp + fp else None,
        zwnj_recall=tp / (tp + fn) if tp + fn else None,
        zwnj_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )
    return out


def unicode_variant_diagnostics(ref: str, hyp: str) -> dict[str, Optional[float] | int]:
    aligned = _aligned_units(ref, hyp, diagnostic_units)
    output_yeh = [h for _, h in aligned if h in {PERSIAN_YEH, ARABIC_YEH}]
    output_kaf = [h for _, h in aligned if h in {PERSIAN_KAF, ARABIC_KAF}]
    return {
        "output_yeh_count": len(output_yeh),
        "persian_yeh_fraction": (
            output_yeh.count(PERSIAN_YEH) / len(output_yeh) if output_yeh else None
        ),
        "output_kaf_count": len(output_kaf),
        "persian_kaf_fraction": (
            output_kaf.count(PERSIAN_KAF) / len(output_kaf) if output_kaf else None
        ),
    }


def punctuation_diagnostics(ref: str, hyp: str) -> dict[str, dict[str, int]]:
    aligned = _aligned_units(ref, hyp, diagnostic_units)
    return {
        mark: {
            "ref": sum(r == mark for r, _ in aligned),
            "correct": sum(r == mark and h == mark for r, h in aligned),
            "missed": sum(r == mark and h != mark for r, h in aligned),
            "inserted": sum(r != mark and h == mark for r, h in aligned),
        }
        for mark in ("،", "؛", "؟", "«", "»", "(", ")")
    }


def _safe_mean(values) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return round(statistics.mean(vs), 4) if vs else None


def edit_statistics(ref: str, hyp: str) -> dict[str, int]:
    aligned = _aligned_units(ref, hyp, graphemes)
    substitutions = sum(1 for r, h in aligned if r is not None and h is not None and r != h)
    deletions = sum(1 for r, h in aligned if r is not None and h is None)
    insertions = sum(1 for r, h in aligned if r is None and h is not None)
    return {
        "ref_graphemes": sum(1 for r, _ in aligned if r is not None),
        "hyp_graphemes": sum(1 for _, h in aligned if h is not None),
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "edit_distance": insertions + deletions + substitutions,
    }


def percentile(values: list[float], probability: float) -> float:
    return values[round((len(values) - 1) * probability)]


def bootstrap_ci(
    values: list[float], *, iterations: int = 10_000, seed: int = 20260712
) -> Optional[list[float]]:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = [
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(iterations)
    ]
    samples.sort()
    return [
        round(percentile(samples, 0.025), 4),
        round(percentile(samples, 0.975), 4),
    ]


def summarize_records(records: list[ImageResult]) -> dict[str, object]:
    successful = [r for r in records if not r.error]
    values = [r.cer_grapheme_canonical for r in successful]
    edit_distance_total = sum(r.canonical_edit_distance or 0 for r in successful)
    ref_total = sum(r.canonical_ref_graphemes or 0 for r in successful)
    return {
        "n_runs": len(records),
        "n_ok": len(successful),
        "n_err": len(records) - len(successful),
        "macro_page_CER_canonical": _safe_mean(values),
        "median_page_CER_canonical": (
            round(statistics.median([v for v in values if v is not None]), 4)
            if values and any(v is not None for v in values)
            else None
        ),
        "mean_grapheme_CER_strict": _safe_mean(
            [r.cer_grapheme_strict for r in successful]
        ),
        "mean_WER_canonical": _safe_mean(
            [r.wer_canonical for r in successful]
        ),
        "micro_corpus_CER_canonical": (
            round(edit_distance_total / ref_total, 4) if ref_total else None
        ),
        "page_bootstrap_95ci": bootstrap_ci([v for v in values if v is not None]),
    }


def metadata_breakdowns(records: list[ImageResult]) -> dict[str, dict[str, object]]:
    breakdowns: dict[str, dict[str, object]] = {}
    for key in (
        "font",
        "font_size",
        "background",
        "capture",
        "layout",
        "degradations",
        "mixed_language",
        "handwriting_support",
    ):
        values: dict[str, list[ImageResult]] = {}
        for record in records:
            raw = record.page_metadata.get(key)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                if item is not None:
                    values.setdefault(str(item), []).append(record)
        if values:
            breakdowns[key] = {
                value: summarize_records(group) for value, group in sorted(values.items())
            }
    return breakdowns


# ---------- tesseract utilities ----------

def require_tesseract_binary() -> str:
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            sys.exit(f"ERROR: TESSERACT_CMD does not exist: {configured_path}")
        return str(configured_path.resolve())

    tcmd = shutil.which("tesseract")
    if not tcmd and os.name == "nt":
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
        tcmd = next((str(path) for path in candidates if path.is_file()), None)
    if not tcmd:
        sys.exit(
            "ERROR: 'tesseract' binary not found on PATH.\n"
            "  Windows: install from "
            "https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS:   brew install tesseract\n"
            "  Linux:   sudo apt install tesseract-ocr"
        )
    return tcmd


def configure_pytesseract(binary: str) -> None:
    """Point the Python wrapper at the exact executable found by preflight."""
    require_pytesseract()
    assert pytesseract is not None
    pytesseract.pytesseract.tesseract_cmd = binary


def require_pytesseract() -> None:
    if pytesseract is None or Image is None:
        sys.exit(
            "ERROR: Python dependencies are not installed.\n"
            "  Run: uv sync\n"
            "  Then: uv run python tesseract_fas.py --pull"
        )


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: failed to download {url}: {e}")
    if len(data) < 1024:
        sys.exit(
            f"ERROR: {url} returned suspiciously small payload "
            f"({len(data)} bytes); refusing to overwrite."
        )
    dest.write_bytes(data)
    print(f"  -> {dest} ({len(data) / 1024:.1f} KB)")


def tessdata_dir_for(variant: str) -> Path:
    return TESSDATA_ROOT / variant


def ensure_tessdata(lang_codes: list[str], variant: str) -> dict[str, Path]:
    """Make sure required .traineddata files exist locally; auto-pull if not."""
    tessdata_dir = tessdata_dir_for(variant)
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TESSDATA_PREFIX"] = str(tessdata_dir) + os.sep
    files: dict[str, Path] = {}
    for lang in lang_codes:
        for code in lang.split("+"):
            code = code.strip()
            if not code:
                continue
            destination = tessdata_dir / f"{code}.traineddata"
            if not destination.exists():
                download(f"{TESSDATA_URLS[variant]}/{destination.name}", destination)
            files[code] = destination
    return files


def verify_langs(binary: str, langs: list[str], tessdata_dir: Path) -> dict[str, bool]:
    try:
        out = subprocess.run(
            [binary, "--list-langs"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "TESSDATA_PREFIX": str(tessdata_dir) + os.sep},
        ).stdout
    except subprocess.CalledProcessError:
        return {lang: False for lang in langs}
    available = set(out.split())
    return {lang: lang in available for lang in langs}


def tesseract_version(binary: str, *, full: bool = False) -> str:
    try:
        output = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return output if full else output.splitlines()[0]
    except subprocess.CalledProcessError:
        return "(unknown)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_versions(names: list[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


# ---------- cmd: --pull ----------

def cmd_pull(args) -> int:
    binary = require_tesseract_binary()

    tessdata_dir = tessdata_dir_for(args.variant)
    base = TESSDATA_URLS[args.variant]
    targets = ["fas.traineddata"]
    if args.with_eng:
        targets.append("eng.traineddata")

    print(f"Project tessdata dir: {tessdata_dir}")
    print(f"Variant:              {args.variant}")
    print(f"Targets:              {', '.join(targets)}")
    print()

    for name in targets:
        dest = tessdata_dir / name
        url = f"{base}/{name}"
        if dest.exists() and not args.force:
            print(f"  already present: {dest}")
            continue
        download(url, dest)

    print()
    print(f"tesseract version: {tesseract_version(binary)}")
    print("Verifying languages:")
    wanted = ["fas"] + (["eng"] if args.with_eng else [])
    for lang, ok in verify_langs(binary, wanted, tessdata_dir).items():
        print(f"  {lang:>4}: {'OK' if ok else 'MISSING'}")

    print(
        "\nNext step:\n"
        f"  python {Path(__file__).name} --small_bench"
    )
    return 0


# ---------- cmd: --small_bench ----------

@dataclass
class ImageResult:
    subdir: str
    track: str
    image: str
    reference_source: str
    reference_quality: str
    page_metadata: dict[str, object]
    lang: str
    oem: int
    psm: int
    preprocess: str
    tesseract_config: str
    seconds: float
    image_sha256: str
    reference_sha256: str
    text_raw: str
    text_canonical: str
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
    tsv_diagnostics: Optional[dict[str, object]]
    failure_image_path: Optional[str]
    error: Optional[str] = None


def load_ground_truth(image_path: Path) -> tuple[str, str, str, dict[str, object]]:
    """Load reviewed JSON text when available, otherwise legacy Markdown.

    Sidecar schema: ``{"text": "...", "quality": "reviewed"}``.
    ``ignore`` regions are reserved for a future layout track and are not
    silently treated as recognition text.
    """
    sidecar = image_path.with_suffix(".reference.json")
    legacy_md = image_path.with_suffix(".md")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            text = payload["text"]
            quality = payload.get("quality", "reviewed")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"text", "quality"}
            }
            return text, str(sidecar), str(quality), metadata
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            sys.exit(f"ERROR: invalid reference sidecar {sidecar}: {exc}")
    if legacy_md.exists():
        return (
            strip_markdown(legacy_md.read_text(encoding="utf-8")),
            str(legacy_md),
            "unreviewed_markdown",
            {},
        )
    raise FileNotFoundError(f"No reference for {image_path}")


def track_for_subdir(subdir: str) -> str:
    return {
        "typed": "printed_smoke",
        "hand-written": "handwriting_smoke",
    }.get(subdir, "unclassified")


def load_manifest(path: Path) -> list[tuple[Path, dict[str, object]]]:
    entries: list[tuple[Path, dict[str, object]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            image = Path(payload.pop("image"))
            if not image.is_absolute():
                image = REPO_ROOT / image
            if not image.is_file():
                raise ValueError(f"image does not exist: {image}")
            entries.append((image, payload))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            sys.exit(f"ERROR: invalid manifest {path}:{line_number}: {exc}")
    if not entries:
        sys.exit(f"ERROR: manifest contains no images: {path}")
    return entries


def run_tesseract(
    image_path: Path,
    lang: str,
    oem: int,
    psm: int,
    timeout: float,
    preprocess_name: str,
    save_tsv: bool,
) -> tuple[str, Optional[dict[str, object]], float, Optional[str]]:
    if pytesseract is None or Image is None:
        return "", None, 0.0, "pytesseract not available"
    try:
        with Image.open(image_path) as source:
            img = preprocess_image(source, PROFILES[preprocess_name])
    except Exception as e:  # noqa: BLE001
        return "", None, 0.0, f"image/preprocessing failed: {e}"
    t0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(
            img,
            lang=lang,
            config=f"--oem {oem} --psm {psm}",
            timeout=timeout,
        )
        tsv_diagnostics = None
        if save_tsv:
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                config=f"--oem {oem} --psm {psm}",
                timeout=timeout,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                float(conf)
                for conf, token in zip(data["conf"], data["text"])
                if str(token).strip() and float(conf) >= 0
            ]
            tsv_diagnostics = {
                "mean_word_confidence": (
                    round(statistics.mean(confidences), 4) if confidences else None
                ),
                "recognized_words": len(confidences),
                "detected_blocks": len(
                    {
                        block
                        for block, token in zip(data["block_num"], data["text"])
                        if str(token).strip()
                    }
                ),
                "tsv": data,
                "extra_tesseract_pass": True,
            }
    except pytesseract.TesseractError as e:
        return "", None, time.perf_counter() - t0, f"tesseract error: {e}"
    except RuntimeError as e:
        return "", None, time.perf_counter() - t0, f"tesseract timeout: {e}"
    except Exception as e:  # noqa: BLE001
        return "", None, time.perf_counter() - t0, f"inference failed: {e}"
    return text, tsv_diagnostics, time.perf_counter() - t0, None


def cmd_small_bench(args) -> int:
    require_pytesseract()
    binary = require_tesseract_binary()
    configure_pytesseract(binary)

    if not SMALL_BENCH.exists():
        sys.exit(f"ERROR: small_bench directory not found at {SMALL_BENCH}")

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    try:
        psms = [int(s.strip()) for s in args.psm.split(",") if s.strip()]
    except ValueError:
        sys.exit("ERROR: --psm must be a comma-separated list of integers.")
    if not langs:
        sys.exit("ERROR: --langs must contain at least one language.")
    if not psms or any(psm < 0 or psm > 13 for psm in psms):
        sys.exit("ERROR: --psm values must be integers from 0 through 13.")
    if args.timeout <= 0:
        sys.exit("ERROR: --timeout must be greater than zero.")
    if args.limit is not None and args.limit <= 0:
        sys.exit("ERROR: --limit must be greater than zero.")
    if args.failure_cer_threshold < 0:
        sys.exit("ERROR: --failure-cer-threshold must be non-negative.")
    oem = args.oem
    if oem != 1:
        sys.exit("ERROR: tessdata_best and tessdata_fast require --oem 1.")
    results_path = Path(args.output)
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path

    # Collect all unique lang codes and make sure traineddata exists
    traineddata_files = ensure_tessdata(langs, args.variant)
    tessdata_dir = tessdata_dir_for(args.variant)
    language_codes = sorted(traineddata_files)
    availability = verify_langs(binary, language_codes, tessdata_dir)
    missing_languages = [code for code, available in availability.items() if not available]
    if missing_languages:
        sys.exit(f"ERROR: Tesseract could not load languages: {missing_languages}")

    print(f"tesseract:           {tesseract_version(binary)}")
    print(f"Executable:          {binary}")
    print(
        f"Configuration:       langs={langs}  oem={oem}  psms={psms}  "
        f"timeout={args.timeout}s"
    )
    print(f"Small bench:         {SMALL_BENCH}")
    print(f"Traineddata variant: {args.variant}")
    print(f"Preprocessing:       {args.preprocess}")
    print(f"Results output:      {results_path}")
    print()

    results: list[ImageResult] = []
    n_ok = 0
    n_err = 0

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if manifest_path.exists():
        image_paths = [
            (image.parent, image, metadata)
            for image, metadata in load_manifest(manifest_path)
        ]
    else:
        subdirs = sorted(d for d in SMALL_BENCH.iterdir() if d.is_dir())
        image_paths = [
            (sub, img_path, {})
            for sub in subdirs
            for img_path in sorted(
                (path for path in sub.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: (
                    0 if path.stem.isdigit() else 1,
                    int(path.stem) if path.stem.isdigit() else path.stem.casefold(),
                ),
            )
        ]
    if args.subdir:
        wanted = set(args.subdir)
        image_paths = [item for item in image_paths if item[0].name in wanted]
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        sys.exit("ERROR: no benchmark images matched the selected inputs.")
    if args.require_reviewed:
        invalid_references = []
        for _, img_path, _ in image_paths:
            sidecar = img_path.with_suffix(".reference.json")
            if not sidecar.exists():
                invalid_references.append(f"{sidecar} (missing)")
                continue
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if payload.get("quality") not in {"reviewed", "double_reviewed"}:
                    invalid_references.append(f"{sidecar} (quality is not reviewed)")
            except (OSError, json.JSONDecodeError, AttributeError):
                invalid_references.append(f"{sidecar} (invalid JSON)")
        if invalid_references:
            sys.exit(
                "ERROR: reviewed reference sidecars are required; missing "
                f"or invalid {len(invalid_references)} file(s), "
                f"first: {invalid_references[0]}"
            )

    for sub, img_path, manifest_metadata in image_paths:
        try:
            ref_raw, reference_source, reference_quality, page_metadata = load_ground_truth(img_path)
            page_metadata = {**manifest_metadata, **page_metadata}
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: {exc}")
        ref_strict = normalize_fa(ref_raw, "strict")
        ref_canonical = normalize_fa(ref_raw, "canonical")
        ref_search = normalize_fa(ref_raw, "search")
        for lang in langs:
            for psm in psms:
                hyp_raw, tsv_diagnostics, secs, err = run_tesseract(
                    img_path,
                    lang,
                    oem,
                    psm,
                    args.timeout,
                    args.preprocess,
                    args.save_tsv,
                )
                hyp_strict = normalize_fa(hyp_raw, "strict")
                hyp_canonical = normalize_fa(hyp_raw, "canonical")
                hyp_search = normalize_fa(hyp_raw, "search")
                diagnostics = (
                    {} if err else orthographic_diagnostics(ref_canonical, hyp_canonical)
                )
                unicode_diagnostics = (
                    {}
                    if err
                    else unicode_variant_diagnostics(
                        normalize_transport(ref_raw), normalize_transport(hyp_raw)
                    )
                )
                punctuation = (
                    {}
                    if err
                    else punctuation_diagnostics(ref_canonical, hyp_canonical)
                )
                edits = {} if err else edit_statistics(ref_canonical, hyp_canonical)
                config_string = f"--oem {oem} --psm {psm}"
                res = ImageResult(
                    subdir=sub.name,
                    track=str(page_metadata.get("track", track_for_subdir(sub.name))),
                    image=img_path.name,
                    reference_source=reference_source,
                    reference_quality=reference_quality,
                    page_metadata=page_metadata,
                    lang=lang,
                    oem=oem,
                    psm=psm,
                    preprocess=args.preprocess,
                    tesseract_config=config_string,
                    seconds=round(secs, 4),
                    image_sha256=sha256_file(img_path),
                    reference_sha256=sha256_file(Path(reference_source)),
                    text_raw=hyp_raw,
                    text_canonical=hyp_canonical,
                    cer_codepoint_strict=(
                        None if err else round(cer(ref_strict, hyp_strict, unit="codepoint"), 4)
                    ),
                    cer_grapheme_strict=(
                        None if err else round(cer(ref_strict, hyp_strict), 4)
                    ),
                    cer_grapheme_canonical=(
                        None if err else round(cer(ref_canonical, hyp_canonical), 4)
                    ),
                    cer_grapheme_search=(
                        None if err else round(cer(ref_search, hyp_search), 4)
                    ),
                    wer_canonical=(
                        None if err else round(wer(ref_canonical, hyp_canonical), 4)
                    ),
                    canonical_ref_graphemes=edits.get("ref_graphemes"),
                    canonical_hyp_graphemes=edits.get("hyp_graphemes"),
                    canonical_insertions=edits.get("insertions"),
                    canonical_deletions=edits.get("deletions"),
                    canonical_substitutions=edits.get("substitutions"),
                    canonical_edit_distance=edits.get("edit_distance"),
                    diagnostics=diagnostics,
                    unicode_form_diagnostics=unicode_diagnostics,
                    punctuation_diagnostics=punctuation,
                    tsv_diagnostics=tsv_diagnostics,
                    failure_image_path=None,
                    error=err,
                )
                if args.save_failure_images and (
                    err
                    or (
                        res.cer_grapheme_canonical is not None
                        and res.cer_grapheme_canonical >= args.failure_cer_threshold
                    )
                ):
                    failure_dir = results_path.parent / "failure_images"
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure_path = failure_dir / (
                        f"{sub.name}_{img_path.stem}_{lang.replace('+', '-')}_"
                        f"psm{psm}_{args.preprocess}.png"
                    )
                    with Image.open(img_path) as source:
                        preprocess_image(source, PROFILES[args.preprocess]).save(
                            failure_path
                        )
                    res.failure_image_path = str(failure_path)
                results.append(res)
                if err:
                    n_err += 1
                else:
                    n_ok += 1
                tag = "OK " if not err else "ERR"
                if err:
                    print(
                        f"  [{tag}] {sub.name}/{img_path.name}  "
                        f"lang={lang:<8} psm={psm}  t={secs:.2f}s\n"
                        f"        {err}"
                    )
                else:
                    print(
                        f"  [{tag}] {sub.name}/{img_path.name}  "
                        f"lang={lang:<8} psm={psm}  "
                        f"CER_canonical={res.cer_grapheme_canonical:.3f} "
                        f"t={secs:.2f}s"
                    )

    ok = [r for r in results if not r.error]
    primary_lang = langs[0]
    primary_psm = psms[0]
    primary = [
        r for r in results if r.lang == primary_lang and r.psm == primary_psm
    ]
    primary_ok = [r for r in primary if not r.error]
    primary_seconds = sorted(r.seconds for r in primary_ok)
    all_seconds = sorted(r.seconds for r in ok)
    config_breakdowns = {
        f"{lang}|psm={psm}": summarize_records(
            [r for r in results if r.lang == lang and r.psm == psm]
        )
        for lang in langs
        for psm in psms
    }
    track_breakdowns = {
        track: summarize_records([r for r in primary if r.track == track])
        for track in sorted({r.track for r in primary})
    }
    summary = {
        "benchmark": {
            "schema": LEADERBOARD_SCHEMA,
            "name": "persian_ocr_smoke20",
            "scope": "OCR benchmark; leaderboard-comparable",
            "reference_quality_counts": dict(
                Counter(
                    dict(
                        ((r.subdir, r.image), r.reference_quality)
                        for r in results
                    ).values()
                )
            ),
        },
        "config": {
            "langs": langs,
            "oem": oem,
            "psms": psms,
            "timeout_seconds": args.timeout,
            "limit": args.limit,
            "manifest": str(manifest_path) if manifest_path.exists() else None,
            "tesseract_executable": binary,
            "tesseract_version": tesseract_version(binary, full=True),
            "variant": args.variant,
            "tessdata_dir": str(tessdata_dir),
            "traineddata": {
                code: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "variant": args.variant,
                }
                for code, path in sorted(traineddata_files.items())
            },
            "primary_lang": primary_lang,
            "primary_psm": primary_psm,
            "preprocess": PROFILES[args.preprocess].to_dict(),
            "tesseract_config": f"--oem {oem} --psm {primary_psm}",
            "save_tsv": args.save_tsv,
            "require_reviewed": args.require_reviewed,
        },
        "n_images": len({(r.subdir, r.image) for r in results}),
        "n_skipped": 0,
        "n_runs": len(results),
        "n_ok": n_ok,
        "n_err": n_err,
        "failure_rate": round(n_err / len(results), 4) if results else None,
        "primary_results": summarize_records(primary),
        "config_breakdowns": config_breakdowns,
        "track_breakdowns_primary_config": track_breakdowns,
        "metadata_breakdowns_primary_config": metadata_breakdowns(primary),
        "metrics_policy": {
            "transport": "line-ending/form-feed cleanup, NFC and outer trim",
            "strict": "transport cleanup only; grapheme and codepoint CER reported",
            "canonical": "NFC plus explicit Persian yeh/kaf and bidi-control removal",
            "search": "canonical plus digit canonicalization, tatweel removal and whitespace collapse",
            "line_accuracy": "removed; visual line segmentation is not annotated",
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "packages": installed_versions(
                ["Pillow", "pytesseract", "rapidfuzz", "regex", "numpy", "opencv-python-headless"]
            ),
        },
        "operations": {
            "all_configurations": {
                "mean_seconds_per_run": round(statistics.mean(all_seconds), 4) if all_seconds else None,
                "median_seconds_per_run": round(statistics.median(all_seconds), 4) if all_seconds else None,
                "p95_seconds_per_run": round(percentile(all_seconds, 0.95), 4) if all_seconds else None,
            },
            "primary_configuration": {
                "mean_seconds_per_run": round(statistics.mean(primary_seconds), 4) if primary_seconds else None,
                "median_seconds_per_run": round(statistics.median(primary_seconds), 4) if primary_seconds else None,
                "p95_seconds_per_run": round(percentile(primary_seconds, 0.95), 4) if primary_seconds else None,
            },
            "peak_VRAM_GB": None,
            "peak_RAM_GB": None,
            "cold_start_seconds": None,
        },
    }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for section, payload in summary.items():
        if isinstance(payload, dict):
            print(f"[{section}]")
            for k, v in payload.items():
                print(f"  {k:>30}: {v}")
        else:
            print(f"  {section:>30}: {payload}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(r) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {results_path}")

    if args.show_failures and ok:
        worst = sorted(
            [r for r in primary if r.cer_grapheme_canonical is not None],
            key=lambda r: r.cer_grapheme_canonical,
            reverse=True,
        )[:5]
        print("\nWorst primary-configuration runs:")
        for r in worst:
            print(
                f"  {r.subdir}/{r.image}  lang={r.lang} psm={r.psm}  "
                f"CER_canonical={r.cer_grapheme_canonical:.3f}  "
                f"WER={r.wer_canonical:.3f}"
            )
    if results and not ok:
        print("\nERROR: every OCR run failed; inspect the errors above.", file=sys.stderr)
        return 2
    return 0


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pull Tesseract Persian (fas.traineddata) and benchmark on "
            "small_bench/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pull", action="store_true",
        help="Download fas.traineddata (+eng if --with-eng) and verify.",
    )
    mode.add_argument(
        "--small_bench", action="store_true",
        help="Benchmark on small_bench/ typed + hand-written.",
    )

    p.add_argument(
        "--with-eng", action="store_true",
        help="Also fetch eng.traineddata (enables fas+eng runs).",
    )
    p.add_argument(
        "--variant", choices=["best", "fast"], default="best",
        help="Which tessdata repository to pull from.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download traineddata even if already present.",
    )

    p.add_argument(
        "--langs", default="fas",
        help="Comma-separated lang strings; the first is the primary configuration.",
    )
    p.add_argument(
        "--oem", type=int, default=1,
        help="OCR Engine Mode (1 = LSTM only, recommended).",
    )
    p.add_argument(
        "--psm", default="6",
        help="Comma-separated Page Segmentation Modes; the first is primary.",
    )
    p.add_argument(
        "--subdir", nargs="*", default=None,
        help="Restrict benchmark to specific small_bench subdirectories.",
    )
    p.add_argument(
        "--manifest", default="small_bench/manifest.jsonl",
        help="Authoritative JSONL image list; falls back to folder discovery if absent.",
    )
    p.add_argument(
        "--show-failures", action="store_true",
        help="Print worst primary-configuration runs after benchmarking.",
    )
    p.add_argument(
        "--preprocess", choices=sorted(PROFILES), default="raw",
        help="Named preprocessing profile; each profile is a separate configuration.",
    )
    p.add_argument(
        "--save-tsv", action="store_true",
        help="Store word confidence/geometry TSV diagnostics (runs Tesseract twice).",
    )
    p.add_argument(
        "--save-failure-images", action="store_true",
        help="Save the exact preprocessed input for OCR errors/high-CER runs.",
    )
    p.add_argument(
        "--failure-cer-threshold", type=float, default=0.5,
        help="Canonical CER threshold used by --save-failure-images.",
    )
    p.add_argument(
        "--output", default=str(RESULTS_PATH.relative_to(REPO_ROOT)),
        help="JSON output path, relative to the repository root by default.",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="Maximum seconds allowed for each OCR run.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Benchmark only the first N images (useful for smoke runs).",
    )
    p.add_argument(
        "--require-reviewed", action="store_true",
        help="Fail unless every selected image has a .reference.json sidecar.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pull:
        return cmd_pull(args)
    if args.small_bench:
        return cmd_small_bench(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
