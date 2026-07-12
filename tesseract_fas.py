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
import json
import os
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

try:
    import pytesseract
    from PIL import Image
except ImportError:  # auto-installed by --pull if missing
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


# ---------- paths ----------

REPO_ROOT = Path(__file__).resolve().parent
TESSDATA_DIR = REPO_ROOT / "tessdata"
SMALL_BENCH = REPO_ROOT / "small_bench"
RESULTS_DIR = REPO_ROOT / "bench_runs"
RESULTS_PATH = RESULTS_DIR / "tesseract_fas.json"

TESSDATA_BASE = "https://github.com/tesseract-ocr/tessdata_best/raw/main"
TESSDATA_FAST_BASE = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"


# ---------- Persian / Arabic constants ----------

PERSIAN_YEH = "ی"   # U+06CC
ARABIC_YEH = "ي"    # U+064A
PERSIAN_KAF = "ک"   # U+06A9
ARABIC_KAF = "ك"    # U+0643
ZWNJ = "‌"           # U+200C (zero-width non-joiner)
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# ---------- text utilities ----------

_MD_HEADING = re.compile(r"^#+\s*", flags=re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_MD_RULE = re.compile(r"^[-*_]{3,}\s*$", flags=re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove markdown markers but preserve Persian text and ZWNJ."""
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC_STAR.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    return text.strip()


def normalize_fa(text: str) -> str:
    """Persian normalization per the catalog benchmark_protocol."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace(ARABIC_YEH, PERSIAN_YEH)
    text = text.replace(ARABIC_KAF, PERSIAN_KAF)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_digits_optional(text: str) -> str:
    """Optional Arabic-digit -> Persian-digit normalization."""
    return "".join(
        PERSIAN_DIGITS[ARABIC_DIGITS.index(ch)] if ch in ARABIC_DIGITS else ch
        for ch in (text or "")
    )


def edit_distance(a, b) -> int:
    """Levenshtein distance over any sequence (chars or words)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)


def wer(ref: str, hyp: str) -> float:
    rw = (ref or "").split()
    hw = (hyp or "").split()
    if not rw:
        return 0.0 if not hw else 1.0
    return edit_distance(rw, hw) / len(rw)


def yeh_accuracy_raw(hyp: str) -> Optional[float]:
    """Fraction of yeh-like chars in raw hypothesis that are Persian."""
    yeh_like = [ch for ch in hyp if ch in (PERSIAN_YEH, ARABIC_YEH)]
    if not yeh_like:
        return None
    return sum(1 for ch in yeh_like if ch == PERSIAN_YEH) / len(yeh_like)


def kaf_accuracy_raw(hyp: str) -> Optional[float]:
    """Fraction of kaf-like chars in raw hypothesis that are Persian."""
    kaf_like = [ch for ch in hyp if ch in (PERSIAN_KAF, ARABIC_KAF)]
    if not kaf_like:
        return None
    return sum(1 for ch in kaf_like if ch == PERSIAN_KAF) / len(kaf_like)


def zwnj_accuracy(ref: str, hyp: str) -> Optional[float]:
    """Fraction of ZWNJs in the reference preserved in the hypothesis."""
    rz = ref.count(ZWNJ)
    if rz == 0:
        return None
    return min(hyp.count(ZWNJ), rz) / rz


def _safe_mean(values) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return round(statistics.mean(vs), 4) if vs else None


# ---------- tesseract utilities ----------

def require_tesseract_binary() -> str:
    tcmd = shutil.which("tesseract")
    if not tcmd and os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", "")) / "Tesseract-OCR" / "tesseract.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Tesseract-OCR" / "tesseract.exe",
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


def require_pytesseract() -> None:
    global pytesseract, Image
    if pytesseract is None or Image is None:
        print("Installing pytesseract + Pillow ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pytesseract", "Pillow"],
            stdout=subprocess.DEVNULL,
        )
        import pytesseract as _p
        from PIL import Image as _I
        pytesseract = _p
        Image = _I


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


def ensure_tessdata(lang_codes: list[str]) -> None:
    """Make sure required .traineddata files exist locally; auto-pull if not."""
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    # pytesseract respects TESSDATA_PREFIX (note trailing separator)
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR) + os.sep
    needed: set[str] = set()
    for lang in lang_codes:
        for code in lang.split("+"):
            code = code.strip()
            if code and not (TESSDATA_DIR / f"{code}.traineddata").exists():
                needed.add(f"{code}.traineddata")
    for name in sorted(needed):
        download(f"{TESSDATA_BASE}/{name}", TESSDATA_DIR / name)


def verify_langs(binary: str, langs: list[str]) -> dict[str, bool]:
    try:
        out = subprocess.run(
            [binary, "--list-langs"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA_DIR) + os.sep},
        ).stdout
    except subprocess.CalledProcessError:
        return {lang: False for lang in langs}
    available = set(out.split())
    return {lang: lang in available for lang in langs}


def tesseract_version(binary: str) -> str:
    try:
        return subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]
    except subprocess.CalledProcessError:
        return "(unknown)"


# ---------- cmd: --pull ----------

def cmd_pull(args) -> int:
    binary = require_tesseract_binary()
    require_pytesseract()

    base = TESSDATA_FAST_BASE if args.variant == "fast" else TESSDATA_BASE
    targets = ["fas.traineddata"]
    if args.with_eng:
        targets.append("eng.traineddata")

    print(f"Project tessdata dir: {TESSDATA_DIR}")
    print(f"Variant:              {args.variant}")
    print(f"Targets:              {', '.join(targets)}")
    print()

    for name in targets:
        dest = TESSDATA_DIR / name
        url = f"{base}/{name}"
        if dest.exists() and not args.force:
            print(f"  already present: {dest}")
            continue
        download(url, dest)

    print()
    print(f"tesseract version: {tesseract_version(binary)}")
    print("Verifying languages:")
    wanted = ["fas"] + (["eng"] if args.with_eng else [])
    for lang, ok in verify_langs(binary, wanted).items():
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
    image: str
    lang: str
    oem: int
    psm: int
    seconds: float
    cer_raw: float
    wer_raw: float
    cer_norm: float
    wer_norm: float
    exact_line_accuracy: Optional[float]
    yeh_acc: Optional[float]
    kaf_acc: Optional[float]
    zwnj_acc: Optional[float]
    error: Optional[str] = None


def load_ground_truth(md_path: Path) -> str:
    return strip_markdown(md_path.read_text(encoding="utf-8"))


def run_tesseract(
    image_path: Path, lang: str, oem: int, psm: int
) -> tuple[str, float, Optional[str]]:
    if pytesseract is None or Image is None:
        return "", 0.0, "pytesseract not available"
    try:
        img = Image.open(image_path)
    except Exception as e:  # noqa: BLE001
        return "", 0.0, f"image open failed: {e}"
    t0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(
            img, lang=lang, config=f"--oem {oem} --psm {psm}"
        )
    except pytesseract.TesseractError as e:
        return "", time.perf_counter() - t0, f"tesseract error: {e}"
    except Exception as e:  # noqa: BLE001
        return "", time.perf_counter() - t0, f"inference failed: {e}"
    return text, time.perf_counter() - t0, None


def cmd_small_bench(args) -> int:
    require_pytesseract()
    binary = require_tesseract_binary()

    if not SMALL_BENCH.exists():
        sys.exit(f"ERROR: small_bench directory not found at {SMALL_BENCH}")

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    psms = [int(s) for s in args.psm.split(",")]
    oem = args.oem

    # Collect all unique lang codes and make sure traineddata exists
    ensure_tessdata(langs)

    print(f"tesseract:           {tesseract_version(binary)}")
    print(f"Configuration:       langs={langs}  oem={oem}  psms={psms}")
    print(f"Small bench:         {SMALL_BENCH}")
    print(f"Results output:      {RESULTS_PATH}")
    print()

    subdirs = sorted(d for d in SMALL_BENCH.iterdir() if d.is_dir())
    if args.subdir:
        wanted = set(args.subdir)
        subdirs = [d for d in subdirs if d.name in wanted]
        if not subdirs:
            sys.exit(f"ERROR: no matching subdirs in {SMALL_BENCH}")

    results: list[ImageResult] = []
    n_ok = 0
    n_err = 0

    for sub in subdirs:
        for img_path in sorted(sub.glob("*.jpg")):
            md_path = img_path.with_suffix(".md")
            if not md_path.exists():
                print(f"[skip] {sub.name}/{img_path.name}: no .md ground truth")
                continue
            ref_raw = load_ground_truth(md_path)
            ref_norm = normalize_fa(ref_raw)
            ref_lines = [ln for ln in ref_raw.splitlines() if ln.strip()]
            for lang in langs:
                for psm in psms:
                    hyp_raw, secs, err = run_tesseract(img_path, lang, oem, psm)
                    hyp_norm = normalize_fa(hyp_raw)
                    hyp_lines = [ln for ln in hyp_raw.splitlines() if ln.strip()]
                    ela = (
                        sum(1 for r, h in zip(ref_lines, hyp_lines) if r == h)
                        / len(ref_lines)
                        if ref_lines
                        else None
                    )
                    res = ImageResult(
                        subdir=sub.name,
                        image=img_path.name,
                        lang=lang,
                        oem=oem,
                        psm=psm,
                        seconds=round(secs, 4),
                        cer_raw=round(cer(ref_raw, hyp_raw), 4),
                        wer_raw=round(wer(ref_raw, hyp_raw), 4),
                        cer_norm=round(cer(ref_norm, hyp_norm), 4),
                        wer_norm=round(wer(ref_norm, hyp_norm), 4),
                        exact_line_accuracy=round(ela, 4) if ela is not None else None,
                        yeh_acc=None if err else yeh_accuracy_raw(hyp_raw),
                        kaf_acc=None if err else kaf_accuracy_raw(hyp_raw),
                        zwnj_acc=None if err else zwnj_accuracy(ref_raw, hyp_raw),
                        error=err,
                    )
                    results.append(res)
                    if err:
                        n_err += 1
                    else:
                        n_ok += 1
                    tag = "OK " if not err else "ERR"
                    print(
                        f"  [{tag}] {sub.name}/{img_path.name}  "
                        f"lang={lang:<8} psm={psm}  "
                        f"CER_raw={res.cer_raw:.3f} CER_norm={res.cer_norm:.3f}  "
                        f"t={secs:.2f}s"
                    )

    ok = [r for r in results if not r.error]
    mean_secs = statistics.mean([r.seconds for r in ok]) if ok else 0.0
    summary = {
        "config": {"langs": langs, "oem": oem, "psms": psms},
        "n_images": len({(r.subdir, r.image) for r in results}),
        "n_runs": len(results),
        "n_ok": n_ok,
        "n_err": n_err,
        "failure_rate": round(n_err / len(results), 4) if results else None,
        "raw_recognition": {
            "mean_CER_raw": _safe_mean([r.cer_raw for r in ok]),
            "mean_WER_raw": _safe_mean([r.wer_raw for r in ok]),
            "mean_exact_line_accuracy": _safe_mean(
                [r.exact_line_accuracy for r in ok]
            ),
        },
        "normalized_recognition": {
            "mean_CER_norm": _safe_mean([r.cer_norm for r in ok]),
            "mean_WER_norm": _safe_mean([r.wer_norm for r in ok]),
        },
        "persian_failure_slices": {
            "yeh_accuracy_macro": _safe_mean([r.yeh_acc for r in ok]),
            "kaf_accuracy_macro": _safe_mean([r.kaf_acc for r in ok]),
            "zwnj_accuracy_macro": _safe_mean([r.zwnj_acc for r in ok]),
        },
        "operations": {
            "mean_seconds_per_page": round(mean_secs, 4) if ok else None,
            "pages_per_second": round(1.0 / mean_secs, 4) if mean_secs else None,
            "peak_VRAM_GB": 0.0,        # CPU only
            "cold_start_seconds": None,  # not measured per page
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

    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(r) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {RESULTS_PATH}")

    if args.show_failures and ok:
        worst = sorted(ok, key=lambda r: r.cer_norm, reverse=True)[:5]
        print("\nWorst CER_norm runs:")
        for r in worst:
            print(
                f"  {r.subdir}/{r.image}  lang={r.lang} psm={r.psm}  "
                f"CER_norm={r.cer_norm:.3f}  WER_norm={r.wer_norm:.3f}"
            )
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
        "--langs", default="fas,fas+eng",
        help="Comma-separated lang strings to test.",
    )
    p.add_argument(
        "--oem", type=int, default=1,
        help="OCR Engine Mode (1 = LSTM only, recommended).",
    )
    p.add_argument(
        "--psm", default="3,4,6",
        help="Comma-separated Page Segmentation Modes.",
    )
    p.add_argument(
        "--subdir", nargs="*", default=None,
        help="Restrict benchmark to specific small_bench subdirectories.",
    )
    p.add_argument(
        "--show-failures", action="store_true",
        help="Print the worst-scoring runs after benchmarking.",
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
