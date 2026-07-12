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
import json
import os
import platform
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

try:
    import pytesseract
    from PIL import Image
except ImportError:  # validated by require_pytesseract before use
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


def normalize_fa(text: str, policy: str = "canonical") -> str:
    """Apply an explicit scoring policy; preserve ZWNJ in all policies."""
    text = unicodedata.normalize("NFC", text or "")
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


def _aligned_units(ref: str, hyp: str) -> list[tuple[Optional[str], Optional[str]]]:
    a, b = graphemes(ref), graphemes(hyp)
    rows, cols = len(a) + 1, len(b) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    aligned: list[tuple[Optional[str], Optional[str]]] = []
    i, j = len(a), len(b)
    while i or j:
        if i and j and distance[i][j] == distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            aligned.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            aligned.append((a[i - 1], None))
            i -= 1
        else:
            aligned.append((None, b[j - 1]))
            j -= 1
    return list(reversed(aligned))


def orthographic_diagnostics(ref: str, hyp: str) -> dict[str, Optional[float] | int]:
    """Reference-grounded Yeh/Kaf substitution, deletion and insertion rates."""
    aligned = _aligned_units(ref, hyp)
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


def _safe_mean(values) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return round(statistics.mean(vs), 4) if vs else None


def summarize_records(records: list[ImageResult]) -> dict[str, Optional[float] | int]:
    successful = [r for r in records if not r.error]
    values = [r.cer_grapheme_canonical for r in successful]
    return {
        "n_runs": len(records),
        "n_ok": len(successful),
        "n_err": len(records) - len(successful),
        "mean_grapheme_CER_canonical": _safe_mean(values),
        "median_grapheme_CER_canonical": (
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
    }


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
    track: str
    image: str
    reference_source: str
    reference_quality: str
    lang: str
    oem: int
    psm: int
    seconds: float
    text_raw: str
    text_canonical: str
    cer_codepoint_strict: Optional[float]
    cer_grapheme_strict: Optional[float]
    cer_grapheme_canonical: Optional[float]
    cer_grapheme_search: Optional[float]
    wer_canonical: Optional[float]
    diagnostics: dict[str, Optional[float] | int]
    error: Optional[str] = None


def load_ground_truth(md_path: Path) -> tuple[str, str, str]:
    """Load reviewed JSON text when available, otherwise legacy Markdown.

    Sidecar schema: ``{"text": "...", "quality": "reviewed"}``.
    ``ignore`` regions are reserved for a future layout track and are not
    silently treated as recognition text.
    """
    sidecar = md_path.with_suffix(".reference.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            text = payload["text"]
            quality = payload.get("quality", "reviewed")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            return text, str(sidecar), str(quality)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            sys.exit(f"ERROR: invalid reference sidecar {sidecar}: {exc}")
    return (
        strip_markdown(md_path.read_text(encoding="utf-8")),
        str(md_path),
        "unreviewed_markdown",
    )


def track_for_subdir(subdir: str) -> str:
    return {
        "typed": "printed_smoke",
        "hand-written": "handwriting_smoke",
    }.get(subdir, "unclassified")


def run_tesseract(
    image_path: Path, lang: str, oem: int, psm: int, timeout: float
) -> tuple[str, float, Optional[str]]:
    if pytesseract is None or Image is None:
        return "", 0.0, "pytesseract not available"
    try:
        with Image.open(image_path) as source:
            img = source.convert("RGB")
    except Exception as e:  # noqa: BLE001
        return "", 0.0, f"image open failed: {e}"
    t0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(
            img,
            lang=lang,
            config=f"--oem {oem} --psm {psm}",
            timeout=timeout,
        )
    except pytesseract.TesseractError as e:
        return "", time.perf_counter() - t0, f"tesseract error: {e}"
    except RuntimeError as e:
        return "", time.perf_counter() - t0, f"tesseract timeout: {e}"
    except Exception as e:  # noqa: BLE001
        return "", time.perf_counter() - t0, f"inference failed: {e}"
    return text, time.perf_counter() - t0, None


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
    oem = args.oem

    # Collect all unique lang codes and make sure traineddata exists
    ensure_tessdata(langs)

    print(f"tesseract:           {tesseract_version(binary)}")
    print(f"Executable:          {binary}")
    print(
        f"Configuration:       langs={langs}  oem={oem}  psms={psms}  "
        f"timeout={args.timeout}s"
    )
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

    image_paths = [
        (sub, img_path)
        for sub in subdirs
        for img_path in sorted(
            sub.glob("*.jpg"),
            key=lambda path: (
                0 if path.stem.isdigit() else 1,
                int(path.stem) if path.stem.isdigit() else path.stem.casefold(),
            ),
        )
    ]
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        sys.exit("ERROR: no JPG benchmark images matched the selected inputs.")

    for sub, img_path in image_paths:
        md_path = img_path.with_suffix(".md")
        if not md_path.exists():
            print(f"[skip] {sub.name}/{img_path.name}: no .md ground truth")
            continue
        ref_raw, reference_source, reference_quality = load_ground_truth(md_path)
        ref_strict = normalize_fa(ref_raw, "strict")
        ref_canonical = normalize_fa(ref_raw, "canonical")
        ref_search = normalize_fa(ref_raw, "search")
        for lang in langs:
            for psm in psms:
                hyp_raw, secs, err = run_tesseract(
                    img_path, lang, oem, psm, args.timeout
                )
                hyp_strict = normalize_fa(hyp_raw, "strict")
                hyp_canonical = normalize_fa(hyp_raw, "canonical")
                hyp_search = normalize_fa(hyp_raw, "search")
                diagnostics = (
                    {} if err else orthographic_diagnostics(ref_canonical, hyp_canonical)
                )
                res = ImageResult(
                    subdir=sub.name,
                    track=track_for_subdir(sub.name),
                    image=img_path.name,
                    reference_source=reference_source,
                    reference_quality=reference_quality,
                    lang=lang,
                    oem=oem,
                    psm=psm,
                    seconds=round(secs, 4),
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
                    diagnostics=diagnostics,
                    error=err,
                )
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
    mean_secs = statistics.mean([r.seconds for r in ok]) if ok else 0.0
    primary_lang = langs[0]
    primary_psm = psms[0]
    primary = [
        r for r in results if r.lang == primary_lang and r.psm == primary_psm
    ]
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
            "name": "persian_ocr_smoke20",
            "scope": "smoke and failure diagnosis; not a model leaderboard",
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
            "tesseract_executable": binary,
            "tesseract_version": tesseract_version(binary),
            "tessdata_dir": str(TESSDATA_DIR),
            "primary_lang": primary_lang,
            "primary_psm": primary_psm,
        },
        "n_images": len({(r.subdir, r.image) for r in results}),
        "n_runs": len(results),
        "n_ok": n_ok,
        "n_err": n_err,
        "failure_rate": round(n_err / len(results), 4) if results else None,
        "primary_results": summarize_records(primary),
        "config_breakdowns": config_breakdowns,
        "track_breakdowns_primary_config": track_breakdowns,
        "metrics_policy": {
            "strict": "NFC only; grapheme CER and codepoint CER are both reported",
            "canonical": "NFC plus explicit Persian yeh/kaf and bidi-control removal",
            "search": "canonical plus digit canonicalization, tatweel removal and whitespace collapse",
            "line_accuracy": "removed; visual line segmentation is not annotated",
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "operations": {
            "mean_seconds_per_page": round(mean_secs, 4) if ok else None,
            "pages_per_second": round(1.0 / mean_secs, 4) if mean_secs else None,
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
        "--show-failures", action="store_true",
        help="Print worst primary-configuration runs after benchmarking.",
    )
    p.add_argument(
        "--timeout", type=float, default=60.0,
        help="Maximum seconds allowed for each OCR run.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Benchmark only the first N images (useful for smoke runs).",
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
