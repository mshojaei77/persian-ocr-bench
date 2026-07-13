"""Recover and validate the versioned smoke20-v1 reference corpus.

The source Markdown and original JPEGs are read from ``small_bench.zip``.
``--check`` never writes. ``--write`` prepares and validates every output
before atomically replacing the two JSON corpora and the JSONL manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
import zipfile


DATASET_ID = "smoke20-v1"
DATASET_VERSION = "1.0.0"
MANIFEST_SCHEMA = "persian_ocr_smoke_manifest_v1"
EXPECTED_PER_SPLIT = 10
SPLITS = ("hand-written", "typed")
REVIEW_STATUS = "ai_assisted_recovered_not_human_reviewed"
ANNOTATION_METHOD = "archived_markdown_recovery_with_ai_assisted_visual_review"
PROVENANCE_STATUS = "source_and_license_unknown"

# These bracketed spans describe graphics or locations rather than visible text.
# The list is deliberately exact. A generic ``[anything]`` rule previously
# deleted whole lines that also contained scorable text.
IMAGE_ONLY_DESCRIPTIONS = {
    "[آیکون گرافیکی]",
    "[تصویر دو گل در گوشه‌های بالایی]",
    "[تصویر یک کتاب باز]",
    "[درون یک دایره در پایین سمت چپ:]",
    "[لوگوی شرکت]",
    "[لوگوی قوه قضائیه - مرکز آمار و فناوری اطلاعات]",
    "[لوگوی قوه قضائیه]",
    "[لوگوی فراشناسا]",
    "[لوگو: تصویری از یک بنای سنتی در دامنه کوه]",
    "[محل تصویر شما]",
}
INLINE_IMAGE_ONLY_DESCRIPTIONS = {"[تصویر یک گل]"}
TABLE_SEPARATOR = re.compile(
    r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$"
)
ONLY_PRESENTATION = re.compile(r"^[-=*#~` ]+$")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
FULL_BRACKET = re.compile(r"^\[([^\]]+)\]$")
BRACKET_SPAN = re.compile(r"\[([^\]]*)\]")
WATERMARK_DESCRIPTION = re.compile(r"^واترمارک[^:]*:\s*(.+)$")


CONDITIONS: dict[str, list[str]] = {
    "hand-written/1.jpg": ["handwritten", "low_contrast", "mixed_script"],
    "hand-written/2.jpg": ["handwritten", "low_resolution"],
    "hand-written/3.jpg": ["handwritten", "lined_paper", "watermark"],
    "hand-written/4.jpg": ["handwritten", "rotated"],
    "hand-written/5.jpg": ["printed", "social_media_graphic", "decorative"],
    "hand-written/6.jpg": ["handwritten", "lined_paper"],
    "hand-written/7.jpg": [
        "handwritten",
        "low_resolution",
        "printed_form_header",
        "watermark",
        "dense_text",
    ],
    "hand-written/8.jpg": ["handwritten", "clean_scan", "dense_text"],
    "hand-written/9.jpg": ["handwritten", "low_resolution", "decorative", "watermark"],
    "hand-written/10.jpg": ["handwritten", "low_resolution", "watermark", "dense_text"],
    "typed/1.jpg": ["printed", "clean_scan", "book_page"],
    "typed/2.jpg": ["printed", "low_resolution", "uneven_illumination", "book_page"],
    "typed/3.jpg": ["printed", "clean_scan", "book_page"],
    "typed/4.jpg": ["printed", "low_light", "cropped", "book_page"],
    "typed/5.jpg": ["printed", "grayscale_photo", "perspective_distortion", "book_page"],
    "typed/6.jpg": ["printed", "table", "form"],
    "typed/7.jpg": ["printed", "low_resolution", "form"],
    "typed/8.jpg": ["printed", "social_media_graphic", "decorative"],
    "typed/9.jpg": ["printed", "legal_document", "source_redactions", "watermark", "dense_text"],
    "typed/10.jpg": ["printed", "low_resolution", "form"],
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative_dataset_path(path: Path, root: Path) -> str:
    """Return a repository-relative path or reject external dataset inputs."""
    base = root.resolve().parent
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset input must be inside {base}: {path}") from exc


def repair_mojibake(text: str) -> str:
    repaired = text
    markers = ("Ã", "Â", "Ø", "Ù", "â")
    if any(marker in text for marker in markers):
        for source_encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(source_encoding).decode("utf-8")
            except UnicodeError:
                continue
            if sum(text.count(marker) for marker in markers) > sum(
                candidate.count(marker) for marker in markers
            ):
                repaired = candidate
                break
    return repaired


def strip_markdown_emphasis(line: str) -> str:
    """Remove presentation markers without deleting visible underscores."""
    line = HEADING.sub("", line)
    if len(line) >= 4 and line.startswith("**") and line.endswith("**"):
        line = line[2:-2]
    line = line.replace("**", "")
    if len(line) >= 2 and line.startswith("`") and line.endswith("`"):
        line = line[1:-1]
    return line


def recover_bracketed_text(line: str) -> str:
    for description in INLINE_IMAGE_ONLY_DESCRIPTIONS:
        line = line.replace(description, "")

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if not content or content == "...":
            return ""
        watermark = WATERMARK_DESCRIPTION.match(content)
        if watermark:
            return watermark.group(1)
        return content

    return BRACKET_SPAN.sub(replace, line)


def normalize_table_row(line: str) -> str:
    if not (line.startswith("|") and line.endswith("|")):
        return line
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return "\t".join(cell for cell in cells if cell)


def repair_text(text: str) -> str:
    """Recover visible text while removing only known presentation metadata."""
    repaired = unicodedata.normalize("NFC", repair_mojibake(text))
    lines: list[str] = []
    for raw_line in repaired.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.lower() == "<br>" or ONLY_PRESENTATION.fullmatch(line):
            continue
        if line in IMAGE_ONLY_DESCRIPTIONS or TABLE_SEPARATOR.fullmatch(line):
            continue
        line = strip_markdown_emphasis(line)
        full_bracket = FULL_BRACKET.fullmatch(line)
        if full_bracket and line in IMAGE_ONLY_DESCRIPTIONS:
            continue
        line = recover_bracketed_text(line)
        line = normalize_table_row(line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def expected_members() -> set[str]:
    return {
        f"small_bench/{split}/{number}.{suffix}"
        for split in SPLITS
        for number in range(1, EXPECTED_PER_SPLIT + 1)
        for suffix in ("jpg", "md")
    }


def numeric_key(path: str) -> int:
    return int(Path(path).stem)


def dataset_digest(rows: list[dict[str, object]]) -> str:
    lines = [f"{DATASET_ID}\n{DATASET_VERSION}"]
    lines.extend(
        f"{row['sample_id']}|{row['image_sha256']}|{row['reference_sha256']}"
        for row in rows
    )
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def generate(archive: Path, root: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, object]]]:
    if not archive.is_file():
        raise ValueError(f"source archive does not exist: {archive}")
    archive_sha256 = sha256_file(archive)
    root_relative = relative_dataset_path(root, root)
    archive_relative = relative_dataset_path(archive, root)
    expected = expected_members()
    corpora: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    manifest: list[dict[str, object]] = []

    with zipfile.ZipFile(archive) as source:
        present = {
            name
            for name in source.namelist()
            if name.startswith("small_bench/") and name.lower().endswith((".jpg", ".md"))
        }
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            raise ValueError(f"archive member mismatch: missing={missing}, extra={extra}")

        live_images = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".jpg"
        }
        expected_images = {
            f"{split}/{number}.jpg"
            for split in SPLITS
            for number in range(1, EXPECTED_PER_SPLIT + 1)
        }
        if live_images != expected_images:
            raise ValueError(
                "live image mismatch: "
                f"missing={sorted(expected_images - live_images)}, "
                f"extra={sorted(live_images - expected_images)}"
            )

        for split in SPLITS:
            members = sorted(
                (name for name in expected if name.startswith(f"small_bench/{split}/") and name.endswith(".md")),
                key=numeric_key,
            )
            for source_member in members:
                number = int(Path(source_member).stem)
                key = f"{split}/{number}.jpg"
                image_path = root / key
                image_member = f"small_bench/{key}"
                live_image = image_path.read_bytes()
                archived_image = source.read(image_member)
                if live_image != archived_image:
                    raise ValueError(f"live image differs from source archive: {key}")

                source_markdown = source.read(source_member)
                try:
                    decoded = source_markdown.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"source reference is not UTF-8: {source_member}") from exc
                text = repair_text(decoded)
                if not text:
                    raise ValueError(f"empty recovered reference: {source_member}")
                if "<br>" in text.lower() or ":---" in text:
                    raise ValueError(f"presentation markup survived recovery: {source_member}")

                corpora[split].append({"image": key, "text": text})
                content_type = "printed" if key == "hand-written/5.jpg" or split == "typed" else "handwritten"
                track = "handwriting_smoke" if content_type == "handwritten" else (
                    "social_media_graphics" if key == "hand-written/5.jpg" else "printed_smoke"
                )
                manifest.append(
                    {
                        "schema": MANIFEST_SCHEMA,
                        "dataset_id": DATASET_ID,
                        "dataset_version": DATASET_VERSION,
                        "sample_id": f"{split}-{number:03d}",
                        "split": "phase1_screening",
                        "content_type": content_type,
                        "condition": CONDITIONS[key],
                        "track": track,
                        "image": f"{root_relative}/{key}",
                        "image_sha256": sha256_bytes(live_image),
                        "reference_corpus": f"{root_relative}/{split}.json",
                        "reference_key": key,
                        "reference_sha256": sha256_bytes(text.encode("utf-8")),
                        "source_archive": archive_relative,
                        "source_archive_sha256": archive_sha256,
                        "source_member": source_member,
                        "source_member_sha256": sha256_bytes(source_markdown),
                        "annotation_method": ANNOTATION_METHOD,
                        "review_status": REVIEW_STATUS,
                        "provenance_status": PROVENANCE_STATUS,
                    }
                )

    if sum(len(records) for records in corpora.values()) != 20 or len(manifest) != 20:
        raise ValueError("smoke20-v1 must contain exactly 20 image/reference pairs")
    digest = dataset_digest(manifest)
    for row in manifest:
        row["dataset_sha256"] = digest
    return corpora, manifest


def serialized_outputs(
    corpora: dict[str, list[dict[str, str]]], manifest: list[dict[str, object]]
) -> dict[str, bytes]:
    outputs = {
        f"{split}.json": (json.dumps(corpora[split], ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        for split in SPLITS
    }
    outputs["manifest.jsonl"] = (
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in manifest) + "\n"
    ).encode("utf-8")
    return outputs


def check_outputs(root: Path, outputs: dict[str, bytes]) -> None:
    mismatches = []
    for name, expected in outputs.items():
        path = root / name
        if not path.is_file() or path.read_bytes() != expected:
            mismatches.append(str(path))
    if mismatches:
        raise ValueError("generated dataset differs from disk: " + ", ".join(mismatches))


def atomic_write_outputs(root: Path, outputs: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary: dict[str, Path] = {}
    try:
        for name, payload in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{name}.", suffix=".tmp", dir=root, delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary[name] = Path(handle.name)
        for name, path in temporary.items():
            os.replace(path, root / name)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate without writing.")
    mode.add_argument("--write", action="store_true", help="Atomically regenerate canonical outputs.")
    parser.add_argument("--root", type=Path, default=Path("small_bench"))
    parser.add_argument("--archive", type=Path, default=Path("small_bench.zip"))
    args = parser.parse_args()

    corpora, manifest = generate(args.archive, args.root)
    outputs = serialized_outputs(corpora, manifest)
    if args.check:
        check_outputs(args.root, outputs)
        action = "validated"
    else:
        atomic_write_outputs(args.root, outputs)
        check_outputs(args.root, outputs)
        action = "wrote"
    print(
        f"PASS {action} dataset={DATASET_ID} version={DATASET_VERSION} "
        f"samples={len(manifest)} dataset_sha256={manifest[0]['dataset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
