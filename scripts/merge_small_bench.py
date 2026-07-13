"""Convert small_bench Markdown references into plain-text JSON corpora."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE_DESCRIPTION = re.compile(r"\[[^\]]*\]")
MARKDOWN = re.compile(r"(```?|[*_~])")


def repair_text(text: str) -> str:
    """Keep visible OCR text while removing presentation and image metadata."""
    repaired = text
    if any(marker in text for marker in ("Ã", "Â", "Ø", "Ù", "â")):
        for source_encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(source_encoding).decode("utf-8")
            except UnicodeError:
                continue
            if sum(text.count(marker) for marker in ("Ã", "Â", "Ø", "Ù", "â")) > sum(
                candidate.count(marker) for marker in ("Ã", "Â", "Ø", "Ù", "â")
            ):
                repaired = candidate
                break
    lines: list[str] = []
    for raw_line in repaired.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or re.fullmatch(r"[-=*#_~` ]+", line):
            continue
        if IMAGE_DESCRIPTION.search(line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = MARKDOWN.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def convert(split_dir: Path, output: Path) -> None:
    records = []
    for source in sorted(split_dir.glob("*.md"), key=lambda p: int(p.stem)):
        text = repair_text(source.read_text(encoding="utf-8"))
        if not text:
            raise ValueError(f"empty cleaned text: {source}")
        records.append({"image": f"{split_dir.name}/{source.stem}.jpg", "text": text})
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("small_bench"))
    args = parser.parse_args()
    convert(args.root / "hand-written", args.root / "hand-written.json")
    convert(args.root / "typed", args.root / "typed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
