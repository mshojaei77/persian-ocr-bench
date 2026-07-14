"""Validate the generated smoke corpus without running model inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.jsonl"


def main() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    errors: list[str] = []
    expected_ids = [f"smoke_{number:03d}" for number in range(1, 21)]
    actual_ids = [record["sample_id"] for record in records]
    if actual_ids != expected_ids:
        errors.append(f"IDs are not exactly smoke_001..smoke_020: {actual_ids}")

    digests: set[str] = set()
    for record in records:
        sample_id = record["sample_id"]
        path = ROOT / record["image_path"]
        if not path.is_file():
            errors.append(f"{sample_id}: missing {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            errors.append(f"{sample_id}: SHA-256 mismatch")
        if digest in digests:
            errors.append(f"{sample_id}: duplicate image checksum")
        digests.add(digest)

        with Image.open(path) as image:
            if image.format != "PNG":
                errors.append(f"{sample_id}: stored master is not PNG")
            if image.width < 200 or image.height < 30:
                errors.append(f"{sample_id}: implausibly small dimensions {image.size}")
            grayscale = image.convert("L")
            spread = ImageStat.Stat(grayscale).extrema[0][1] - ImageStat.Stat(grayscale).extrema[0][0]
            if not record["is_blank"] and spread < 40:
                errors.append(f"{sample_id}: text image has insufficient contrast")
            if record["is_blank"] and ImageChops.invert(grayscale).getbbox() is None:
                errors.append(f"{sample_id}: blank case has no diagnostic non-text variation")

        reference = record["reference_raw"]
        if record["is_blank"] != (reference == ""):
            errors.append(f"{sample_id}: blank/reference contract mismatch")
        for span in record["numeric_spans"]:
            if reference[span["start"] : span["end"]] != span["raw"]:
                errors.append(f"{sample_id}: numeric span offsets do not select {span['raw']!r}")
        if "zwnj" in record["attributes"] and "\u200c" not in reference:
            errors.append(f"{sample_id}: zwnj attribute without U+200C")

    if errors:
        raise SystemExit("Smoke validation failed:\n- " + "\n- ".join(errors))
    print(f"Validated {len(records)} unique PNG smoke images, labels, spans, and checksums.")


if __name__ == "__main__":
    main()
