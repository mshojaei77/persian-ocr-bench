"""Deterministic, dependency-free regression smoke for smoke20-v1."""

from __future__ import annotations

import argparse
from pathlib import Path

from merge_small_bench import (
    DATASET_ID,
    DATASET_VERSION,
    check_outputs,
    generate,
    serialized_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("small_bench"))
    parser.add_argument("--archive", type=Path, default=Path("small_bench.zip"))
    args = parser.parse_args()

    corpora, manifest = generate(args.archive, args.root)
    check_outputs(args.root, serialized_outputs(corpora, manifest))
    records = {
        record["image"]: record["text"]
        for split_records in corpora.values()
        for record in split_records
    }

    typed9 = records["typed/9.jpg"]
    required_typed9 = (
        "تجدیدنظرخواه: آقای",
        "تجدیدنظرخوانده: خانم",
        "در خصوص تجدیدنظرخواهی آقای",
        "Scanned with CamScanner",
    )
    missing = [text for text in required_typed9 if text not in typed9]
    if missing:
        raise RuntimeError(f"typed/9 recovery regression; missing={missing}")
    if "#چشم_هایش" not in records["hand-written/5.jpg"]:
        raise RuntimeError("underscore/hashtag recovery regression in hand-written/5")
    if "۱۰۲ | تکه‌هایی" not in records["typed/3.jpg"]:
        raise RuntimeError("visible non-table pipe recovery regression in typed/3")
    if any("<br>" in text.lower() or ":---" in text for text in records.values()):
        raise RuntimeError("Markdown/HTML presentation markup survived recovery")
    if len(records) != 20 or len(manifest) != 20:
        raise RuntimeError("smoke20-v1 must contain exactly 20 validated samples")

    handwritten7 = next(row for row in manifest if row["sample_id"] == "hand-written-007")
    if handwritten7["content_type"] != "handwritten" or handwritten7["track"] != "handwriting_smoke":
        raise RuntimeError("hand-written/7 classification regression")
    if any(row["review_status"] == "human_reviewed" for row in manifest):
        raise RuntimeError("dataset must not claim a human review that did not occur")

    print(
        f"PASS dataset={DATASET_ID} version={DATASET_VERSION} samples=20 "
        f"typed9_recovery=ok dataset_sha256={manifest[0]['dataset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
