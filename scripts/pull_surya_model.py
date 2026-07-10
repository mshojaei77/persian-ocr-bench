from __future__ import annotations

import os

# Must be set before importing huggingface_hub
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def _print_existing_partials(local_dir: Path) -> None:
    download_dir = local_dir / ".cache" / "huggingface" / "download"
    partials = sorted(download_dir.glob("*.incomplete"), key=lambda p: p.stat().st_size, reverse=True)
    if not partials:
        return

    total_mb = sum(path.stat().st_size for path in partials) / 1024 / 1024
    print(f"Found {len(partials)} incomplete download file(s): {total_mb:.2f} MB already on disk")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Surya OCR 2 weights from Hugging Face."
    )
    parser.add_argument("--repo-id", default="datalab-to/surya-ocr-2")
    parser.add_argument("--local-dir", default="models/surya-ocr-2")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    _print_existing_partials(local_dir)

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=local_dir,
        max_workers=args.max_workers,
    )

    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
