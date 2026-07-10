from __future__ import annotations

import os

# Must be set before importing huggingface_hub
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Surya OCR 2 weights from Hugging Face."
    )
    parser.add_argument("--repo-id", default="datalab-to/surya-ocr-2")
    parser.add_argument("--local-dir", default="models/surya-ocr-2")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir),
        max_workers=8,
    )

    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()