from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing huggingface_hub.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PaddleOCR-VL-0.9B from Hugging Face.")
    parser.add_argument("--repo-id", default="lvyufeng/PaddleOCR-VL-0.9B")
    parser.add_argument("--local-dir", default="models/PaddleOCR-VL-0.9B")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir),
        max_workers=args.max_workers,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
