from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing huggingface_hub.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download tencent/HunyuanOCR from Hugging Face.")
    parser.add_argument("--repo-id", default="tencent/HunyuanOCR")
    parser.add_argument("--local-dir", default="models/HunyuanOCR")
    parser.add_argument("--include-dflash", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir),
        ignore_patterns=[] if args.include_dflash else ["dflash/*", "v1.0/*"],
        max_workers=args.max_workers,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
