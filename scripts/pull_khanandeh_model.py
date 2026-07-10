from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing huggingface_hub.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


def download(repo_id: str, local_dir: str, max_workers: int) -> None:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=Path(local_dir),
        max_workers=max_workers,
    )
    print(f"{repo_id} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Khanandeh Persian OCR adapter and base model.")
    parser.add_argument("--adapter-repo-id", default="oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct")
    parser.add_argument("--base-repo-id", default="unsloth/qwen2-vl-2b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--adapter-dir", default="models/Khanandeh-0.1-Persian-OCR-2B-Instruct")
    parser.add_argument("--base-dir", default="models/qwen2-vl-2b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    download(args.adapter_repo_id, args.adapter_dir, args.max_workers)
    download(args.base_repo_id, args.base_dir, args.max_workers)


if __name__ == "__main__":
    main()
