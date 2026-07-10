from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing huggingface_hub.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Unlimited-OCR GGUF model and mmproj.")
    parser.add_argument("--repo-id", default="sahilchachra/Unlimited-OCR-GGUF")
    parser.add_argument("--local-dir", default="models/Unlimited-OCR-GGUF")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    include = [
        f"Unlimited-OCR-{args.quant}.gguf",
        "mmproj-Unlimited-OCR-F16.gguf",
    ]
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir),
        allow_patterns=include,
        max_workers=args.max_workers,
    )
    print(f"Downloaded to: {path}")
    print(f"Model: {Path(path) / include[0]}")
    print(f"MMProj: {Path(path) / include[1]}")


if __name__ == "__main__":
    main()
