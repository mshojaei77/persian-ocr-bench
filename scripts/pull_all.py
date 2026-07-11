"""Resumable coordinator for the unified Hugging Face downloader."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = ROOT / "models" / ".pull_all_state.json"
DEFAULT_MANIFEST = ROOT / "models" / "manifest.json"


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("done", []))


def save_state(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"done": sorted(done)}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    from persian_ocr.registry import MODELS

    parser = argparse.ArgumentParser(
        description="Pull registered OCR models with resumable HF caching."
    )
    parser.add_argument(
        "--model", "--models", default="all",
        help="Model ID, comma-separated IDs, or 'all'.",
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "huggingface")
    parser.add_argument("--restart", action="store_true", help="Ignore completed model state.")
    parser.add_argument("--list", action="store_true", help="Print model order and exit.")
    args = parser.parse_args()

    model_ids = list(MODELS) if args.model == "all" else [x.strip() for x in args.model.split(",") if x.strip()]
    unknown = sorted(set(model_ids) - set(MODELS))
    if unknown:
        parser.error(f"Unknown models: {', '.join(unknown)}")
    if args.list:
        print("\n".join(model_ids))
        return 0

    done = set() if args.restart else load_state(args.state_file)
    command = [
        sys.executable, str(ROOT / "scripts" / "pull.py"),
        "--cache-dir", str(args.cache_dir), "--manifest", str(args.manifest),
    ]
    env = os.environ.copy()
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    env.setdefault("HF_XET_CACHE", str(args.cache_dir / "xet"))

    for index, model_id in enumerate(model_ids, 1):
        if model_id in done:
            print(f"[{index}/{len(model_ids)}] skip {model_id}", flush=True)
            continue

        print(f"[{index}/{len(model_ids)}] pull {model_id}", flush=True)
        result = subprocess.run([*command, "--model", model_id], cwd=ROOT, env=env)
        if result.returncode:
            print(f"Failed: {model_id}. Re-run to resume; HF cache is retained.", file=sys.stderr)
            return result.returncode

        done.add(model_id)
        save_state(args.state_file, done)

    print("All selected models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
