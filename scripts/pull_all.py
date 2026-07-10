from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_STATE_FILE = ROOT / "models" / ".pull_all_state.json"


def pull_scripts() -> list[Path]:
    return sorted(
        path
        for path in SCRIPTS_DIR.glob("pull_*_model.py")
        if path.name != Path(__file__).name
    )


def load_done(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return set(data.get("done", []))


def save_done(state_file: Path, done: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps({"done": sorted(done)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull every OCR model one by one, resumably.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--restart", action="store_true", help="Ignore previous completed-script state.")
    parser.add_argument("--list", action="store_true", help="Print pull order and exit.")
    args = parser.parse_args()

    scripts = pull_scripts()
    if args.list:
        for script in scripts:
            print(script.name)
        return 0

    done = set() if args.restart else load_done(args.state_file)
    for index, script in enumerate(scripts, start=1):
        if script.name in done:
            print(f"[{index}/{len(scripts)}] skip {script.name}")
            continue

        print(f"[{index}/{len(scripts)}] run {script.name}", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode:
            print(f"Failed: {script.name}. Re-run this command to resume.", file=sys.stderr)
            return result.returncode

        done.add(script.name)
        save_done(args.state_file, done)

    print("All pull scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
