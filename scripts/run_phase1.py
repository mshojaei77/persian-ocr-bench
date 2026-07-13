"""Run the Phase 1 smoke20 screen sequentially with resumable state.

This orchestrator intentionally invokes the compatibility entry points under
``src/``.  They remain stable while the installed package and adapters evolve.
Run it from any working directory; all paths are anchored to the repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "bench_runs" / "smoke20-v1"
STATE_PATH = RUN_ROOT / "state.json"
LOG_ROOT = RUN_ROOT / "logs"
ARCHIVE_ROOT = REPO_ROOT / "bench_runs" / "archives"

MODEL_COMMANDS: dict[str, list[str]] = {
    "tesseract_fas": [
        "src/tesseract_fas.py",
        "--small_bench",
        "--langs",
        "fas",
        "--psm",
        "3",
        "--variant",
        "best",
        "--preprocess",
        "raw",
        "--require-reviewed",
    ],
    "ppocrv5_arabic_mobile_rec": [
        "src/ppocrv5_arabic_mobile_rec.py",
        "--small_bench",
        "--device",
        "auto",
        "--preprocess",
        "raw",
        "--require-reviewed",
    ],
    "easyocr_fa": [
        "src/easyocr_fa.py",
        "--small_bench",
        "--device",
        "auto",
        "--preprocess",
        "raw",
        "--ordering",
        "rtl_rows",
        "--require-reviewed",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"schema": "persian_ocr_phase1_state_v1", "models": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def output_path(model_id: str) -> Path:
    return RUN_ROOT / f"{model_id}.json"


def reset_phase1() -> Path | None:
    """Archive an existing screen before starting a fresh, reproducible attempt."""
    if not RUN_ROOT.exists():
        return None
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = ARCHIVE_ROOT / f"smoke20-v1-{stamp}"
    suffix = 1
    while archive.exists():
        suffix += 1
        archive = ARCHIVE_ROOT / f"smoke20-v1-{stamp}-{suffix}"
    shutil.move(str(RUN_ROOT), str(archive))
    return archive


def build_command(model_id: str, limit: int | None, device: str) -> list[str]:
    command = [sys.executable, *MODEL_COMMANDS[model_id]]
    if model_id in {"ppocrv5_arabic_mobile_rec", "easyocr_fa"}:
        device_index = command.index("--device") + 1
        command[device_index] = device
    if limit is not None:
        command.extend(["--limit", str(limit)])
    command.extend(["--output", str(output_path(model_id).relative_to(REPO_ROOT))])
    return command


def run_model(model_id: str, command: list[str]) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{model_id}.log"
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    print(f"\n[{model_id}] {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                safe_line = line.encode(encoding, errors="replace").decode(encoding)
                sys.stdout.write(safe_line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def capture_provenance() -> None:
    """Persist commands and environment metadata inside the archivable run root."""
    subprocess.run(
        [sys.executable, "scripts/capture_benchmark_provenance.py"],
        cwd=REPO_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_COMMANDS),
        help="Model to run; repeat the option. Defaults to every implemented adapter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for neural adapters: auto (GPU first, then CPU), or explicit device.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun successful models.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Archive the current smoke20-v1 state and artifacts, then start a fresh Phase 1 run.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue after a failed adapter."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.reset:
        archive = reset_phase1()
        if archive:
            print(f"Archived prior Phase 1 run: {archive.relative_to(REPO_ROOT)}")
        else:
            print("No prior Phase 1 run exists; starting fresh.")
    selected = args.model or list(MODEL_COMMANDS)
    state = load_state()
    state["updated_at"] = utc_now()
    state["dataset"] = "smoke20-v1"
    state["limit"] = args.limit
    model_state = state.setdefault("models", {})

    failed = False
    for model_id in selected:
        previous = model_state.get(model_id, {})
        if (
            not args.force
            and previous.get("status") == "complete"
            and output_path(model_id).is_file()
        ):
            print(f"[{model_id}] already complete; use --force to rerun")
            continue
        command = build_command(model_id, args.limit, args.device)
        if args.dry_run:
            print(f"[{model_id}] {' '.join(command)}")
            continue
        model_state[model_id] = {
            "status": "running",
            "started_at": utc_now(),
            "command": command,
            "output": str(output_path(model_id).relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        write_state(state)
        try:
            return_code = run_model(model_id, command)
            run_error = None
        except Exception as exc:  # keep resumable state even if orchestration fails
            return_code = 1
            run_error = f"{type(exc).__name__}: {exc}"
        complete = return_code == 0 and output_path(model_id).is_file()
        model_state[model_id].update(
            {
                "status": "complete" if complete else "failed",
                "finished_at": utc_now(),
                "return_code": return_code,
                "error": run_error,
            }
        )
        write_state(state)
        if not complete:
            failed = True
            if not args.continue_on_error:
                break
    capture_provenance()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
