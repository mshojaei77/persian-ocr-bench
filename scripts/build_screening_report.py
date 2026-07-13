"""Build the validated Phase 1 screening report (never a ranked leaderboard)."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persian_ocr.reporting import cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli(default_mode="screening"))
