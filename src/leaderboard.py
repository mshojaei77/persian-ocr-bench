"""Compatibility entry point for the validated final leaderboard report.

Phase 1 uses ``scripts/build_screening_report.py`` and never emits numeric
ranks.  This historical entry point now routes to the strict v2 reporting
boundary and defaults to final-benchmark mode.
"""

from __future__ import annotations

from persian_ocr.reporting import cli


def main() -> int:
    return cli(default_mode="final")


if __name__ == "__main__":
    raise SystemExit(main())
