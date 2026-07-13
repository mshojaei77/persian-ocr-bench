#!/usr/bin/env python3
"""Compatibility entry point for :mod:`persian_ocr.adapters.easyocr_fa`."""

from persian_ocr.adapters.easyocr_fa import *  # noqa: F403
from persian_ocr.adapters.easyocr_fa import main


if __name__ == "__main__":
    raise SystemExit(main())
