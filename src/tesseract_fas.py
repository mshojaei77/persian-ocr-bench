#!/usr/bin/env python3
"""Compatibility entry point for :mod:`persian_ocr.adapters.tesseract_fas`."""

from persian_ocr.adapters.tesseract_fas import *  # noqa: F403
from persian_ocr.adapters.tesseract_fas import main


if __name__ == "__main__":
    raise SystemExit(main())
