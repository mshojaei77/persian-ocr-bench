#!/usr/bin/env python3
"""Compatibility entry point for the packaged PP-OCRv5 adapter."""

from persian_ocr.adapters.ppocrv5_arabic_mobile_rec import *  # noqa: F403
from persian_ocr.adapters.ppocrv5_arabic_mobile_rec import main


if __name__ == "__main__":
    raise SystemExit(main())
