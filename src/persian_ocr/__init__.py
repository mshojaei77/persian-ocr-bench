"""Shared, runtime-light foundations for the Persian OCR benchmark."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("persian-ocr")
except PackageNotFoundError:  # Source tree before the editable install exists.
    __version__ = "0.2.0"

__all__ = ["__version__"]
