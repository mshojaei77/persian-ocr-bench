"""OCRAdapter base class and dynamic resolver."""

from __future__ import annotations

import importlib

from persian_ocr.adapters.base import OCRAdapter


def resolve_adapter(import_path: str) -> type[OCRAdapter]:
    """Dynamically import an adapter class from a dotted ``module:Class`` path."""
    module_name, class_name = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, OCRAdapter):
        raise TypeError(f"{import_path} must be a class inheriting OCRAdapter")
    return cls
