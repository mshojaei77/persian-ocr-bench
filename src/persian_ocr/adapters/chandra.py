"""Chandra OCR 2 adapter.

Chandra shares Surya's inference infrastructure but uses a different
model repo.  This adapter reuses the same backend configuration and
text-extraction logic.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.adapters.surya import SuryaAdapter
from persian_ocr.registry import ModelSpec


class ChandraAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self._surya_adapter = None

    def load(self) -> None:
        # Delegate to SuryaAdapter which handles the same inference stack
        self._surya_adapter = SuryaAdapter(
            spec=self.spec,
            model_path=self.model_path,
            device=self.device,
        )
        self._surya_adapter.load()

    def predict(self, image: Image.Image) -> str:
        if self._surya_adapter is None:
            raise RuntimeError("Call load() before predict()")
        return self._surya_adapter.predict(image)

    def close(self) -> None:
        if self._surya_adapter:
            self._surya_adapter.close()
            self._surya_adapter = None
