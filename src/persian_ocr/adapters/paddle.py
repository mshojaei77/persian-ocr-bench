"""Adapter for PaddleOCR (PP-OCRv5 Persian and PaddleOCR-VL).

PP-OCRv5 loads the recognition model lazily on first inference.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


class PaddleOCRAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.ocr = None

    @classmethod
    def pull(cls, spec: ModelSpec, cache_dir: Path, token: str | None) -> Path | None:
        # PaddleOCR auto-downloads models on first use; nothing to do up front.
        return None

    def load(self) -> None:
        from paddleocr import PaddleOCR

        lang = self.spec.extra.get("lang", "fa")
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image: Image.Image) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            result = self.ocr.predict(tmp.name)

        lines = []
        for page in result:
            for region in page:
                if isinstance(region, (list, tuple)) and len(region) > 1:
                    text = region[1][0] if isinstance(region[1], (list, tuple)) else str(region[1])
                    if text.strip():
                        lines.append(text.strip())
        return "\n".join(lines)
