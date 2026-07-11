"""Adapters for traditional OCR engines:

- EasyOCR (Persian)
- Tesseract (fas)
- Kraken (fas/Arab)
- Hezar AI (crnn-base-fa-v2)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


# ── EasyOCR ───────────────────────────────────────────────────────────


class EasyOCRAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.reader = None

    @classmethod
    def pull(cls, spec: ModelSpec, cache_dir: Path, token: str | None) -> Path | None:
        # EasyOCR downloads models on first Reader instantiation.
        return None

    def load(self) -> None:
        import easyocr

        languages = self.spec.extra.get("languages", ["fa", "en"])
        self.reader = easyocr.Reader(
            languages,
            gpu=(self.device == "cuda"),
            model_storage_directory=str(self.model_path or "models/easyocr"),
            download_enabled=True,
        )

    def predict(self, image: Image.Image) -> str:
        import tempfile
        import numpy as np

        # EasyOCR prefers numpy array
        img_array = np.array(image.convert("RGB"))
        results = self.reader.readtext(img_array, detail=1, paragraph=True)
        lines = []
        for bbox, text, conf in results:
            if text.strip():
                lines.append(text.strip())
        return "\n".join(lines)


# ── Tesseract ─────────────────────────────────────────────────────────


class TesseractAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self._lang = self.spec.extra.get("language", "fas")

    @classmethod
    def pull(cls, spec: ModelSpec, cache_dir: Path, token: str | None) -> Path | None:
        # Tesseract data must be downloaded separately.
        import urllib.request
        from pathlib import Path

        tessdata_dir = Path("models/tessdata")
        tessdata_dir.mkdir(parents=True, exist_ok=True)
        out_path = tessdata_dir / "fas.traineddata"
        if not out_path.exists():
            url = "https://github.com/tesseract-ocr/tessdata/raw/main/fas.traineddata"
            urllib.request.urlretrieve(url, out_path)
        return tessdata_dir

    def load(self) -> None:
        import pytesseract
        self._pytesseract = pytesseract

    def predict(self, image: Image.Image) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            result = self._pytesseract.image_to_string(
                tmp.name, lang=self._lang
            )
        return result.strip()


# ── Kraken ────────────────────────────────────────────────────────────


class KrakenAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self._lang = self.spec.extra.get("language", "fas")
        self._script = self.spec.extra.get("script", "Arab")

    @classmethod
    def pull(cls, spec: ModelSpec, cache_dir: Path, token: str | None) -> Path | None:
        # Kraken models are managed via the `kraken` CLI.
        return None

    def load(self) -> None:
        pass  # kraken CLI is called per-image

    def predict(self, image: Image.Image) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp.name)
            result = subprocess.run(
                ["kraken", "-i", tmp.name, "ocr", "--text", "-"],
                capture_output=True, text=True, timeout=120,
            )
        return result.stdout.strip()


# ── Hezar AI ──────────────────────────────────────────────────────────


class HezarAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.model = None

    def load(self) -> None:
        from hezar.models import Model as HezarModel
        path = str(self.model_path or self.spec.repo_id or "hezarai/crnn-base-fa-v2")
        self.model = HezarModel.load(path)

    def predict(self, image: Image.Image) -> str:
        import tempfile
        import numpy as np

        img_array = np.array(image.convert("RGB"))
        outputs = self.model.predict(img_array)
        return self._extract_text(outputs)

    @staticmethod
    def _extract_text(outputs) -> str:
        if hasattr(outputs, "text"):
            return outputs.text.strip()
        if isinstance(outputs, dict):
            return outputs.get("text", str(outputs)).strip()
        return str(outputs).strip()
