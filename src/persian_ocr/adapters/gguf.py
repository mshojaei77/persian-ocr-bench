"""Adapter for GGUF-quantized models via llama.cpp.

Currently supports Unlimited-OCR-GGUF.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


class UnlimitedOCRAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self._model_path = None
        self._mmproj_path = None

    def load(self) -> None:
        import re

        base = self.model_path
        if base is None:
            raise RuntimeError("Model must be downloaded before loading")

        quant = self.spec.extra.get("quant", "Q4_K_M")
        files = list(base.iterdir())
        self._model_path = next(
            (f for f in files if quant in f.name and f.suffix == ".gguf"), None
        )
        self._mmproj_path = next(
            (f for f in files if f.name.startswith("mmproj")), None
        )

        if self._model_path is None:
            raise FileNotFoundError(
                f"GGUF model file with '{quant}' not found in {base}"
            )
        if self._mmproj_path is None:
            raise FileNotFoundError(f"mmproj file not found in {base}")

        # Verify llama-server is available
        result = subprocess.run(
            ["llama-server", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "llama-server is required for GGUF inference. "
                "Install from https://github.com/ggml-org/llama.cpp"
            )

    def predict(self, image: Image.Image) -> str:
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "llama-cli",
                    "-m", str(self._model_path),
                    "--mmproj", str(self._mmproj_path),
                    "--image", tmp_path,
                    "-p", "Extract all visible text from this image.",
                    "--temp", "0",
                    "--no-display-prompt",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result.stdout.strip()
