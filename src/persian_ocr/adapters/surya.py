"""Surya OCR 2 adapter.

Uses ``surya.recognition.RecognitionPredictor`` with its inference
manager.  Text is extracted from ``PageResult.blocks[].html`` via
BeautifulSoup.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


class SuryaAdapter(OCRAdapter):
    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.manager = None
        self.predictor = None
        self._configured_backend = None

    def load(self) -> None:
        self._configure_backend()
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        self.manager = SuryaInferenceManager()
        self.predictor = RecognitionPredictor(self.manager)

        # Log model provenance
        print(f"  model_source={getattr(self.predictor, 'model_name', None)}")
        print(f"  model_revision={getattr(self.predictor, 'model_revision', None)}")
        print(f"  device={getattr(self.predictor, 'device', None)}")

    def predict(self, image: Image.Image) -> str:
        if self.predictor is None:
            raise RuntimeError("Call load() before predict()")
        prediction = self.predictor([image.convert("RGB")])[0]
        return self._extract_text(prediction)

    def close(self) -> None:
        self.predictor = None
        self.manager = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── helpers ─────────────────────────────────────────────────────

    def _configure_backend(self) -> None:
        backend = self.spec.extra.get("backend")
        inference_url = self.spec.extra.get("inference_url")

        if backend:
            os.environ["SURYA_INFERENCE_BACKEND"] = backend
        if inference_url:
            os.environ["SURYA_INFERENCE_URL"] = inference_url
        os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")
        os.environ.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "1")

        url = os.environ.get("SURYA_INFERENCE_URL")
        if url:
            self._configured_backend = "remote"
            return

        be = os.environ.get("SURYA_INFERENCE_BACKEND")
        if be is None and not shutil.which("docker"):
            if shutil.which("llama-server"):
                os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
                self._configured_backend = "llamacpp"
                return
            raise RuntimeError(
                "Surya needs a serving backend. Install Docker, install llama-server, "
                "or pass --backend / --inference-url."
            )
        self._configured_backend = be or "auto"

    @staticmethod
    def _extract_text(page_result) -> str:
        from bs4 import BeautifulSoup

        blocks = getattr(page_result, "blocks", None)
        if blocks is None and isinstance(page_result, dict):
            blocks = page_result.get("blocks", [])

        parts = []
        for block in sorted(
            blocks,
            key=lambda b: (
                getattr(b, "reading_order", None)
                if not isinstance(b, dict)
                else b.get("reading_order", 0)
            )
            or 0,
        ):
            skipped = (
                getattr(block, "skipped", False)
                if not isinstance(block, dict)
                else block.get("skipped", False)
            )
            err = (
                getattr(block, "error", False)
                if not isinstance(block, dict)
                else block.get("error", False)
            )
            if skipped or err:
                continue
            html = (
                getattr(block, "html", "")
                if not isinstance(block, dict)
                else block.get("html", "")
            )
            if not html:
                continue

            text = BeautifulSoup(html, "html.parser").get_text(
                separator="\n", strip=True
            )
            if text:
                parts.append(text)

        return SuryaAdapter._clean("\n".join(parts))

    @staticmethod
    def _clean(text: str) -> str:
        import re
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in text.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()
