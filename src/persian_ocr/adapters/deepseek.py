"""Adapter for DeepSeek OCR and DeepSeek OCR 2.

Both models are loaded via ``AutoModel`` with ``trust_remote_code=True``
and use the model's own ``.infer()`` method.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


class _DeepSeekBase(OCRAdapter):
    """Shared logic for DeepSeek OCR v1 and v2."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = str(self.model_path or self.spec.repo_id)
        flash = self.spec.extra.get("flash_attention", False)
        kwargs = {"trust_remote_code": True, "use_safetensors": True}
        if flash:
            kwargs["_attn_implementation"] = "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, **kwargs).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.bfloat16)

    def predict(self, image: Image.Image) -> str:
        import tempfile
        import torch

        prompt = self.spec.prompt or "<image>\n<|grounding|>Convert the document to markdown."
        base_size = self.spec.extra.get("base_size", 768)
        image_size = self.spec.extra.get("image_size", 1344)
        crop_mode = self.spec.extra.get("crop_mode", True)
        test_compress = self.spec.extra.get("test_compress", True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_img = Path(tmp_dir) / "input.png"
            image.save(tmp_img)

            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(tmp_img),
                output_path=tmp_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=test_compress,
            )

        if isinstance(result, str) and result.strip():
            return self._strip_grounding(result)
        out_dir = Path(tmp_dir)
        candidates = sorted(out_dir.glob("*.md")) + sorted(out_dir.glob("*.txt"))
        if candidates:
            return self._strip_grounding(
                candidates[-1].read_text(encoding="utf-8", errors="ignore")
            )
        return self._strip_grounding(str(result))

    @staticmethod
    def _strip_grounding(text: str) -> str:
        text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|/?[a-zA-Z_]+\|>", "", text)
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in text.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()


class DeepSeekOCRAdapter(_DeepSeekBase):
    """DeepSeek OCR v1."""


class DeepSeekOCR2Adapter(_DeepSeekBase):
    """DeepSeek OCR v2."""
