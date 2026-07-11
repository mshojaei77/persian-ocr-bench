"""Generic adapter for standard HuggingFace transformer VLMs.

Handles models loaded via ``AutoModelForVision2Seq`` or
``AutoProcessor``.  Individual model quirks are configured through the
``ModelSpec.extra`` dict rather than subclassing.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from persian_ocr.adapters.base import OCRAdapter
from persian_ocr.registry import ModelSpec


class GenericVLMAdapter(OCRAdapter):
    """For any VLM loadable via AutoProcessor + AutoModelForVision2Seq."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        path = str(self.model_path or self.spec.repo_id)
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=self.spec.trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=self.spec.trust_remote_code,
        ).eval()

    def predict(self, image: Image.Image) -> str:
        import torch

        prompt = self.spec.prompt or "<image>\nExtract all visible text from this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        output = self.processor.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return self._clean(output)

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<\|/?[a-zA-Z_]+\|>", "", text)
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in text.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()


class QwenVLAdapter(OCRAdapter):
    """Adapter for Qwen2/3-VL models via Qwen2VLForConditionalGeneration."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        path = str(self.model_path or self.spec.repo_id)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def predict(self, image: Image.Image) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self.spec.prompt or "Extract all visible text from this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        output = self.processor.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return self._clean(output)

    @staticmethod
    def _clean(text: str) -> str:
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in text.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()


class WeightedAIAdapter(OCRAdapter):
    """Adapter for WeightedAI/Persian_OCR (likely a standard VLM)."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        path = str(self.model_path or self.spec.repo_id)
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def predict(self, image: Image.Image) -> str:
        import torch

        prompt = self.spec.prompt or "Extract all visible text from this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        output = self.processor.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return self._clean(output)

    @staticmethod
    def _clean(text: str) -> str:
        import re
        text = re.sub(r"<\|/?[a-zA-Z_]+\|>", "", text)
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in text.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()


class KhanandehAdapter(OCRAdapter):
    """Adapter for Khanandeh — PEFT LoRA on Qwen2-VL-2B-Instruct."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path | None,
        device: str,
    ) -> None:
        super().__init__(spec, model_path, device)
        self.processor = None
        self.model = None
        self.base_path = None
        self.adapter_path = None

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        base_repo = self.spec.extra.get("base_repo_id", "unsloth/qwen2-vl-2b-instruct-unsloth-bnb-4bit")
        adapter_repo = self.spec.repo_id

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_repo,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_repo)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            adapter_repo, trust_remote_code=True
        )

    def predict(self, image: Image.Image) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self.spec.prompt or "Extract all visible text from this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        output = self.processor.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        lines = [
            re.sub(r"[ \t]+", " ", ln).strip()
            for ln in output.replace("\r\n", "\n").split("\n")
        ]
        return "\n".join(ln for ln in lines if ln).strip()
