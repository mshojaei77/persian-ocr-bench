from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from persian_ocr.registry import ModelSpec


class OCRAdapter(ABC):
    def __init__(self, spec: ModelSpec, model_path: Path | None, device: str) -> None:
        self.spec = spec
        self.model_path = model_path
        self.device = device

    @classmethod
    def pull(cls, spec: ModelSpec, cache_dir: Path, token: str | None) -> Path | None:
        return None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image) -> str:
        ...

    def close(self) -> None:
        pass
