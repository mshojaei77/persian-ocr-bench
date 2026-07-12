"""Named, reproducible image preprocessing profiles for Tesseract benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    min_width: int = 0
    deskew: bool = False
    denoise: bool = False
    clahe: bool = False
    threshold: str = "none"
    border_px: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROFILES = {
    "raw": PreprocessConfig("raw"),
    "grayscale": PreprocessConfig("grayscale"),
    "upscale": PreprocessConfig("upscale", min_width=1800),
    "document_clean": PreprocessConfig(
        "document_clean",
        min_width=1800,
        deskew=True,
        denoise=True,
        clahe=True,
        threshold="adaptive",
        border_px=12,
    ),
}


def _deskew(gray: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(gray)
    points = np.column_stack(np.where(inverted > 0))
    if len(points) < 20:
        return gray
    angle = cv2.minAreaRect(points)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.05 or abs(angle) > 15:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess_image(image: Image.Image, profile: PreprocessConfig) -> Image.Image:
    if profile.name == "raw":
        return image.convert("RGB")
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if profile.min_width and gray.shape[1] < profile.min_width:
        scale = profile.min_width / gray.shape[1]
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    if profile.denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=8)
    if profile.clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if profile.deskew:
        gray = _deskew(gray)
    if profile.threshold == "adaptive":
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
    if profile.border_px:
        gray = cv2.copyMakeBorder(
            gray,
            profile.border_px,
            profile.border_px,
            profile.border_px,
            profile.border_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
    return Image.fromarray(gray)
