"""
Benchmark dataset loader.

Expects:
    small_bench/
        typed/      1.jpg  1.md  …  10.md
        hand-written/  1.jpg  1.md  …  10.md

Each ``.jpg`` has a sibling ``.md`` with the reference transcription.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True)
class BenchmarkSample:
    """A single image + ground-truth pair."""

    sample_id: str  # e.g. "typed/1"
    split: str  # "typed" or "hand-written"
    image_path: Path
    reference_path: Path
    reference: str


def load_dataset(root: Path) -> list[BenchmarkSample]:
    """Discover all image/reference pairs under *root*."""
    samples: list[BenchmarkSample] = []

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for image_path in sorted(split_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            ref_path = image_path.with_suffix(".md")
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Missing reference for {image_path}: {ref_path}"
                )
            samples.append(
                BenchmarkSample(
                    sample_id=f"{split}/{image_path.stem}",
                    split=split,
                    image_path=image_path,
                    reference_path=ref_path,
                    reference=ref_path.read_text(encoding="utf-8").strip(),
                )
            )

    if not samples:
        raise RuntimeError(f"No benchmark samples found under {root}")
    return samples
