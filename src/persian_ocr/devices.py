"""Runtime device selection shared by the neural OCR adapters.

``auto`` deliberately prefers an available accelerator and falls back to CPU
when the installed runtime is CPU-only or no accelerator is available. An
explicit device remains an explicit request and is passed through unchanged.
"""

from __future__ import annotations


DEFAULT_DEVICE = "auto"


def resolve_torch_device(requested: str) -> str:
    """Resolve ``auto`` for Torch-backed runtimes (CUDA, MPS, then CPU)."""
    if requested != DEFAULT_DEVICE:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_paddle_device(requested: str) -> str:
    """Resolve ``auto`` for PaddleOCR (GPU first, then CPU)."""
    if requested != DEFAULT_DEVICE:
        return requested
    try:
        import paddle
    except ImportError:
        return "cpu"
    if paddle.device.is_compiled_with_cuda():
        try:
            if paddle.device.cuda.device_count() > 0:
                return "gpu:0"
        except (AttributeError, RuntimeError):
            # A partially installed or CPU-only Paddle build must still use
            # the documented CPU fallback instead of breaking auto mode.
            pass
    return "cpu"


__all__ = ["DEFAULT_DEVICE", "resolve_paddle_device", "resolve_torch_device"]
