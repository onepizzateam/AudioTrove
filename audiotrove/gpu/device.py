"""Centralised device resolution used by all GPU-aware components.

This module never fails on import even when torch is unavailable. Every helper
degrades gracefully to CPU so the pure-Python curation path keeps working.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a core dependency
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


def _mps_available() -> bool:
    """Return True when the Metal Performance Shaders backend is usable."""
    if not HAS_TORCH:
        return False
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # noqa: BLE001 - some torch builds raise here
        return False


def get_device(preference: str = "auto") -> "torch.device":
    """Resolve a torch device from a string preference.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
            ``"auto"`` picks CUDA > MPS > CPU in that order.

    Returns:
        A ``torch.device``. Falls back to CPU when the requested backend is
        unavailable so callers never crash on machines without a GPU.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for get_device()")

    preference = (preference or "auto").lower()

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if preference == "mps":
        if _mps_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    # "auto" (or anything unexpected) => best available backend
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict[str, Any]:
    """Return a dict describing detected backends for CLI display."""
    info: dict[str, Any] = {
        "torch_available": HAS_TORCH,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "mps_available": False,
        "selected": "cpu",
    }
    if not HAS_TORCH:
        return info

    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        count = torch.cuda.device_count()
        info["cuda_device_count"] = count
        info["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    info["mps_available"] = _mps_available()
    info["selected"] = str(get_device("auto"))
    return info


def to_device(tensor: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    """Move a tensor to ``device`` with a dtype appropriate for the backend.

    CUDA uses fp16 for floating point tensors to save memory/bandwidth; MPS and
    CPU keep fp32 for numerical stability. Integer/boolean tensors are moved
    without dtype changes.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for to_device()")

    device = device if isinstance(device, torch.device) else torch.device(device)

    if tensor.is_floating_point() and device.type == "cuda":
        return tensor.to(device=device, dtype=torch.float16)
    return tensor.to(device=device)
