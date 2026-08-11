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


def ipex_available() -> bool:
    """Return True when Intel Extension for PyTorch (IPEX) is importable.

    IPEX accelerates CPU (and Intel GPU) training/inference. It is entirely
    optional; every caller must work without it.
    """
    if not HAS_TORCH:
        return False
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception:  # noqa: BLE001 - ImportError or partial/broken installs
        return False
    return True


def bf16_supported(device: "torch.device") -> bool:
    """Return True when bfloat16 is supported on ``device``.

    - CUDA: supported on compute capability >= 8.0 (Ampere and newer).
    - CPU: supported when the build reports an AVX512 capability, which is the
      practical requirement for usable bf16 throughput.
    - MPS: not supported.
    """
    if not HAS_TORCH:
        return False

    device = device if isinstance(device, torch.device) else torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        try:
            major, _minor = torch.cuda.get_device_capability(device)
            return major >= 8
        except Exception:  # noqa: BLE001
            return False

    if device.type == "cpu":
        try:
            cap = torch.backends.cpu.get_cpu_capability()
            return "avx512" in str(cap).lower()
        except Exception:  # noqa: BLE001 - attribute missing on older torch
            return False

    # MPS and any other backend
    return False


def optimal_dtype(device: "torch.device") -> "torch.dtype":
    """Pick the best training/inference dtype for ``device``.

    Preference order: bfloat16 (when supported) > float16 (CUDA only) > float32.
    bf16 is preferred over fp16 for its wider dynamic range and greater training
    stability; fp16 is used on CUDA GPUs that predate bf16 support; everything
    else stays in fp32.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for optimal_dtype()")

    device = device if isinstance(device, torch.device) else torch.device(device)

    if bf16_supported(device):
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32
