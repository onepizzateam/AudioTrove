"""Optional Rust IO accelerators with a dependency-free Python fallback."""

import numpy as np

try:
    import audiotrove_core as _core
except (ImportError, ModuleNotFoundError):
    _core = None


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float audio, using Rust when the optional extension exists."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    if _core is not None and hasattr(_core, "resample"):
        return np.asarray(_core.resample(audio.tolist(), source_rate, target_rate), dtype=np.float32)
    count = int(len(audio) * float(target_rate) / float(source_rate))
    return np.interp(np.linspace(0, len(audio), count, endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)
