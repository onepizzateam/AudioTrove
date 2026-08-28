"""Optional Rust IO accelerators with a dependency-free Python fallback."""

import numpy as np
import glob as _pyglob

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


def glob_paths(pattern: str) -> list[str]:
    """Expand local paths with Rust when present, otherwise pathlib."""
    if _core is not None and hasattr(_core, "glob_paths"):
        return list(_core.glob_paths(pattern))
    return _pyglob.glob(pattern)


def decode_wav(path: str):
    """Decode a local WAV with Rust when available; return ``None`` otherwise."""
    if _core is None or not hasattr(_core, "decode_wav"):
        return None
    audio, sample_rate = _core.decode_wav(path)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)
