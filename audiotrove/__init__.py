__version__ = "0.3.0"


from audiotrove.base import (
    AudioFilter,
    AudioTransformer,
    GPUFilter,
    GPUTransformer,
)
from audiotrove.document import AudioDocument
from audiotrove.pipelines.tts import tts_pipeline


def gpu_available() -> bool:
    """Return True when a CUDA or MPS backend is usable on this machine."""
    try:
        from audiotrove.gpu.device import get_device

        return get_device("auto").type != "cpu"
    except Exception:  # noqa: BLE001 - never fail feature detection
        return False


__all__ = [
    "AudioDocument",
    "AudioFilter",
    "AudioTransformer",
    "GPUFilter",
    "GPUTransformer",
    "tts_pipeline",
    "gpu_available",
]
