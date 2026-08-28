__version__ = "0.1.2"

from audiotrove.base import AudioFilter, AudioTransformer, GPUFilter, GPUTransformer
from audiotrove.document import AudioDocument
from audiotrove.pipelines.tts import tts_pipeline


def gpu_available() -> bool:
    """Return whether a CUDA or MPS backend is usable."""
    try:
        from audiotrove.gpu.device import get_device
        return get_device("auto").type != "cpu"
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "AudioDocument", "AudioFilter", "AudioTransformer", "GPUFilter",
    "GPUTransformer", "gpu_available", "tts_pipeline",
]
