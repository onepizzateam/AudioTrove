"""Silence trimming transformation."""

import numpy as np

from audiotrove.base import GPUTransformer
from audiotrove.document import AudioDocument
from audiotrove.filters.vad import SileroVADFilter

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a core dependency
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


def _resolve_device(device):
    if not HAS_TORCH:
        return None
    try:
        if isinstance(device, torch.device):
            return device
        return torch.device(device)
    except Exception:  # noqa: BLE001 - tolerate stubbed torch in tests
        return None



class SilenceTrimmingTransformer(GPUTransformer):
    """Trim leading and trailing silence around VAD speech timestamps."""

    name = "silence_trim"

    def __init__(
        self,
        padding_ms: int = 150,
        min_duration_seconds: float = 1.0,
        device: str = "cpu",
    ):
        """Initialize the silence trimming transformer.

        Args:
            padding_ms: Silence to preserve on each side of detected speech.
            min_duration_seconds: Minimum duration allowed after trimming.
            device: Torch device string for the GPU trim path.
        """
        self.padding_ms = padding_ms
        self.min_duration_seconds = min_duration_seconds
        self._device = _resolve_device(device)
        self._vad_filter = None

    @property
    def device(self):
        """The torch.device this transformer operates on (None without torch)."""
        return self._device

    def to(self, device) -> "SilenceTrimmingTransformer":
        """Set the compute device and return self."""
        self._device = _resolve_device(device)
        return self

    def __getstate__(self):
        """Exclude lazy-loaded VAD model state from pickling."""
        state = self.__dict__.copy()
        state["_vad_filter"] = None
        state["_device"] = str(self._device) if self._device is not None else "cpu"
        return state

    def __setstate__(self, state):
        device = state.pop("_device", "cpu")
        self.__dict__.update(state)
        self._device = _resolve_device(device)

    def _get_timestamps(self, doc: AudioDocument) -> list[dict]:
        """Return existing VAD timestamps or run Silero VAD inline."""
        timestamps = doc.metadata.get("vad_speech_timestamps")
        if timestamps is not None:
            return timestamps

        if self._vad_filter is None:
            self._vad_filter = SileroVADFilter()
        self._vad_filter.filter(doc)
        return doc.metadata.get("vad_speech_timestamps", [])

    def transform(self, doc: AudioDocument) -> AudioDocument:
        """Trim silence from an audio document while preserving minimum duration.

        When ``doc.gpu_tensor`` is present the trim is computed on-device and the
        resulting tensor is cached back on the document; otherwise the numpy
        path is used. In both cases ``doc.audio`` stays the source of truth.

        Args:
            doc: Input AudioDocument.

        Returns:
            The input document with trimmed audio and updated metadata.
        """
        timestamps = self._get_timestamps(doc)
        if not timestamps:
            doc.metadata["trimmed_duration_seconds"] = doc.duration_seconds
            return doc

        padding_samples = int(doc.sample_rate * self.padding_ms / 1000)
        length = len(doc.audio)
        start = max(0, min(int(ts["start"]) for ts in timestamps) - padding_samples)
        end = min(length, max(int(ts["end"]) for ts in timestamps) + padding_samples)
        trimmed_duration = float(end - start) / doc.sample_rate

        if trimmed_duration < self.min_duration_seconds:
            doc.metadata["trimmed_duration_seconds"] = doc.duration_seconds
            return doc

        gpu_tensor = getattr(doc, "gpu_tensor", None)
        if gpu_tensor is not None and HAS_TORCH and torch.is_tensor(gpu_tensor):
            trimmed_t = gpu_tensor[start:end]
            doc.gpu_tensor = trimmed_t
            doc.audio = np.ascontiguousarray(
                trimmed_t.detach().cpu().numpy(), dtype=np.float32
            )
        else:
            doc.audio = np.ascontiguousarray(doc.audio[start:end], dtype=np.float32)

        doc.duration_seconds = trimmed_duration
        doc.metadata["vad_speech_timestamps"] = [
            {
                "start": max(0, int(ts["start"]) - start),
                "end": min(end - start, int(ts["end"]) - start),
            }
            for ts in timestamps
        ]
        doc.metadata["trimmed_duration_seconds"] = trimmed_duration
        return doc
