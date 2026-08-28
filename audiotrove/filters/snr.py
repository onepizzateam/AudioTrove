"""
Signal noise ratio filtering.
"""

import numpy as np

from audiotrove.base import GPUFilter
from audiotrove.document import AudioDocument

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



class SNRFilter(GPUFilter):
    name = "snr_filter"

    def __init__(
        self,
        min_snr_db: float = 15.0,
        use_mos_scoring: bool = False,
        device: str = "cpu",
    ):
        self.min_snr_db = min_snr_db
        self.use_mos_scoring = use_mos_scoring
        self._device = _resolve_device(device)

    @property
    def device(self):
        """The torch.device this filter operates on (None without torch)."""
        return self._device

    def to(self, device) -> "SNRFilter":
        """Set the compute device and return self."""
        self._device = _resolve_device(device)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_device"] = str(self._device) if self._device is not None else "cpu"
        return state

    def __setstate__(self, state):
        device = state.pop("_device", "cpu")
        self.__dict__.update(state)
        self._device = _resolve_device(device)

    def filter(self, doc: AudioDocument) -> bool:
        gpu_tensor = getattr(doc, "gpu_tensor", None)
        if gpu_tensor is not None and HAS_TORCH and torch.is_tensor(gpu_tensor):
            snr_db = self._compute_snr_gpu(doc)
        else:
            snr_db = self._compute_snr(doc)
        doc.metadata["snr_db"] = round(snr_db, 2)
        return snr_db >= self.min_snr_db

    def _compute_snr_gpu(self, doc: AudioDocument) -> float:
        """Torch-based SNR on the gpu_tensor already present on ``doc``.

        Falls back to the CPU energy heuristic when VAD timestamps are missing.
        """
        audio_t = doc.gpu_tensor
        timestamps = doc.metadata.get("vad_speech_timestamps")
        if not timestamps:
            return self._energy_fallback_snr(doc.audio)

        n = audio_t.shape[0]
        device = audio_t.device
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for ts in timestamps:
            start = int(ts["start"])
            end = int(ts["end"])
            mask[start:end] = True

        speech = audio_t[mask]
        noise = audio_t[~mask]

        if noise.numel() < doc.sample_rate * 0.1:
            doc.metadata["snr_note"] = "insufficient_noise_floor"
            return 40.0

        speech = speech.float()
        noise = noise.float()
        signal_power = speech.pow(2).mean() if speech.numel() > 0 else torch.tensor(0.0)
        noise_power = noise.pow(2).mean() if noise.numel() > 0 else torch.tensor(0.0)
        if float(noise_power) == 0.0:
            return 40.0
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return float(snr.detach().cpu())

    def _compute_snr(self, doc: AudioDocument) -> float:
        timestamps = doc.metadata.get("vad_speech_timestamps")
        audio = doc.audio
        sr = doc.sample_rate

        if not timestamps:
            return self._energy_fallback_snr(audio)

        speech_mask = np.zeros(len(audio), dtype=bool)
        for ts in timestamps:
            start = int(ts["start"])
            end = int(ts["end"])
            speech_mask[start:end] = True

        speech_frames = audio[speech_mask]
        noise_frames = audio[~speech_mask]

        if len(noise_frames) < sr * 0.1:
            doc.metadata["snr_note"] = "insufficient_noise_floor"
            return 40.0

        signal_power = np.mean(speech_frames**2) if len(speech_frames) > 0 else 0.0
        noise_power = np.mean(noise_frames**2) if len(noise_frames) > 0 else 0.0

        if noise_power == 0.0:
            return 40.0

        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return float(snr_db)

    def _energy_fallback_snr(self, audio: np.ndarray) -> float:
        frame_size = 512
        if len(audio) < frame_size:
            return 40.0
        n = len(audio) - (len(audio) % frame_size)
        frames = audio[:n].reshape(-1, frame_size)
        frame_energies = np.mean(frames**2, axis=1)
        threshold = np.percentile(frame_energies, 75)
        signal_power = np.mean(frame_energies[frame_energies >= threshold])
        noise_power = np.mean(frame_energies[frame_energies < threshold])
        return float(10 * np.log10(signal_power / (noise_power + 1e-10)))
