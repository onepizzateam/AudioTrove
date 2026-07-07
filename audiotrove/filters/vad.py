"""
Voice activity detection.
"""
import logging
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)


class SileroVADFilter(AudioFilter):
    name = "silero_vad"

    def __init__(self, min_speech_ratio: float = 0.3, threshold: float = 0.5,
                 window_size_samples: int = 512):
        self.min_speech_ratio = min_speech_ratio
        self.threshold = threshold
        self.window_size = window_size_samples
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not HAS_TORCH:
                return None
            # Lazy-load silero vad model
            try:
                self._model, utils = torch.hub.load(
                    'snakers4/silero-vad', 'silero_vad', force_reload=False
                )
                # utils may contain get_speech_timestamps
                self._utils = utils
            except Exception as e:
                # If model cannot be loaded, log the failure and fallback
                logger.warning(f"Failed to load Silero VAD model: {e}. Falling back to energy-based VAD.")
                self._model = None
        return self._model

    def _energy_vad(self, audio: np.ndarray, sr: int):
        # Improved energy-based VAD fallback that better detects speech
        frame = self.window_size
        energies = np.array([np.mean(audio[i:i+frame] ** 2)
                             for i in range(0, len(audio), frame)])
        
        # Get statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # If energy is very low everywhere, it's silence
        if mean_energy < 1e-4:
            return []
        
        # Use a more robust threshold: mean + std (roughly 68th percentile)
        thresh = mean_energy + std_energy
        
        speech_frames = energies > thresh
        
        # If less than 10% of frames above threshold, probably no speech
        if np.sum(speech_frames) / len(speech_frames) < 0.1:
            return []
        
        timestamps = []
        idx = 0
        for flag in speech_frames:
            start = idx * frame
            end = min(len(audio), (idx + 1) * frame)
            if flag:
                timestamps.append({'start': start, 'end': end})
            idx += 1
        return timestamps

    def filter(self, doc: AudioDocument) -> bool:
        audio = doc.audio
        sr = doc.sample_rate

        timestamps = None
        backend_used = None
        model = self.model
        if model is not None and hasattr(self, '_utils') and HAS_TORCH:
            try:
                get_speech_timestamps = getattr(self._utils, 'get_speech_timestamps', None)
                if get_speech_timestamps is not None:
                    # silero expects torch.Tensor
                    audio_t = torch.from_numpy(audio)
                    timestamps = get_speech_timestamps(
                        audio_t, model,
                        sampling_rate=sr,
                        threshold=self.threshold,
                        window_size_samples=self.window_size
                    )
                    # convert to simple list of dicts
                    ts = []
                    for t in timestamps:
                        ts.append({'start': int(t['start']), 'end': int(t['end'])})
                    timestamps = ts
                    backend_used = 'silero'
            except Exception as e:
                logger.warning(f"Silero VAD inference failed for {doc.source_path}: {e}. Falling back to energy-based VAD.")
                timestamps = None

        if timestamps is None:
            timestamps = self._energy_vad(audio, sr)
            backend_used = 'energy_fallback'

        if not timestamps:
            doc.metadata['vad_speech_ratio'] = 0.0
            doc.metadata['vad_speech_timestamps'] = []
            doc.metadata['vad_backend'] = backend_used or 'energy_fallback'
            return False

        speech_samples = sum(t['end'] - t['start'] for t in timestamps)
        ratio = float(speech_samples) / float(len(audio))
        doc.metadata['vad_speech_ratio'] = round(ratio, 4)
        doc.metadata['vad_speech_timestamps'] = timestamps
        doc.metadata['vad_backend'] = backend_used or 'energy_fallback'
        return bool(ratio >= self.min_speech_ratio)
