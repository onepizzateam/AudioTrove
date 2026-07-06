"""
Voice activity detection.
"""
import torch
import numpy as np
from typing import Optional

from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument


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
            # Lazy-load silero vad model
            try:
                self._model, utils = torch.hub.load(
                    'snakers4/silero-vad', 'silero_vad', force_reload=False
                )
                # utils may contain get_speech_timestamps
                self._utils = utils
            except Exception:
                # If model cannot be loaded, leave as None and fallback
                self._model = None
        return self._model

    def _energy_vad(self, audio: np.ndarray, sr: int):
        # Simple energy-based VAD fallback
        frame = self.window_size
        energies = np.array([np.mean(audio[i:i+frame] ** 2)
                             for i in range(0, len(audio), frame)])
        thresh = np.percentile(energies, 75) * 0.5
        speech_frames = energies > thresh
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
        model = self.model
        if model is not None and hasattr(self, '_utils'):
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
            except Exception:
                timestamps = None

        if timestamps is None:
            timestamps = self._energy_vad(audio, sr)

        if not timestamps:
            doc.metadata['vad_speech_ratio'] = 0.0
            doc.metadata['vad_speech_timestamps'] = []
            return False

        speech_samples = sum(t['end'] - t['start'] for t in timestamps)
        ratio = speech_samples / len(audio)
        doc.metadata['vad_speech_ratio'] = round(ratio, 4)
        doc.metadata['vad_speech_timestamps'] = timestamps
        return ratio >= self.min_speech_ratio
