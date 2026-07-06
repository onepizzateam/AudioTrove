"""
Signal noise ratio filtering.
"""
import numpy as np

from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument


class SNRFilter(AudioFilter):
    name = "snr_filter"

    def __init__(self, min_snr_db: float = 15.0, use_mos_scoring: bool = False):
        self.min_snr_db = min_snr_db
        self.use_mos_scoring = use_mos_scoring

    def filter(self, doc: AudioDocument) -> bool:
        snr_db = self._compute_snr(doc)
        doc.metadata['snr_db'] = round(snr_db, 2)
        return snr_db >= self.min_snr_db

    def _compute_snr(self, doc: AudioDocument) -> float:
        timestamps = doc.metadata.get('vad_speech_timestamps')
        audio = doc.audio
        sr = doc.sample_rate

        if not timestamps:
            return self._energy_fallback_snr(audio)

        speech_mask = np.zeros(len(audio), dtype=bool)
        for ts in timestamps:
            start = int(ts['start'])
            end = int(ts['end'])
            speech_mask[start:end] = True

        speech_frames = audio[speech_mask]
        noise_frames = audio[~speech_mask]

        if len(noise_frames) < sr * 0.1:
            doc.metadata['snr_note'] = 'insufficient_noise_floor'
            return 40.0

        signal_power = np.mean(speech_frames ** 2) if len(speech_frames) > 0 else 0.0
        noise_power = np.mean(noise_frames ** 2) if len(noise_frames) > 0 else 0.0

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
        frame_energies = np.mean(frames ** 2, axis=1)
        threshold = np.percentile(frame_energies, 75)
        signal_power = np.mean(frame_energies[frame_energies >= threshold])
        noise_power = np.mean(frame_energies[frame_energies < threshold])
        return float(10 * np.log10(signal_power / (noise_power + 1e-10)))
