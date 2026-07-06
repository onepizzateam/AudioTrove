"""
Audio readers.
"""
from typing import Iterator, Optional
import numpy as np

try:
    import fsspec
except Exception:  # pragma: no cover - fsspec optional in some envs
    fsspec = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


class LocalAudioReader:
    """Reads audio files from a local directory or glob pattern via fsspec.

    Returns `AudioDocument` objects. Resamples to `target_sample_rate` and
    downmixes to mono.
    """

    def __init__(self, path_pattern: str, target_sample_rate: int = 16000,
                 max_duration_seconds: Optional[float] = None,
                 min_duration_seconds: float = 0.5):
        self.path_pattern = path_pattern
        self.target_sr = target_sample_rate
        self.max_duration = max_duration_seconds
        self.min_duration = min_duration_seconds

    def __iter__(self) -> Iterator[Optional[AudioDocument]]:
        if fsspec is None:
            raise RuntimeError("fsspec is required for LocalAudioReader")

        fs, path = fsspec.core.url_to_fs(self.path_pattern)
        for fpath in fs.glob(path):
            try:
                doc = self._load(fpath, fs)
                yield doc
            except Exception:
                # Log and continue
                yield None

    def _load(self, path: str, fs) -> AudioDocument:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to load audio files")

        with fs.open(path, 'rb') as f:
            waveform, sr = torchaudio.load(f)

        # waveform: (channels, samples)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        audio = waveform.squeeze(0).numpy().astype(np.float32)
        duration = float(len(audio)) / float(self.target_sr)

        if duration < self.min_duration:
            return None

        if self.max_duration and duration > self.max_duration:
            audio = audio[: int(self.max_duration * self.target_sr)]
            duration = self.max_duration

        return AudioDocument(
            audio=audio,
            sample_rate=self.target_sr,
            source_path=path,
            duration_seconds=duration,
            doc_id=make_doc_id(path),
        )
