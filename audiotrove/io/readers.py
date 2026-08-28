"""
Audio readers.
"""

import logging
from typing import Iterator, Optional, Union, List
import numpy as np

try:
    import fsspec
except Exception:  # noqa: BLE001  # pragma: no cover - fsspec optional in some envs
    fsspec = None

try:
    import torchaudio
except Exception:  # noqa: BLE001  # pragma: no cover
    torchaudio = None
import io as _io

try:
    import soundfile as _soundfile
except Exception:  # noqa: BLE001  # pragma: no cover - optional, we'll raise if needed at runtime
    _soundfile = None

from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id
from audiotrove.io.rust_backend import decode_wav as _rust_decode_wav
from audiotrove.io.rust_backend import glob_paths as _rust_glob_paths
from audiotrove.io.rust_backend import resample as _resample

logger = logging.getLogger(__name__)


class LocalAudioReader:
    """Reads audio files from local directories or glob patterns via fsspec.

    Returns `AudioDocument` objects. Resamples to `target_sample_rate` and
    downmixes to mono. Supports multiple glob patterns for multi-format support.
    """

    def __init__(
        self,
        path_pattern: Union[str, List[str]],
        target_sample_rate: int = 16000,
        max_duration_seconds: Optional[float] = None,
        min_duration_seconds: float = 0.5,
        max_ram_per_worker: Optional[float] = None,
    ):
        # Support both single pattern (str) and multiple patterns (list)
        if isinstance(path_pattern, str):
            self.path_patterns = [path_pattern]
        else:
            self.path_patterns = list(path_pattern)
        self.target_sr = target_sample_rate
        self.max_duration = max_duration_seconds
        self.min_duration = min_duration_seconds
        self.max_ram_per_worker = max_ram_per_worker

    def __iter__(self) -> Iterator[Optional[AudioDocument]]:
        if fsspec is None:
            raise RuntimeError("fsspec is required for LocalAudioReader")

        seen_paths = set()  # Track paths we've already yielded to avoid duplicates

        for pattern in self.path_patterns:
            fs, path = fsspec.core.url_to_fs(pattern)
            protocol = getattr(fs, "protocol", "file")
            protocols = protocol if isinstance(protocol, (tuple, list)) else (protocol,)
            is_local = any(value in {"file", "local"} for value in protocols)
            paths = (_rust_glob_paths(path) if is_local else fs.glob(path))
            for fpath in paths:
                # Skip if we've already processed this path
                if fpath in seen_paths:
                    continue
                seen_paths.add(fpath)

                try:
                    doc = self._load(fpath, fs)
                    yield doc
                except Exception as e:  # noqa: BLE001
                    # Log and continue — don't let one bad file stop the whole load
                    logger.warning(f"Failed to load {fpath}: {e}")
                    yield None

    def _load(self, path: str, fs) -> Optional[AudioDocument]:
        if self.max_ram_per_worker is not None:
            try:
                size = fs.info(path).get("size")
                ceiling = self.max_ram_per_worker * (1024 ** 3)
                if size and size > ceiling:
                    logger.warning(
                        "Streaming decode input %s is %.2f GiB, above the %.2f GiB/worker ceiling",
                        path, size / (1024 ** 3), self.max_ram_per_worker,
                    )
            except (AttributeError, OSError, TypeError, ValueError):
                logger.debug("Could not inspect size for %s", path, exc_info=True)
        protocol = getattr(fs, "protocol", "file")
        protocols = protocol if isinstance(protocol, (tuple, list)) else (protocol,)
        is_local = any(value in {"file", "local"} for value in protocols)
        if is_local and path.lower().endswith(".wav"):
            rust_audio = _rust_decode_wav(path)
            if rust_audio is not None:
                audio, sr = rust_audio
                if sr != self.target_sr:
                    audio = _resample(audio, sr, self.target_sr)
                duration = float(len(audio)) / float(self.target_sr)
                if duration < self.min_duration:
                    return None
                if self.max_duration and duration > self.max_duration:
                    audio = audio[: int(self.max_duration * self.target_sr)]
                    duration = self.max_duration
                return AudioDocument(audio=audio, sample_rate=self.target_sr,
                                     source_path=path, duration_seconds=duration,
                                     doc_id=make_doc_id(path))
        if torchaudio is None:
            if _soundfile is None:
                raise RuntimeError("torchaudio or soundfile is required to load audio files")

            with fs.open(path, "rb") as f:
                audio, sr = _soundfile.read(f, dtype="float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.target_sr:
                audio = _resample(audio, sr, self.target_sr)

            duration = float(len(audio)) / float(self.target_sr)
            if duration < self.min_duration:
                return None
            if self.max_duration and duration > self.max_duration:
                audio = audio[: int(self.max_duration * self.target_sr)]
                duration = self.max_duration

            return AudioDocument(
                audio=np.asarray(audio, dtype=np.float32),
                sample_rate=self.target_sr,
                source_path=path,
                duration_seconds=duration,
                doc_id=make_doc_id(path),
            )

        # Prefer torchaudio.load, but fall back to soundfile if torchaudio's
        # torchcodec backend fails (common when torchcodec/ffmpeg libs missing).
        waveform = None
        sr = None
        with fs.open(path, "rb") as f:
            try:
                waveform, sr = torchaudio.load(f)
            except Exception:  # noqa: BLE001
                # Attempt to use soundfile fallback
                if _soundfile is None:
                    raise

                f.seek(0)
                data = f.read()
                bio = _io.BytesIO(data)
                audio_sf, sr = _soundfile.read(bio, dtype="float32")
                # audio_sf shape: (frames,) or (frames, channels)
                import torch
                import numpy as _np

                if audio_sf.ndim > 1:
                    audio_mono = audio_sf.mean(axis=1)
                else:
                    audio_mono = audio_sf

                waveform = torch.from_numpy(_np.asarray(audio_mono, dtype=_np.float32)).unsqueeze(0)

        # waveform: (channels, samples)
        # waveform: (channels, samples)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            try:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            except Exception:  # noqa: BLE001
                # Fallback to simple numpy resampling if torchaudio resampler fails
                import numpy as _np

                audio_np = waveform.squeeze(0).numpy()
                resampled = _resample(audio_np, sr, self.target_sr)
                import torch

                waveform = torch.from_numpy(resampled.astype(_np.float32)).unsqueeze(0)

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
