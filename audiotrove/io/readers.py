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
    ):
        # Support both single pattern (str) and multiple patterns (list)
        if isinstance(path_pattern, str):
            self.path_patterns = [path_pattern]
        else:
            self.path_patterns = list(path_pattern)
        self.target_sr = target_sample_rate
        self.max_duration = max_duration_seconds
        self.min_duration = min_duration_seconds

    def __iter__(self) -> Iterator[Optional[AudioDocument]]:
        if fsspec is None:
            raise RuntimeError("fsspec is required for LocalAudioReader")

        # Collect all matching paths across every pattern, de-duplicated while
        # preserving discovery order. ``fs`` is the filesystem handle of the last
        # pattern; for local globbing (the common case) every pattern shares the
        # same handle, and the Python fallback below only needs it for opening.
        seen_paths = set()
        paths: list[str] = []
        fs = None
        for pattern in self.path_patterns:
            fs, path = fsspec.core.url_to_fs(pattern)
            for fpath in fs.glob(path):
                if fpath in seen_paths:
                    continue
                seen_paths.add(fpath)
                paths.append(fpath)

        if not paths:
            return

        # Prefer the Rust extension's batch decoder when it is installed: it
        # decodes and sinc-resamples off the GIL in parallel. When the extension
        # isn't available we transparently use the pure-Python per-file path.
        try:
            from audiotrove_core import decode_audio_batch
        except ImportError:
            decode_audio_batch = None

        if decode_audio_batch is not None:
            yield from self._iter_rust(paths, decode_audio_batch, fs)
        else:
            yield from self._iter_python(paths, fs)

    def _iter_rust(self, paths, decode_audio_batch, fs) -> Iterator[Optional[AudioDocument]]:
        """Decode a batch of local files through the Rust extension.

        Rust returns ``(samples, actual_sr, path)`` tuples with mono f32 already
        resampled to ``target_sr``. Files it could not read come back with an
        empty sample array and ``sr == 0``; we surface those as ``None`` so the
        executor's skip/stat accounting is identical to the Python path. If the
        batch call itself fails we degrade to the Python path rather than losing
        the whole batch.
        """
        try:
            results = decode_audio_batch(paths, int(self.target_sr))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Rust batch decode failed ({e}); using Python fallback.")
            yield from self._iter_python(paths, fs)
            return

        for samples, actual_sr, path in results:
            if actual_sr == 0 or samples is None or len(samples) == 0:
                logger.warning(f"Failed to decode {path} via Rust; skipping.")
                yield None
                continue
            audio = np.asarray(samples, dtype=np.float32)
            yield self._make_doc(audio, path)

    def _iter_python(self, paths, fs) -> Iterator[Optional[AudioDocument]]:
        """Per-file decode via torchaudio/soundfile (the original behaviour)."""
        for path in paths:
            try:
                yield self._load(path, fs)
            except Exception as e:  # noqa: BLE001
                # Log and continue — don't let one bad file stop the whole load
                logger.warning(f"Failed to load {path}: {e}")
                yield None

    def _make_doc(self, audio: np.ndarray, path: str) -> Optional[AudioDocument]:
        """Build an AudioDocument from already-decoded mono samples at target_sr.

        Applies the same min/max duration policy as ``_load`` so the Rust and
        Python paths produce identical documents.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)

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

    def _load(self, path: str, fs) -> Optional[AudioDocument]:
        if torchaudio is None:
            if _soundfile is None:
                raise RuntimeError("torchaudio or soundfile is required to load audio files")

            with fs.open(path, "rb") as f:
                audio, sr = _soundfile.read(f, dtype="float32")

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.target_sr:
                num_samples = int(len(audio) * float(self.target_sr) / float(sr))
                audio = np.interp(
                    np.linspace(0, len(audio), num_samples, endpoint=False),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

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

                num_samples = int(waveform.shape[-1] * float(self.target_sr) / float(sr))
                audio_np = waveform.squeeze(0).numpy()
                resampled = _np.interp(
                    _np.linspace(0, len(audio_np), num_samples, endpoint=False),
                    _np.arange(len(audio_np)),
                    audio_np,
                )
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
