"""
Voice activity detection.
"""

import logging
import numpy as np
import hashlib
import threading

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from audiotrove.base import AudioFilter, AudioFanOutTransformer
from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)
_SILERO_INFERENCE_LOCK = threading.Lock()


def _load_silero_model():
    from pathlib import Path

    local = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
    if local.exists():
        m = torch.hub.load(str(local), "silero_vad", source="local", trust_repo=True)
        # torch.hub.load may return (model, utils) where utils is a tuple
        # Convert tuple utils into a simple namespace with attributes for
        # compatibility with existing code that expects attribute access.
        model, utils = m
        if isinstance(utils, tuple):
            from types import SimpleNamespace

            ns = SimpleNamespace()
            for item in utils:
                name = getattr(item, "__name__", None)
                if name:
                    setattr(ns, name, item)
            return model, ns
        return model, utils
    return torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True)


class SileroVADFilter(AudioFilter):
    name = "silero_vad"

    def __init__(
        self, min_speech_ratio: float = 0.3, threshold: float = 0.5, window_size_samples: int = 512
    ):
        self.min_speech_ratio = min_speech_ratio
        self.threshold = threshold
        self.window_size = window_size_samples
        self._model = None
        self._utils = None

    def __getstate__(self):
        """Exclude the lazy-loaded torch model from pickling.

        When this filter is pickled for multiprocessing (ProcessPoolExecutor),
        we exclude the torch model so it doesn't get serialized. The model will
        be lazily reloaded in the worker process when needed.
        """
        state = self.__dict__.copy()
        # Remove unpicklable torch model and utils
        state["_model"] = None
        state["_utils"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # _model and _utils will be lazily loaded on first access

    @property
    def model(self):
        if self._model is None:
            if not HAS_TORCH:
                return None
            # Lazy-load silero vad model
            try:
                self._model, utils = _load_silero_model()
                # utils may contain get_speech_timestamps
                self._utils = utils
            except Exception as e:  # noqa: BLE001
                # If model cannot be loaded, log the failure and fallback
                logger.warning(
                    f"Failed to load Silero VAD model: {e}. Falling back to energy-based VAD."
                )
                self._model = None
        return self._model

    def _energy_vad(self, audio: np.ndarray, sr: int):
        # Improved energy-based VAD fallback that better detects speech
        frame = self.window_size
        energies = np.array(
            [np.mean(audio[i : i + frame] ** 2) for i in range(0, len(audio), frame)]
        )

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
        for idx, flag in enumerate(speech_frames):
            start = idx * frame
            end = min(len(audio), (idx + 1) * frame)
            if flag:
                timestamps.append({"start": start, "end": end})
        return timestamps

    def filter(self, doc: AudioDocument) -> bool:
        audio = doc.audio
        sr = doc.sample_rate

        timestamps = None
        backend_used = None
        model = self.model
        if model is not None and hasattr(self, "_utils") and HAS_TORCH:
            try:
                get_speech_timestamps = getattr(self._utils, "get_speech_timestamps", None)
                if get_speech_timestamps is not None:
                    # silero expects torch.Tensor
                    audio_t = torch.from_numpy(audio)
                    with _SILERO_INFERENCE_LOCK:
                        timestamps = get_speech_timestamps(
                            audio_t,
                            model,
                            sampling_rate=sr,
                            threshold=self.threshold,
                            window_size_samples=self.window_size,
                        )
                    # convert to simple list of dicts
                    ts = []
                    for t in timestamps:
                        ts.append({"start": int(t["start"]), "end": int(t["end"])})
                    # If Silero returned no timestamps (empty list), treat as no result
                    # so we fall back to the energy-based VAD which handles synthetic/test signals.
                    if not ts:
                        timestamps = None
                    else:
                        timestamps = ts
                        backend_used = "silero"
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Silero VAD inference failed for {doc.source_path}: {e}. Falling back to energy-based VAD."
                )
                timestamps = None

        if timestamps is None:
            timestamps = self._energy_vad(audio, sr)
            backend_used = "energy_fallback"

        if not timestamps:
            doc.metadata["vad_speech_ratio"] = 0.0
            doc.metadata["vad_speech_timestamps"] = []
            doc.metadata["vad_backend"] = backend_used or "energy_fallback"
            return False

        speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        ratio = float(speech_samples) / float(len(audio))
        doc.metadata["vad_speech_ratio"] = round(ratio, 4)
        doc.metadata["vad_speech_timestamps"] = timestamps
        doc.metadata["vad_backend"] = backend_used or "energy_fallback"
        return bool(ratio >= self.min_speech_ratio)


class VADSegmenter(AudioFanOutTransformer):
    """Fan-out transformer that segments audio into speech regions.

    Instead of keeping/discarding an entire file based on aggregate speech ratio,
    VADSegmenter splits each file into per-speech-segment sub-documents. This way,
    one bad segment doesn't sink an otherwise-good file, and downstream filtering
    works on shorter, more homogeneous clips.

    Each segment becomes its own AudioDocument with:
    - Mono audio containing only the speech region
    - duration_seconds updated to match segment length
    - doc_id deterministically derived from parent doc_id + segment start/end
    - metadata carrying provenance (parent_doc_id, segment_start, segment_end, vad_backend)
    """

    name = "vad_segmenter"

    def __init__(self, threshold: float = 0.5, window_size_samples: int = 512):
        """Initialize VAD segmenter.

        Args:
            threshold: Silero VAD threshold (0-1). Default 0.5.
            window_size_samples: Window size for energy fallback. Default 512.
        """
        self.threshold = threshold
        self.window_size = window_size_samples
        self._model = None
        self._utils = None

    def __getstate__(self):
        """Exclude the lazy-loaded torch model from pickling."""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_utils"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)

    @property
    def model(self):
        if self._model is None:
            if not HAS_TORCH:
                return None
            try:
                self._model, utils = _load_silero_model()
                self._utils = utils
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Failed to load Silero VAD model: {e}. Falling back to energy-based VAD."
                )
                self._model = None
        return self._model

    def _energy_vad(self, audio: np.ndarray, sr: int):
        """Energy-based VAD fallback."""
        frame = self.window_size
        energies = np.array(
            [np.mean(audio[i : i + frame] ** 2) for i in range(0, len(audio), frame)]
        )

        mean_energy = np.mean(energies)
        std_energy = np.std(energies)

        if mean_energy < 1e-4:
            return []

        thresh = mean_energy + std_energy
        speech_frames = energies > thresh

        if np.sum(speech_frames) / len(speech_frames) < 0.1:
            return []

        timestamps = []
        for idx, flag in enumerate(speech_frames):
            if flag:
                start = idx * frame
                end = min(len(audio), (idx + 1) * frame)
                timestamps.append({"start": start, "end": end})
        return timestamps

    def _get_speech_timestamps(self, audio: np.ndarray, sr: int):
        """Get speech timestamps from VAD."""
        timestamps = None
        backend_used = "energy_fallback"

        model = self.model
        if model is not None and hasattr(self, "_utils") and HAS_TORCH:
            try:
                get_speech_timestamps = getattr(self._utils, "get_speech_timestamps", None)
                if get_speech_timestamps is not None:
                    audio_t = torch.from_numpy(audio)
                    with _SILERO_INFERENCE_LOCK:
                        timestamps = get_speech_timestamps(
                            audio_t,
                            model,
                            sampling_rate=sr,
                            threshold=self.threshold,
                            window_size_samples=self.window_size,
                        )
                    ts = []
                    for t in timestamps:
                        ts.append({"start": int(t["start"]), "end": int(t["end"])})
                    # If Silero returned no timestamps, fall back to energy VAD
                    if not ts:
                        timestamps = None
                    else:
                        timestamps = ts
                        backend_used = "silero"
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Silero VAD failed: {e}. Using energy fallback.")
                timestamps = None

        if timestamps is None:
            timestamps = self._energy_vad(audio, sr)
            backend_used = "energy_fallback"

        return timestamps, backend_used

    def _make_segment_doc_id(self, parent_doc_id: str, start: int, end: int) -> str:
        """Generate a deterministic doc_id for a segment.

        Derived from parent doc_id + start/end positions to ensure:
        - Re-running produces same segment doc_ids (idempotent)
        - Checkpoint skip still works
        """
        segment_key = f"{parent_doc_id}:{start}:{end}"
        return hashlib.sha256(segment_key.encode()).hexdigest()[:16]

    def transform(self, doc: AudioDocument) -> list[AudioDocument]:
        """Segment audio into speech regions.

        Args:
            doc: Input AudioDocument.

        Returns:
            List of AudioDocument objects (one per speech segment).
            Returns empty list if no speech detected.
        """
        audio = doc.audio
        sr = doc.sample_rate

        timestamps, backend_used = self._get_speech_timestamps(audio, sr)

        logger.debug(
            "VADSegmenter produced %d timestamp(s) for %s via %s",
            len(timestamps),
            doc.source_path,
            backend_used,
        )
        if not timestamps:
            # No speech detected, return empty list
            return []

        segments = []
        for ts in timestamps:
            start = ts["start"]
            end = ts["end"]

            # Extract segment audio
            segment_audio = audio[start:end].astype(np.float32)
            segment_duration = float(len(segment_audio)) / sr

            # Generate deterministic segment doc_id
            segment_doc_id = self._make_segment_doc_id(doc.doc_id, start, end)

            # Create segment document
            segment_doc = AudioDocument(
                audio=segment_audio,
                sample_rate=sr,
                source_path=doc.source_path,  # Preserve original source path
                duration_seconds=segment_duration,
                doc_id=segment_doc_id,
                metadata={
                    "parent_doc_id": doc.doc_id,
                    "segment_start_sample": start,
                    "segment_end_sample": end,
                    "segment_start_seconds": start / sr,
                    "segment_end_seconds": end / sr,
                    "vad_backend": backend_used,
                    **doc.metadata,  # Include parent metadata
                },
            )
            segments.append(segment_doc)

        return segments
