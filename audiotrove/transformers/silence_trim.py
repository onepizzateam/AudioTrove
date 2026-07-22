"""Silence trimming transformation."""

import numpy as np

from audiotrove.base import AudioTransformer
from audiotrove.document import AudioDocument
from audiotrove.filters.vad import SileroVADFilter


class SilenceTrimmingTransformer(AudioTransformer):
    """Trim leading and trailing silence around VAD speech timestamps."""

    name = "silence_trim"

    def __init__(self, padding_ms: int = 150, min_duration_seconds: float = 1.0):
        """Initialize the silence trimming transformer.

        Args:
            padding_ms: Silence to preserve on each side of detected speech.
            min_duration_seconds: Minimum duration allowed after trimming.
        """
        self.padding_ms = padding_ms
        self.min_duration_seconds = min_duration_seconds
        self._vad_filter = None

    def __getstate__(self):
        """Exclude lazy-loaded VAD model state from pickling."""
        state = self.__dict__.copy()
        state["_vad_filter"] = None
        return state

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
        start = max(0, min(int(ts["start"]) for ts in timestamps) - padding_samples)
        end = min(len(doc.audio), max(int(ts["end"]) for ts in timestamps) + padding_samples)
        trimmed_duration = float(end - start) / doc.sample_rate

        if trimmed_duration < self.min_duration_seconds:
            doc.metadata["trimmed_duration_seconds"] = doc.duration_seconds
            return doc

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
