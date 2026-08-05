"""Standalone VAD inference session.

Wraps the Silero VAD model as a reusable inference session so callers can detect
speech regions in arbitrary audio files without building a full curation
pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from audiotrove.inference.base import InferenceResult, InferenceSession

logger = logging.getLogger(__name__)


class SileroVADInferenceSession(InferenceSession):
    """Standalone Silero VAD session returning speech timestamps."""

    def __init__(self, device: str = "auto", threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        self._filter = None

    def load(self) -> None:
        """Instantiate the underlying SileroVADFilter and force model load."""
        from audiotrove.filters.vad import SileroVADFilter

        self._filter = SileroVADFilter(threshold=self.threshold, device=self.device)
        # Force lazy model load so the first run() is warm.
        _ = self._filter.model

    def run(self, audio_path: str, **kwargs: Any) -> InferenceResult:
        """Detect speech regions in an audio file.

        Returns:
            InferenceResult with ``metadata['speech_timestamps']`` populated and
            ``metadata['speech_ratio']`` describing the speech fraction.
        """
        if self._filter is None:
            raise RuntimeError("Session not loaded. Call load() first.")

        from audiotrove.io.readers import LocalAudioReader

        reader = LocalAudioReader(
            [audio_path], min_duration_seconds=0.0, max_duration_seconds=None
        )
        doc = next(iter(reader), None)
        if doc is None:
            raise ValueError(f"Could not read audio: {audio_path}")

        self._filter.filter(doc)
        timestamps = doc.metadata.get("vad_speech_timestamps", [])
        return InferenceResult(
            audio=None,
            text=None,
            sample_rate=doc.sample_rate,
            metadata={
                "speech_timestamps": timestamps,
                "speech_ratio": doc.metadata.get("speech_ratio"),
            },
        )

    def unload(self) -> None:
        self._filter = None


def get_vad_session(**kwargs: Any) -> InferenceSession:
    """Return a standalone VAD inference session."""
    return SileroVADInferenceSession(**kwargs)
