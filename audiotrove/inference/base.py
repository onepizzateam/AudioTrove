"""Inference session base classes and result container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class InferenceResult:
    """Result of a single inference request.

    Attributes:
        audio: Output waveform for TTS / VC / source separation, else ``None``.
        text: Transcript for ASR / alignment, else ``None``.
        sample_rate: Sample rate of ``audio`` when present (0 otherwise).
        metadata: Backend-specific extras (e.g. word timestamps, language).
    """

    audio: Optional[np.ndarray] = None
    text: Optional[str] = None
    sample_rate: int = 0
    metadata: dict = field(default_factory=dict)


class InferenceSession(ABC):
    """Base class for all AudioTrove inference sessions.

    Concrete sessions load a model lazily in :meth:`load`, execute one request
    per :meth:`run`, and release memory in :meth:`unload`. Sessions support the
    context-manager protocol so callers can write ``with session: ...``.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the device."""

    @abstractmethod
    def run(self, **kwargs: Any) -> InferenceResult:
        """Execute one inference request."""

    def unload(self) -> None:
        """Release model/GPU memory. Safe default is a no-op."""
        return None

    def __enter__(self) -> "InferenceSession":
        self.load()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.unload()
