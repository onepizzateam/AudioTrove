"""Voice conversion inference sessions."""

from __future__ import annotations

import logging
from typing import Any

from audiotrove.inference.base import InferenceResult, InferenceSession

logger = logging.getLogger(__name__)


class SeedVCSession(InferenceSession):
    """SeedVC zero-shot voice conversion session."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the SeedVC model."""
        try:
            import seed_vc
        except ImportError as e:
            raise ImportError("SeedVC requires: pip install seed-vc") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = seed_vc.load_model(self.model_path, device=str(resolved))
        logger.info("Loaded SeedVC from %s", self.model_path)

    def run(
        self,
        source_audio_path: str,
        target_voice_path: str,
        **kwargs: Any,
    ) -> InferenceResult:
        """Convert source speech to the target voice."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        wav, sr = self._model.convert(source_audio_path, target_voice_path, **kwargs)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


class RVCSession(InferenceSession):
    """RVC (v2) voice conversion session."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the RVC model."""
        try:
            import rvc
        except ImportError as e:
            raise ImportError("RVC requires: pip install rvc") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = rvc.load_model(self.model_path, device=str(resolved))
        logger.info("Loaded RVC from %s", self.model_path)

    def run(
        self,
        source_audio_path: str,
        target_voice_path: str,
        **kwargs: Any,
    ) -> InferenceResult:
        """Convert source speech to the target voice."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        wav, sr = self._model.convert(source_audio_path, target_voice_path, **kwargs)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


def get_vc_session(family: str, **kwargs: Any) -> InferenceSession:
    """Factory function returning a voice-conversion inference session.

    Args:
        family: VC family name. One of: ``"seed_vc"``, ``"rvc"``.
        **kwargs: Parameters passed to the session constructor.

    Returns:
        An :class:`InferenceSession` for the requested family.

    Raises:
        ValueError: Unknown VC family.
    """
    registry = {
        "seed_vc": SeedVCSession,
        "rvc": RVCSession,
    }
    cls = registry.get(family.lower())
    if cls is None:
        raise ValueError(
            f"Unknown VC family: {family!r}. Choose from {sorted(registry.keys())}"
        )
    return cls(**kwargs)
