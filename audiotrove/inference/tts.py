"""TTS inference sessions."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from audiotrove.inference.base import InferenceResult, InferenceSession

logger = logging.getLogger(__name__)


class F5TTSSession(InferenceSession):
    """F5-TTS voice cloning inference session.

    Requires: pip install audiotrove[infer]
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        voice_ref: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.voice_ref = voice_ref
        self._model = None

    def load(self) -> None:
        """Load F5-TTS model."""
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise ImportError(
                "F5-TTS requires: pip install audiotrove[infer] or pip install f5-tts"
            ) from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = F5TTS(
            model_type="F5-TTS",
            ckpt_file=self.model_path,
            device=str(resolved),
        )
        logger.info("Loaded F5-TTS model from %s on %s", self.model_path, resolved)

    def run(
        self,
        text: str,
        voice_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Synthesize speech from text using the loaded voice.

        Args:
            text: Text to synthesize.
            voice_ref: Path to reference voice WAV (overrides session default).
            **kwargs: Additional F5-TTS parameters.

        Returns:
            InferenceResult with synthesized audio.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first or use context manager.")

        ref = voice_ref or self.voice_ref
        if not ref:
            raise ValueError("voice_ref required (session default or run() parameter)")

        wav, sr, _ = self._model.infer(
            ref_file=ref,
            ref_text="",  # F5-TTS auto-transcribes ref
            gen_text=text,
            **kwargs,
        )
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        """Release model memory."""
        self._model = None


class StyleTTS2Session(InferenceSession):
    """StyleTTS2 zero-shot voice cloning inference session."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        voice_ref: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.voice_ref = voice_ref
        self._model = None

    def load(self) -> None:
        """Load StyleTTS2 model."""
        try:
            from styletts2 import tts_inference
        except ImportError as e:
            raise ImportError("StyleTTS2 requires: pip install styletts2") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = tts_inference.load_model(self.model_path, device=str(resolved))
        logger.info("Loaded StyleTTS2 from %s on %s", self.model_path, resolved)

    def run(
        self,
        text: str,
        voice_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Synthesize speech from text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        ref = voice_ref or self.voice_ref
        if not ref:
            raise ValueError("voice_ref required")

        wav = self._model.inference(text, ref, **kwargs)
        sr = getattr(self._model, "sampling_rate", 22050)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


class PiperSession(InferenceSession):
    """Piper TTS inference session (fast CPU/GPU synthesis)."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load Piper model."""
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ImportError("Piper requires: pip install piper-tts") from e

        self._model = PiperVoice.load(self.model_path)
        logger.info("Loaded Piper model from %s", self.model_path)

    def run(self, text: str, **kwargs: Any) -> InferenceResult:
        """Synthesize speech from text."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        wav_bytes = self._model.synthesize(text, **kwargs)
        # Piper returns bytes; convert to float32 numpy
        wav = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        sr = self._model.config.sample_rate
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


class ChatterboxSession(InferenceSession):
    """Chatterbox multilingual TTS session."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        voice_ref: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.voice_ref = voice_ref
        self._model = None

    def load(self) -> None:
        """Load Chatterbox model."""
        try:
            import chatterbox
        except ImportError as e:
            raise ImportError("Chatterbox requires: pip install chatterbox-tts") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = chatterbox.load_model(self.model_path, device=str(resolved))
        logger.info("Loaded Chatterbox from %s", self.model_path)

    def run(
        self,
        text: str,
        voice_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Synthesize speech."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        ref = voice_ref or self.voice_ref
        if not ref:
            raise ValueError("voice_ref required")

        wav, sr = self._model.synthesize(text, ref, **kwargs)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


class MatchaTTSSession(InferenceSession):
    """Matcha-TTS flow-matching TTS session."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load Matcha-TTS model."""
        try:
            from matcha_tts import load_model
        except ImportError as e:
            raise ImportError("Matcha-TTS requires: pip install matcha-tts") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._model = load_model(self.model_path, device=str(resolved))
        logger.info("Loaded Matcha-TTS from %s", self.model_path)

    def run(self, text: str, **kwargs: Any) -> InferenceResult:
        """Synthesize speech from text."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        wav, sr = self._model.synthesize(text, **kwargs)
        return InferenceResult(audio=wav, text=None, sample_rate=sr, metadata={})

    def unload(self) -> None:
        self._model = None


def get_tts_session(family: str, **kwargs: Any) -> InferenceSession:
    """Factory function returning a TTS inference session.

    Args:
        family: TTS family name. One of: ``"f5tts"``, ``"styletts2"``,
            ``"piper"``, ``"chatterbox"``, ``"matcha"``.
        **kwargs: Parameters passed to the session constructor.

    Returns:
        An :class:`InferenceSession` instance for the requested family.

    Raises:
        ValueError: Unknown TTS family.
    """
    registry = {
        "f5tts": F5TTSSession,
        "styletts2": StyleTTS2Session,
        "piper": PiperSession,
        "chatterbox": ChatterboxSession,
        "matcha": MatchaTTSSession,
    }
    cls = registry.get(family.lower())
    if cls is None:
        raise ValueError(
            f"Unknown TTS family: {family!r}. Choose from {sorted(registry.keys())}"
        )
    return cls(**kwargs)
