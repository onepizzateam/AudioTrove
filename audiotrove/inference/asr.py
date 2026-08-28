"""ASR inference sessions."""

from __future__ import annotations

import logging
from typing import Any

from audiotrove.inference.base import InferenceResult, InferenceSession

logger = logging.getLogger(__name__)


class FasterWhisperSession(InferenceSession):
    """faster-whisper (CTranslate2) ASR session — CUDA/CPU."""

    def __init__(
        self,
        model_path: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._device_str = "cpu"

    def load(self) -> None:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper requires: pip install audiotrove[transcribe-gpu]"
            ) from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        # faster-whisper accepts "cuda"/"cpu"; map mps->cpu (unsupported by CT2)
        self._device_str = resolved.type if resolved.type in ("cuda", "cpu") else "cpu"

        if self.compute_type == "auto":
            ct = "float16" if self._device_str == "cuda" else "int8"
        else:
            ct = self.compute_type

        self._model = WhisperModel(
            self.model_path, device=self._device_str, compute_type=ct
        )
        logger.info("Loaded faster-whisper %s on %s", self.model_path, self._device_str)

    def run(self, audio_path: str, **kwargs: Any) -> InferenceResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file to transcribe.
            **kwargs: Extra options forwarded to ``transcribe`` (e.g. beam_size).

        Returns:
            InferenceResult with ``text`` set to the transcript and word-level
            timestamps (when available) in ``metadata``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        beam_size = kwargs.pop("beam_size", 5)
        segments, info = self._model.transcribe(audio_path, beam_size=beam_size, **kwargs)
        seg_list = list(segments)
        text = " ".join(s.text for s in seg_list).strip()
        metadata = {
            "language": getattr(info, "language", None),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in seg_list
            ],
        }
        return InferenceResult(audio=None, text=text, sample_rate=0, metadata=metadata)

    def unload(self) -> None:
        self._model = None


class Qwen3ASRSession(InferenceSession):
    """Qwen3-ASR multilingual ASR session (transformers-based)."""

    def __init__(self, model_path: str = "Qwen/Qwen3-ASR", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the Qwen3-ASR model and processor."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError("Qwen3-ASR requires: pip install transformers") from e

        from audiotrove.gpu.device import get_device

        resolved = get_device(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_path).to(
            str(resolved)
        )
        logger.info("Loaded Qwen3-ASR %s on %s", self.model_path, resolved)

    def run(self, audio_path: str, **kwargs: Any) -> InferenceResult:
        """Transcribe an audio file."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        import soundfile as sf

        audio, sr = sf.read(audio_path)
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        generated = self._model.generate(**inputs, **kwargs)
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return InferenceResult(audio=None, text=text, sample_rate=0, metadata={})

    def unload(self) -> None:
        self._model = None
        self._processor = None


class ParakeetSession(InferenceSession):
    """NVIDIA Parakeet-TDT ASR session (NeMo-based, CUDA optimised)."""

    def __init__(
        self,
        model_path: str = "nvidia/parakeet-tdt-1.1b",
        device: str = "auto",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the Parakeet model."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError("Parakeet requires: pip install nemo_toolkit[asr]") from e

        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_path)
        logger.info("Loaded Parakeet %s", self.model_path)

    def run(self, audio_path: str, **kwargs: Any) -> InferenceResult:
        """Transcribe an audio file."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        transcripts = self._model.transcribe([audio_path], **kwargs)
        text = transcripts[0] if transcripts else ""
        if isinstance(text, (list, tuple)):
            text = text[0]
        return InferenceResult(audio=None, text=str(text).strip(), sample_rate=0, metadata={})

    def unload(self) -> None:
        self._model = None


def get_asr_session(family: str = "faster_whisper", **kwargs: Any) -> InferenceSession:
    """Factory function returning an ASR inference session.

    Args:
        family: ASR family name. One of: ``"faster_whisper"``, ``"qwen3"``,
            ``"parakeet"``.
        **kwargs: Parameters passed to the session constructor.

    Returns:
        An :class:`InferenceSession` for the requested family.

    Raises:
        ValueError: Unknown ASR family.
    """
    registry = {
        "faster_whisper": FasterWhisperSession,
        "qwen3": Qwen3ASRSession,
        "parakeet": ParakeetSession,
    }
    cls = registry.get(family.lower())
    if cls is None:
        raise ValueError(
            f"Unknown ASR family: {family!r}. Choose from {sorted(registry.keys())}"
        )
    return cls(**kwargs)
