"""Optional Whisper transcription transformer."""

from __future__ import annotations

import logging

from audiotrove.base import GPUTransformer
from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)


class WhisperTranscriber(GPUTransformer):
    """
    Transcribes each AudioDocument using faster-whisper (GPU) or openai-whisper (CPU).

    When faster-whisper is installed it is used for GPU-accelerated transcription with
    CTranslate2. Falls back to openai-whisper on CPU when faster-whisper is unavailable.

    Requires: pip install audiotrove[transcribe-gpu] for GPU path
              pip install audiotrove[transcribe] for CPU path

    Writes doc.metadata["transcription"] with the detected text.
    Does not filter — always returns the document.
    """

    name = "whisper_transcriber"

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "auto",
    ) -> None:
        self.model_name = model_name
        self._device_str = device
        self._compute_type = compute_type
        self._model = None
        self._backend = None  # "faster_whisper" | "openai_whisper"

    @property
    def device(self):
        """The device this transcriber targets (string, not torch.device)."""
        return self._device_str

    def to(self, device) -> "WhisperTranscriber":
        """Update the target device and reset the model for lazy reload."""
        if hasattr(device, "type"):
            # torch.device passed
            self._device_str = device.type
        else:
            self._device_str = str(device)
        self._model = None  # Force reload on next access
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def model(self):
        if self._model is None:
            # Try faster-whisper first (GPU-capable via CTranslate2)
            try:
                from faster_whisper import WhisperModel

                # Resolve compute_type based on device
                if self._compute_type == "auto":
                    if self._device_str == "cuda":
                        ct = "float16"
                    else:
                        ct = "int8"
                else:
                    ct = self._compute_type

                self._model = WhisperModel(
                    self.model_name,
                    device=self._device_str,
                    compute_type=ct,
                )
                self._backend = "faster_whisper"
                logger.debug(
                    "Loaded faster-whisper %s on %s with %s",
                    self.model_name,
                    self._device_str,
                    ct,
                )
            except ImportError:
                # Fall back to openai-whisper (CPU only)
                try:
                    import whisper

                    self._model = whisper.load_model(self.model_name)
                    self._backend = "openai_whisper"
                    logger.debug("Loaded openai-whisper %s (CPU)", self.model_name)
                except ImportError as e:
                    raise ImportError(
                        "Whisper transcription requires either faster-whisper "
                        "(pip install audiotrove[transcribe-gpu]) or openai-whisper "
                        "(pip install audiotrove[transcribe])"
                    ) from e
        return self._model

    def transform(self, doc: AudioDocument) -> AudioDocument:
        import numpy as np

        audio = doc.audio.astype(np.float32)

        if self._backend == "faster_whisper":
            # faster-whisper returns segments iterator
            segments, _ = self.model.transcribe(audio, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
        else:
            # openai-whisper
            import whisper

            result = whisper.transcribe(self.model, audio, fp16=False)
            text = result.get("text", "").strip()

        doc.metadata["transcription"] = text
        doc.metadata["transcription_backend"] = self._backend
        logger.debug("Transcribed %s: %r", doc.source_path, text[:80])
        return doc
