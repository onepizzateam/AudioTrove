"""Optional Whisper transcription transformer."""

from __future__ import annotations

import logging

from audiotrove.base import AudioTransformer
from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)


class WhisperTranscriber(AudioTransformer):
    """
    Transcribes each AudioDocument using an optional Whisper CPU backend.

    Requires: pip install audiotrove[transcribe]

    Writes doc.metadata["transcription"] with the detected text.
    Does not filter — always returns the document.
    """

    name = "whisper_transcriber"

    def __init__(self, model_name: str = "base", backend: str = "faster") -> None:
        if backend not in {"openai", "faster"}:
            raise ValueError("backend must be 'openai' or 'faster'")
        try:
            __import__("faster_whisper" if backend == "faster" else "whisper")
        except ImportError as e:
            raise ImportError(
                "Whisper transcription requires the 'transcribe' extra: "
                "pip install audiotrove[transcribe]"
            ) from e
        self.model_name = model_name
        self.backend = backend
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.backend == "faster":
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            else:
                import whisper
                self._model = whisper.load_model(self.model_name)
        return self._model

    def transform(self, doc: AudioDocument) -> AudioDocument:
        import numpy as np
        audio = doc.audio.astype(np.float32)
        if getattr(self, "backend", "openai") == "faster":
            segments, _ = self.model.transcribe(audio, language=None)
            text = " ".join(segment.text for segment in segments).strip()
        else:
            import whisper
            result = whisper.transcribe(self.model, audio, fp16=False)
            text = result.get("text", "").strip()
        doc.metadata["transcription"] = text
        logger.debug("Transcribed %s: %r", doc.source_path, text[:80])
        return doc
