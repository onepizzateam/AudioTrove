"""Optional Whisper transcription transformer."""

from __future__ import annotations

import logging

from audiotrove.base import AudioTransformer
from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)


class WhisperTranscriber(AudioTransformer):
    """
    Transcribes each AudioDocument using openai-whisper (CPU).

    Requires: pip install audiotrove[transcribe]

    Writes doc.metadata["transcription"] with the detected text.
    Does not filter — always returns the document.
    """

    name = "whisper_transcriber"

    def __init__(self, model_name: str = "base") -> None:
        try:
            import whisper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Whisper transcription requires the 'transcribe' extra: "
                "pip install audiotrove[transcribe]"
            ) from e
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_name)
        return self._model

    def transform(self, doc: AudioDocument) -> AudioDocument:
        import numpy as np
        import whisper

        audio = doc.audio.astype(np.float32)
        result = whisper.transcribe(self.model, audio, fp16=False)
        text = result.get("text", "").strip()
        doc.metadata["transcription"] = text
        logger.debug("Transcribed %s: %r", doc.source_path, text[:80])
        return doc
