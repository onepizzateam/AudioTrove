"""Tests for WhisperTranscriber — does not require openai-whisper installed."""

import sys
import types

import numpy as np
import pytest

from audiotrove.document import AudioDocument


def _make_doc():
    return AudioDocument(
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        source_path="test.wav",
        duration_seconds=1.0,
        doc_id="abc123",
    )


def test_whisper_transcriber_raises_without_whisper(monkeypatch):
    """WhisperTranscriber.__init__ raises ImportError if openai-whisper is absent."""
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(ImportError, match="transcribe"):
        from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

        WhisperTranscriber()


def test_whisper_transcriber_writes_metadata(monkeypatch):
    """WhisperTranscriber.transform writes transcription to doc.metadata."""
    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = lambda name: object()
    fake_whisper.transcribe = lambda model, audio, **kwargs: {"text": " hello world"}
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    sys.modules.pop("audiotrove.transformers.whisper_transcribe", None)
    from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

    transformer = WhisperTranscriber.__new__(WhisperTranscriber)
    transformer.model_name = "base"
    transformer._model = object()

    result = transformer.transform(_make_doc())
    assert result.metadata["transcription"] == "hello world"


def test_tts_pipeline_accepts_transcribe_flag():
    """tts_pipeline() accepts transcription options."""
    import inspect

    from audiotrove.pipelines.tts import tts_pipeline

    sig = inspect.signature(tts_pipeline)
    assert "transcribe" in sig.parameters
    assert "whisper_model" in sig.parameters
