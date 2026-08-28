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
    """WhisperTranscriber lazy-loads and raises ImportError if no backend available."""
    # Block both faster-whisper and openai-whisper
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "whisper", None)
    
    sys.modules.pop("audiotrove.transformers.whisper_transcribe", None)
    from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

    transformer = WhisperTranscriber()
    # The error happens when accessing .model (lazy load), not in __init__
    with pytest.raises(ImportError, match="transcribe"):
        _ = transformer.model


def test_whisper_transcriber_writes_metadata(monkeypatch):
    """WhisperTranscriber.transform writes transcription using openai-whisper fallback."""
    # Block faster-whisper, provide fake openai-whisper
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    
    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = lambda name: object()
    fake_whisper.transcribe = lambda model, audio, **kwargs: {"text": " hello world"}
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    sys.modules.pop("audiotrove.transformers.whisper_transcribe", None)
    from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

    transformer = WhisperTranscriber()
    result = transformer.transform(_make_doc())
    assert result.metadata["transcription"] == "hello world"
    assert result.metadata["transcription_backend"] == "openai_whisper"


def test_tts_pipeline_accepts_transcribe_flag():
    """tts_pipeline() accepts transcription options."""
    import inspect

    from audiotrove.pipelines.tts import tts_pipeline

    sig = inspect.signature(tts_pipeline)
    assert "transcribe" in sig.parameters
    assert "whisper_model" in sig.parameters


def test_whisper_backends_emit_schema_identical_metadata(tmp_path, monkeypatch):
    """Both installed backends write the same metadata.csv columns and types."""
    pytest.importorskip("faster_whisper")
    pytest.importorskip("whisper")
    from audiotrove.exporters.tts_manifest import TTSManifestExporter
    from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

    fake_faster = types.ModuleType("faster_whisper")
    fake_faster.WhisperModel = object
    fake_openai = types.ModuleType("whisper")
    fake_openai.transcribe = lambda model, audio, **kwargs: {"text": "same text"}
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster)
    monkeypatch.setitem(sys.modules, "whisper", fake_openai)

    outputs = []
    for backend in ("faster", "openai"):
        doc = _make_doc()
        transformer = WhisperTranscriber.__new__(WhisperTranscriber)
        transformer.backend = backend
        transformer._model = types.SimpleNamespace(
            transcribe=lambda audio, language=None: ([types.SimpleNamespace(text="same text")], None)
        )
        transformer.transform(doc)
        out = tmp_path / backend
        TTSManifestExporter(str(out), export_format=["ljspeech"]).write(doc)
        row = (out / "metadata.csv").read_text(encoding="utf-8").strip().split("|")
        outputs.append([type(value).__name__ for value in row])
    assert outputs[0] == outputs[1] == ["str", "str", "str"]
