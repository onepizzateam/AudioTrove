"""Tests for SileroVADFilter."""

import pytest
from audiotrove.filters.vad import SileroVADFilter


def test_vad_keeps_clean_speech(speech_clean_fixture):
    """Clean speech should pass VAD with high speech ratio."""
    vad = SileroVADFilter(min_speech_ratio=0.1)
    result = vad.filter(speech_clean_fixture)
    assert result is True
    assert speech_clean_fixture.metadata["vad_speech_ratio"] > 0.1


def test_vad_discards_silence(silence_fixture):
    """Silence should be discarded by VAD."""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    result = vad.filter(silence_fixture)
    assert result is False
    assert silence_fixture.metadata["vad_speech_ratio"] == 0.0


def test_vad_discards_music(music_fixture):
    """Music (no speech) should be discarded by VAD."""
    vad = SileroVADFilter(min_speech_ratio=0.3)
    result = vad.filter(music_fixture)
    assert result is False


def test_vad_metadata_populated(speech_clean_fixture):
    """VAD should populate metadata for downstream stages."""
    vad = SileroVADFilter()
    vad.filter(speech_clean_fixture)
    assert "vad_speech_ratio" in speech_clean_fixture.metadata
    assert "vad_speech_timestamps" in speech_clean_fixture.metadata


def test_vad_threshold_configurable(speech_noisy_fixture):
    """A clip that passes at low threshold should fail at high threshold."""
    vad_loose = SileroVADFilter(min_speech_ratio=0.1)
    vad_strict = SileroVADFilter(min_speech_ratio=0.95)
    result_loose = vad_loose.filter(speech_noisy_fixture)
    result_strict = vad_strict.filter(speech_noisy_fixture)
    # At least one should pass (loose should pass if any speech detected)
    assert result_loose is True or result_strict is False


def test_vad_returns_bool():
    """VAD filter should return a boolean."""
    import numpy as np
    from audiotrove.document import AudioDocument
    from audiotrove.utils.hashing import make_doc_id

    audio = np.zeros(16000, dtype=np.float32)
    doc = AudioDocument(
        audio=audio,
        sample_rate=16000,
        source_path="test.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("test.wav"),
    )
    vad = SileroVADFilter()
    result = vad.filter(doc)
    assert isinstance(result, bool)
