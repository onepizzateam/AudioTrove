"""Tests for SNRFilter."""
import pytest
import numpy as np
from audiotrove.filters.snr import SNRFilter
from audiotrove.filters.vad import SileroVADFilter
from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


def test_snr_populates_metadata(speech_clean_fixture):
    """SNR filter should populate snr_db in metadata."""
    snr = SNRFilter(min_snr_db=0.0)
    snr.filter(speech_clean_fixture)
    assert 'snr_db' in speech_clean_fixture.metadata


def test_snr_returns_bool(speech_clean_fixture):
    """SNR filter should return a boolean."""
    snr = SNRFilter()
    result = snr.filter(speech_clean_fixture)
    assert isinstance(result, bool)


def test_snr_fallback_without_vad():
    """SNR filter should work even without prior VAD metadata (fallback)."""
    audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    doc = AudioDocument(
        audio=audio,
        sample_rate=16000,
        source_path="test.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("test.wav"),
    )
    snr = SNRFilter(min_snr_db=0.0)
    result = snr.filter(doc)
    assert isinstance(result, bool)
    assert 'snr_db' in doc.metadata


def test_snr_with_vad_metadata(speech_clean_fixture):
    """SNR filter should use VAD timestamps if available."""
    # First run VAD to populate timestamps
    vad = SileroVADFilter(min_speech_ratio=0.0)
    vad.filter(speech_clean_fixture)
    
    # Then run SNR
    snr = SNRFilter(min_snr_db=0.0)
    snr.filter(speech_clean_fixture)
    
    # SNR should be computed and metadata populated
    assert 'snr_db' in speech_clean_fixture.metadata
    assert isinstance(speech_clean_fixture.metadata['snr_db'], (float, int))


def test_snr_threshold_filtering():
    """SNR filter should discard low-SNR audio."""
    # Create very noisy audio (mostly noise, little signal)
    noise = np.random.normal(0, 0.3, 16000).astype(np.float32)
    doc = AudioDocument(
        audio=noise,
        sample_rate=16000,
        source_path="noisy.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("noisy.wav"),
    )
    
    # Strict threshold (very high SNR required)
    snr = SNRFilter(min_snr_db=50.0)
    result = snr.filter(doc)
    assert result is False  # Should be rejected


def test_snr_clean_audio_passes():
    """SNR filter should accept clean audio."""
    # Create signal with explicit noise for better SNR computation
    np.random.seed(42)
    t = np.arange(16000) / 16000
    signal = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    # Add very small noise
    noise = np.random.normal(0, 0.01, 16000).astype(np.float32)
    clean_audio = (signal + noise).astype(np.float32)
    
    doc = AudioDocument(
        audio=clean_audio,
        sample_rate=16000,
        source_path="clean.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("clean.wav"),
    )
    
    # Threshold at 0 dB (should accept)
    snr = SNRFilter(min_snr_db=0.0)
    result = snr.filter(doc)
    assert result is True  # Should be accepted
