"""Pytest configuration and fixtures for AudioTrove tests."""
import numpy as np
from pathlib import Path
import pytest

from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


def pytest_configure(config):
    """Called before test collection; we don't need to generate WAV files."""
    # Fixtures are generated on-demand by pytest fixtures below
    pass


# Pytest fixtures that generate synthetic audio on-demand
@pytest.fixture
def speech_clean_fixture():
    """Generate clean speech-like audio fixture."""
    sr = 16000
    duration_s = 5
    samples = sr * duration_s
    t = np.arange(samples) / sr
    
    # Speech-like signal: 200Hz + 300Hz sine with envelope
    speech = 0.2 * (np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 300 * t))
    try:
        from scipy.signal import windows
        envelope = windows.hann(samples)
    except ImportError:
        envelope = np.hanning(samples)
    audio = (speech * envelope * 0.5).astype(np.float32)
    
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="speech_clean.wav",
        duration_seconds=duration_s,
        doc_id=make_doc_id("speech_clean.wav"),
    )


@pytest.fixture
def speech_noisy_fixture():
    """Generate noisy speech fixture."""
    sr = 16000
    duration_s = 5
    samples = sr * duration_s
    t = np.arange(samples) / sr
    
    # Speech + noise
    speech = 0.2 * (np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 300 * t))
    try:
        from scipy.signal import windows
        envelope = windows.hann(samples)
    except ImportError:
        envelope = np.hanning(samples)
    speech = speech * envelope * 0.5
    
    noise = np.random.normal(0, 0.1, samples).astype(np.float32)
    audio = (0.7 * speech + 0.3 * noise).astype(np.float32)
    np.clip(audio, -1, 1, out=audio)
    
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="speech_noisy.wav",
        duration_seconds=duration_s,
        doc_id=make_doc_id("speech_noisy.wav"),
    )


@pytest.fixture
def silence_fixture():
    """Generate silence fixture."""
    sr = 16000
    duration_s = 5
    samples = sr * duration_s
    
    # Near-silence
    audio = np.random.normal(0, 0.001, samples).astype(np.float32)
    
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="silence.wav",
        duration_seconds=duration_s,
        doc_id=make_doc_id("silence.wav"),
    )


@pytest.fixture
def music_fixture():
    """Generate music fixture (no speech)."""
    sr = 16000
    duration_s = 5
    samples = sr * duration_s
    t = np.arange(samples) / sr
    
    # Music: 1kHz + 2kHz sine, no envelope
    audio = (0.3 * (0.5 * np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t))).astype(np.float32)
    
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="music.wav",
        duration_seconds=duration_s,
        doc_id=make_doc_id("music.wav"),
    )


@pytest.fixture
def corrupt_fixture():
    """Generate corrupt/truncated fixture."""
    sr = 16000
    samples = sr  # Only 1 second
    t = np.arange(samples) / sr
    
    # Truncated speech
    speech = 0.2 * (np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 300 * t))
    try:
        from scipy.signal import windows
        envelope = windows.hann(samples)
    except ImportError:
        envelope = np.hanning(samples)
    audio = (speech * envelope * 0.5).astype(np.float32)
    
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="corrupt.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("corrupt.wav"),
    )
