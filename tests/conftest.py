"""Pytest fixtures that load real WAV files from tests/fixtures/."""

import pytest
import numpy as np
from pathlib import Path

from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


@pytest.fixture(autouse=True)
def prevent_network_model_downloads(monkeypatch):
    """Keep the suite offline: VAD production fallback handles this failure."""
    try:
        import torch
    except ImportError:
        return

    def fail_fast(*_args, **_kwargs):
        raise RuntimeError("network model downloads are disabled in tests")

    monkeypatch.setattr(torch.hub, "load", fail_fast)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> AudioDocument:
    import wave

    path = FIXTURES_DIR / name
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    # Only 16-bit PCM expected from our generator
    if sampwidth == 2:
        dtype = np.int16
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 32767.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    return AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path=str(path),
        duration_seconds=float(len(audio)) / sr,
        doc_id=make_doc_id(str(path)),
    )


@pytest.fixture
def speech_clean():
    return _load_fixture("speech_clean.wav")


@pytest.fixture
def speech_noisy():
    return _load_fixture("speech_noisy.wav")


@pytest.fixture
def silence():
    return _load_fixture("silence.wav")


@pytest.fixture
def music():
    return _load_fixture("music.wav")


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


# Backwards-compatible fixture names used by older tests
@pytest.fixture
def speech_clean_fixture(speech_clean):
    return speech_clean


@pytest.fixture
def speech_noisy_fixture(speech_noisy):
    return speech_noisy


@pytest.fixture
def silence_fixture(silence):
    return silence


@pytest.fixture
def music_fixture(music):
    return music
