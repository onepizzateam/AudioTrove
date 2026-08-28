"""Tests for standalone VAD inference session."""


import numpy as np
import pytest

from audiotrove.inference.vad import SileroVADInferenceSession, get_vad_session


def test_get_vad_session():
    """get_vad_session() returns a SileroVADInferenceSession."""
    session = get_vad_session()
    assert isinstance(session, SileroVADInferenceSession)


def test_vad_session_instantiation():
    """SileroVADInferenceSession can be instantiated."""
    session = SileroVADInferenceSession(device="cpu", threshold=0.5)
    assert session.device == "cpu"
    assert session.threshold == 0.5
    assert session._filter is None


def test_vad_session_load(monkeypatch):
    """load() instantiates the SileroVADFilter and forces model load."""
    from audiotrove.inference.vad import SileroVADInferenceSession

    loaded = {"count": 0}

    class _FakeFilter:
        def __init__(self, threshold, device):
            self.threshold = threshold
            self.device = device
            self._model = None

        @property
        def model(self):
            loaded["count"] += 1
            if self._model is None:
                self._model = object()
            return self._model

        def filter(self, doc):
            doc.metadata["vad_speech_timestamps"] = []
            doc.metadata["speech_ratio"] = 0.0

    import audiotrove.filters.vad as vad_mod

    monkeypatch.setattr(vad_mod, "SileroVADFilter", _FakeFilter)

    session = SileroVADInferenceSession(device="cpu", threshold=0.6)
    session.load()
    assert session._filter is not None
    assert session._filter.threshold == 0.6
    assert loaded["count"] == 1  # Model was accessed during load


def test_vad_session_run_not_loaded_raises():
    """run() raises RuntimeError if not loaded."""
    session = SileroVADInferenceSession()
    with pytest.raises(RuntimeError, match="not loaded"):
        session.run(audio_path="/fake.wav")


def test_vad_session_run(tmp_path, monkeypatch):
    """run() reads audio and returns speech timestamps via InferenceResult."""
    from scipy.io import wavfile

    from audiotrove.inference.vad import SileroVADInferenceSession

    # Create a real audio file
    sr = 16000
    duration_s = 1.0
    samples = int(sr * duration_s)
    audio = (np.sin(2 * np.pi * 440 * np.arange(samples) / sr) * 32767).astype(
        np.int16
    )
    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), sr, audio)

    # Fake filter that populates metadata
    class _FakeFilter:
        def __init__(self, threshold, device):
            self.threshold = threshold
            self.device = device

        @property
        def model(self):
            return object()

        def filter(self, doc):
            doc.metadata["vad_speech_timestamps"] = [
                {"start": 100, "end": 8000},
                {"start": 10000, "end": 15000},
            ]
            doc.metadata["speech_ratio"] = 0.8

    import audiotrove.filters.vad as vad_mod

    monkeypatch.setattr(vad_mod, "SileroVADFilter", _FakeFilter)

    session = SileroVADInferenceSession(device="cpu", threshold=0.5)
    session.load()
    result = session.run(audio_path=str(wav_path))

    assert result.audio is None
    assert result.text is None
    assert result.sample_rate == sr
    assert len(result.metadata["speech_timestamps"]) == 2
    assert result.metadata["speech_ratio"] == 0.8


def test_vad_session_unload():
    """unload() clears the filter."""
    session = SileroVADInferenceSession()
    session._filter = object()
    session.unload()
    assert session._filter is None


def test_vad_session_context_manager(tmp_path, monkeypatch):
    """SileroVADInferenceSession works as a context manager."""
    from scipy.io import wavfile

    from audiotrove.inference.vad import SileroVADInferenceSession

    sr = 16000
    audio = np.zeros(sr, dtype=np.int16)
    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), sr, audio)

    class _FakeFilter:
        def __init__(self, threshold, device):
            pass

        @property
        def model(self):
            return object()

        def filter(self, doc):
            doc.metadata["vad_speech_timestamps"] = []
            doc.metadata["speech_ratio"] = 0.0

    import audiotrove.filters.vad as vad_mod

    monkeypatch.setattr(vad_mod, "SileroVADFilter", _FakeFilter)

    session = SileroVADInferenceSession()
    with session:
        result = session.run(audio_path=str(wav_path))
        assert result.metadata["speech_ratio"] == 0.0
    # After exit, unload() was called
    assert session._filter is None
