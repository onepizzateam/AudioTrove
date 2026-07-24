import numpy as np
from audiotrove.filters.vad import SileroVADFilter, VADSegmenter
from audiotrove.filters.snr import SNRFilter
from audiotrove.document import AudioDocument


def _make_doc_with_audio(length=16000, peaks=None):
    audio = np.zeros(length, dtype=float)
    if peaks:
        for s, e in peaks:
            audio[s:e] = 0.1
    return AudioDocument(
        audio=audio,
        sample_rate=16000,
        source_path="x.wav",
        duration_seconds=float(length) / 16000.0,
        doc_id="doc-vad",
    )


def test_energy_vad_silence_returns_empty():
    f = SileroVADFilter(min_speech_ratio=0.1)
    doc = _make_doc_with_audio(1600, peaks=None)
    ts = f._energy_vad(doc.audio, doc.sample_rate)
    assert ts == []


def test_energy_vad_detects_peaks():
    f = SileroVADFilter(min_speech_ratio=0.01, window_size_samples=128)
    # insert a speech-like region
    doc = _make_doc_with_audio(1600, peaks=[(200, 400), (800, 900)])
    ts = f._energy_vad(doc.audio, doc.sample_rate)
    assert isinstance(ts, list)
    assert len(ts) >= 1


def test_silero_filter_energy_fallback(monkeypatch):
    # Ensure HAS_TORCH is False for fallback
    import audiotrove.filters.vad as vad_mod

    monkeypatch.setattr(vad_mod, "HAS_TORCH", False)
    f = SileroVADFilter(min_speech_ratio=0.0, window_size_samples=128)
    doc = _make_doc_with_audio(1600, peaks=[(0, 200)])
    f.filter(doc)
    assert "vad_backend" in doc.metadata
    assert doc.metadata["vad_backend"] == "energy_fallback"


def test_vad_segmenter_transform_returns_segments():
    seg = VADSegmenter(window_size_samples=128)
    doc = _make_doc_with_audio(1600, peaks=[(100, 300), (600, 720)])
    segments = seg.transform(doc)
    assert isinstance(segments, list)
    for s in segments:
        assert "parent_doc_id" in s.metadata


def test_snr_energy_fallback_short_audio():
    s = SNRFilter(min_snr_db=0.0)
    doc = AudioDocument(
        audio=np.zeros(100, dtype=float),
        sample_rate=16000,
        source_path="a.wav",
        duration_seconds=0.01,
        doc_id="d",
    )
    val = s._energy_fallback_snr(doc.audio)
    assert val == 40.0


def test_compute_snr_insufficient_noise():
    s = SNRFilter(min_snr_db=0.0)
    # audio with small noise region
    audio = np.ones(16000) * 0.1
    doc = AudioDocument(
        audio=audio, sample_rate=16000, source_path="a.wav", duration_seconds=1.0, doc_id="d"
    )
    # mark most as speech so noise < 0.1s
    doc.metadata["vad_speech_timestamps"] = [{"start": 0, "end": 15900}]
    val = s._compute_snr(doc)
    assert val == 40.0
    assert doc.metadata.get("snr_note") == "insufficient_noise_floor"


def test_silero_vad_path(monkeypatch):
    import audiotrove.filters.vad as vad_mod

    # Fake torch with from_numpy
    class FakeTorch:
        @staticmethod
        def from_numpy(x):
            return x

        hub = type("H", (), {})()

    monkeypatch.setattr(vad_mod, "torch", FakeTorch, raising=False)
    monkeypatch.setattr(vad_mod, "HAS_TORCH", True, raising=False)

    f = SileroVADFilter(min_speech_ratio=0.0)
    # Set fake model and utils
    f._model = True

    class Utils:
        @staticmethod
        def get_speech_timestamps(audio_t, model, sampling_rate, threshold, window_size_samples):
            return [{"start": 0, "end": 200}]

    f._utils = Utils()
    doc = _make_doc_with_audio(1600, peaks=[(0, 200)])
    f.filter(doc)
    assert doc.metadata["vad_backend"] == "silero"
    assert doc.metadata["vad_speech_timestamps"]


def test_silero_model_load_failure(monkeypatch):
    import audiotrove.filters.vad as vad_mod

    class FakeTorch2:
        class hub:
            @staticmethod
            def load(*a, **k):
                raise RuntimeError("no-net")

    monkeypatch.setattr(vad_mod, "torch", FakeTorch2, raising=False)
    monkeypatch.setattr(vad_mod, "HAS_TORCH", True, raising=False)
    f = SileroVADFilter()
    # Accessing model should catch the exception and return None
    m = f.model
    assert m is None
