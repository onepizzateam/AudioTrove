import numpy as np

from audiotrove.io import rust_backend


def test_resample_python_fallback(monkeypatch):
    monkeypatch.setattr(rust_backend, "_core", None)
    result = rust_backend.resample(np.arange(4, dtype=np.float32), 4, 8)
    assert result.dtype == np.float32
    assert len(result) == 8


def test_resample_uses_optional_extension(monkeypatch):
    class FakeCore:
        @staticmethod
        def resample(audio, source_rate, target_rate):
            return [42.0]

    monkeypatch.setattr(rust_backend, "_core", FakeCore())
    result = rust_backend.resample(np.arange(4, dtype=np.float32), 4, 8)
    np.testing.assert_array_equal(result, np.array([42.0], dtype=np.float32))
