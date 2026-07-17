import pytest


df = pytest.importorskip("df")


def test_deepfilternet_enhance_fixture(fixtures_dir):
    import soundfile as sf

    from audiotrove.filters.enhance import DeepFilterEnhancer

    audio, sample_rate = sf.read(str(fixtures_dir / "speech_clean.wav"), dtype="float32")
    enhanced = DeepFilterEnhancer().enhance(audio, sample_rate)

    assert enhanced.shape == audio.shape
    assert enhanced.dtype == "float32"