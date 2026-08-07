"""Tests for TTS inference sessions."""

import sys
import types

import numpy as np
import pytest


def test_inference_result_dataclass():
    """InferenceResult should hold audio/text/metadata."""
    from audiotrove.inference.base import InferenceResult

    result = InferenceResult(
        audio=np.array([0.1, 0.2], dtype=np.float32),
        text=None,
        sample_rate=22050,
        metadata={"model": "test"},
    )
    assert result.audio is not None
    assert result.text is None
    assert result.sample_rate == 22050
    assert result.metadata["model"] == "test"


def test_inference_result_defaults():
    """InferenceResult should default to empty audio/text/metadata."""
    from audiotrove.inference.base import InferenceResult

    result = InferenceResult()
    assert result.audio is None
    assert result.text is None
    assert result.sample_rate == 0
    assert result.metadata == {}


def test_get_tts_session_f5tts():
    """get_tts_session('f5tts') should return F5TTSSession."""
    from audiotrove.inference.tts import get_tts_session

    session = get_tts_session("f5tts", model_path="/fake/model.pt", device="cpu")
    assert session.__class__.__name__ == "F5TTSSession"


def test_get_tts_session_case_insensitive():
    """Factory should accept mixed-case family names."""
    from audiotrove.inference.tts import get_tts_session

    session = get_tts_session("F5TTS", model_path="/fake", device="cpu")
    assert session.__class__.__name__ == "F5TTSSession"


def test_get_tts_session_unknown_raises():
    """get_tts_session with unknown family should raise ValueError."""
    from audiotrove.inference.tts import get_tts_session

    with pytest.raises(ValueError, match="Unknown TTS family"):
        get_tts_session("nonexistent")


def test_package_level_factory():
    """audiotrove.inference.get_tts_session should proxy the tts factory."""
    from audiotrove.inference import get_tts_session

    session = get_tts_session("piper", model_path="/fake", device="cpu")
    assert session.__class__.__name__ == "PiperSession"


def test_f5tts_session_load_without_dep_raises():
    """F5TTSSession.load() should raise ImportError when f5_tts missing."""
    from audiotrove.inference.tts import F5TTSSession

    if "f5_tts" in sys.modules:
        pytest.skip("f5_tts is installed; import-guard path not exercised")

    session = F5TTSSession(model_path="/fake/model.pt", device="cpu")
    with pytest.raises(ImportError, match="F5-TTS requires"):
        session.load()


def test_f5tts_session_run_not_loaded_raises():
    """F5TTSSession.run() should raise RuntimeError when model not loaded."""
    from audiotrove.inference.tts import F5TTSSession

    session = F5TTSSession(model_path="/fake/model.pt", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(text="Hello")


def test_f5tts_session_load_and_run_with_fake_backend(monkeypatch):
    """load()/run()/unload() should drive the F5TTS backend when importable."""
    from audiotrove.inference.tts import F5TTSSession

    calls = {}

    class _FakeF5TTS:
        def __init__(self, model_type, ckpt_file, device):
            calls["init"] = (model_type, ckpt_file, device)

        def infer(self, ref_file, ref_text, gen_text, **kwargs):
            calls["infer"] = (ref_file, ref_text, gen_text)
            return np.zeros(8, dtype=np.float32), 24000, None

    fake_api = types.ModuleType("f5_tts.api")
    fake_api.F5TTS = _FakeF5TTS
    fake_pkg = types.ModuleType("f5_tts")
    monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)
    monkeypatch.setitem(sys.modules, "f5_tts.api", fake_api)

    session = F5TTSSession(model_path="/fake/model.pt", device="cpu", voice_ref="ref.wav")
    session.load()
    result = session.run(text="hello world")
    assert result.sample_rate == 24000
    assert result.audio is not None
    assert calls["infer"][2] == "hello world"
    session.unload()
    assert session._model is None


def test_f5tts_session_run_requires_voice_ref(monkeypatch):
    """run() should raise ValueError when no voice_ref is available."""
    from audiotrove.inference.tts import F5TTSSession

    class _FakeF5TTS:
        def __init__(self, **kwargs):
            pass

    fake_api = types.ModuleType("f5_tts.api")
    fake_api.F5TTS = _FakeF5TTS
    fake_pkg = types.ModuleType("f5_tts")
    monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)
    monkeypatch.setitem(sys.modules, "f5_tts.api", fake_api)

    session = F5TTSSession(model_path="/fake/model.pt", device="cpu")
    session.load()
    with pytest.raises(ValueError, match="voice_ref required"):
        session.run(text="hello")


def test_styletts2_session_instantiation():
    """StyleTTS2Session should instantiate without immediate import."""
    from audiotrove.inference.tts import StyleTTS2Session

    session = StyleTTS2Session(model_path="/fake/model.pt", device="cpu")
    assert session.model_path == "/fake/model.pt"


def test_styletts2_run_not_loaded_raises():
    from audiotrove.inference.tts import StyleTTS2Session

    session = StyleTTS2Session(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(text="hi")


def test_piper_session_instantiation():
    """PiperSession should instantiate without immediate import."""
    from audiotrove.inference.tts import PiperSession

    session = PiperSession(model_path="/fake/model.onnx", device="cpu")
    assert session is not None


def test_piper_run_not_loaded_raises():
    from audiotrove.inference.tts import PiperSession

    session = PiperSession(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(text="hi")


def test_chatterbox_session_instantiation():
    """ChatterboxSession should instantiate without immediate import."""
    from audiotrove.inference.tts import ChatterboxSession

    session = ChatterboxSession(model_path="/fake/model.pt", device="cpu")
    assert session is not None


def test_chatterbox_run_not_loaded_raises():
    from audiotrove.inference.tts import ChatterboxSession

    session = ChatterboxSession(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(text="hi")


def test_matcha_session_instantiation():
    """MatchaTTSSession should instantiate without immediate import."""
    from audiotrove.inference.tts import MatchaTTSSession

    session = MatchaTTSSession(model_path="/fake/model.pt", device="cpu")
    assert session is not None


def test_matcha_run_not_loaded_raises():
    from audiotrove.inference.tts import MatchaTTSSession

    session = MatchaTTSSession(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(text="hi")


def test_inference_session_context_manager(monkeypatch):
    """InferenceSession should support the context-manager protocol."""
    from audiotrove.inference.tts import F5TTSSession

    class _FakeF5TTS:
        def __init__(self, **kwargs):
            pass

    fake_api = types.ModuleType("f5_tts.api")
    fake_api.F5TTS = _FakeF5TTS
    fake_pkg = types.ModuleType("f5_tts")
    monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)
    monkeypatch.setitem(sys.modules, "f5_tts.api", fake_api)

    session = F5TTSSession(model_path="/fake/model.pt", device="cpu")
    with session as s:
        assert s is session
        assert session._model is not None
    assert session._model is None


def test_get_tts_session_factory_coverage():
    """Ensure all TTS families in the registry instantiate correctly."""
    from audiotrove.inference.tts import get_tts_session

    for family in ["f5tts", "styletts2", "piper", "chatterbox", "matcha"]:
        session = get_tts_session(family, model_path="/fake", device="cpu")
        assert session is not None


def test_styletts2_load_and_run_with_fake_backend(monkeypatch):
    """StyleTTS2Session should invoke the backend's inference method."""
    from audiotrove.inference.tts import StyleTTS2Session

    class _FakeModel:
        sampling_rate = 24000

        def inference(self, text, ref, **kwargs):
            return np.zeros(16, dtype=np.float32)

    class _FakeTTSInference:
        @staticmethod
        def load_model(path, device):
            return _FakeModel()

    fake_pkg = types.ModuleType("styletts2")
    fake_pkg.tts_inference = _FakeTTSInference()
    monkeypatch.setitem(sys.modules, "styletts2", fake_pkg)

    session = StyleTTS2Session(model_path="/fake", device="cpu", voice_ref="ref.wav")
    session.load()
    result = session.run(text="test")
    assert result.sample_rate == 24000
    assert result.audio is not None


def test_piper_load_and_run_with_fake_backend(monkeypatch):
    """PiperSession should call the backend's synthesize method."""
    from audiotrove.inference.tts import PiperSession

    class _FakeConfig:
        sample_rate = 22050

    class _FakeVoice:
        config = _FakeConfig()

        def synthesize(self, text, **kwargs):
            return (np.zeros(8, dtype=np.int16) + 16000).tobytes()

    class _FakePiper:
        class PiperVoice:
            @staticmethod
            def load(path):
                return _FakeVoice()

    fake_pkg = types.ModuleType("piper")
    fake_pkg.PiperVoice = _FakePiper.PiperVoice
    monkeypatch.setitem(sys.modules, "piper", fake_pkg)

    session = PiperSession(model_path="/fake.onnx", device="cpu")
    session.load()
    result = session.run(text="hello")
    assert result.sample_rate == 22050
    assert result.audio is not None


def test_chatterbox_load_and_run_with_fake_backend(monkeypatch):
    """ChatterboxSession should call synthesize with ref audio."""
    from audiotrove.inference.tts import ChatterboxSession

    class _FakeModel:
        def synthesize(self, text, ref, **kwargs):
            return np.zeros(12, dtype=np.float32), 24000

    def _fake_load_model(path, device):
        return _FakeModel()

    fake_pkg = types.ModuleType("chatterbox")
    fake_pkg.load_model = _fake_load_model
    monkeypatch.setitem(sys.modules, "chatterbox", fake_pkg)

    session = ChatterboxSession(model_path="/fake", device="cpu", voice_ref="ref.wav")
    session.load()
    result = session.run(text="test")
    assert result.sample_rate == 24000


def test_matcha_load_and_run_with_fake_backend(monkeypatch):
    """MatchaTTSSession should call the backend synthesize."""
    from audiotrove.inference.tts import MatchaTTSSession

    class _FakeModel:
        def synthesize(self, text, **kwargs):
            return np.zeros(10, dtype=np.float32), 22050

    def _fake_load_model(path, device):
        return _FakeModel()

    fake_pkg = types.ModuleType("matcha_tts")
    fake_pkg.load_model = _fake_load_model
    monkeypatch.setitem(sys.modules, "matcha_tts", fake_pkg)

    session = MatchaTTSSession(model_path="/fake", device="cpu")
    session.load()
    result = session.run(text="hello world")
    assert result.sample_rate == 22050
