"""Tests for ASR inference sessions."""

import sys
import types

import numpy as np
import pytest


def test_get_asr_session_default():
    """get_asr_session() should default to FasterWhisperSession."""
    from audiotrove.inference.asr import get_asr_session

    session = get_asr_session()
    assert session.__class__.__name__ == "FasterWhisperSession"


def test_get_asr_session_families():
    from audiotrove.inference.asr import get_asr_session

    assert get_asr_session("faster_whisper").__class__.__name__ == "FasterWhisperSession"
    assert get_asr_session("qwen3").__class__.__name__ == "Qwen3ASRSession"
    assert get_asr_session("parakeet").__class__.__name__ == "ParakeetSession"


def test_get_asr_session_unknown_raises():
    from audiotrove.inference.asr import get_asr_session

    with pytest.raises(ValueError, match="Unknown ASR family"):
        get_asr_session("nope")


def test_package_level_asr_factory():
    from audiotrove.inference import get_asr_session

    session = get_asr_session("faster_whisper")
    assert session.__class__.__name__ == "FasterWhisperSession"


def test_faster_whisper_load_without_dep_raises():
    from audiotrove.inference.asr import FasterWhisperSession

    if "faster_whisper" in sys.modules:
        pytest.skip("faster_whisper installed; guard path not exercised")

    session = FasterWhisperSession(model_path="base", device="cpu")
    with pytest.raises(ImportError, match="faster-whisper requires"):
        session.load()


def test_faster_whisper_run_not_loaded_raises():
    from audiotrove.inference.asr import FasterWhisperSession

    session = FasterWhisperSession()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(audio_path="/fake.wav")


def test_faster_whisper_load_and_run_with_fake_backend(monkeypatch):
    """load()/run() should transcribe using a fake faster_whisper backend."""
    from audiotrove.inference.asr import FasterWhisperSession

    class _Seg:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    class _Info:
        language = "en"

    class _FakeModel:
        def __init__(self, model_path, device, compute_type):
            self.args = (model_path, device, compute_type)

        def transcribe(self, audio_path, beam_size=5, **kwargs):
            return iter([_Seg("hello", 0.0, 1.0), _Seg("world", 1.0, 2.0)]), _Info()

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)

    session = FasterWhisperSession(model_path="base", device="cpu")
    session.load()
    result = session.run(audio_path="/fake.wav")
    assert result.text == "hello world"
    assert result.metadata["language"] == "en"
    assert len(result.metadata["segments"]) == 2
    session.unload()
    assert session._model is None


def test_faster_whisper_explicit_compute_type(monkeypatch):
    """A non-auto compute_type should be forwarded to the backend."""
    from audiotrove.inference.asr import FasterWhisperSession

    captured = {}

    class _FakeModel:
        def __init__(self, model_path, device, compute_type):
            captured["compute_type"] = compute_type

        def transcribe(self, audio_path, beam_size=5, **kwargs):
            return iter([]), types.SimpleNamespace(language=None)

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)

    session = FasterWhisperSession(model_path="base", device="cpu", compute_type="float32")
    session.load()
    assert captured["compute_type"] == "float32"


def test_qwen3_session_instantiation():
    from audiotrove.inference.asr import Qwen3ASRSession

    session = Qwen3ASRSession()
    assert session.model_path == "Qwen/Qwen3-ASR"


def test_qwen3_run_not_loaded_raises():
    from audiotrove.inference.asr import Qwen3ASRSession

    session = Qwen3ASRSession()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(audio_path="/fake.wav")


def test_parakeet_session_instantiation():
    from audiotrove.inference.asr import ParakeetSession

    session = ParakeetSession()
    assert "parakeet" in session.model_path


def test_parakeet_run_not_loaded_raises():
    from audiotrove.inference.asr import ParakeetSession

    session = ParakeetSession()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(audio_path="/fake.wav")


def test_qwen3_load_and_run_with_fake_backend(tmp_path, monkeypatch):
    """Qwen3ASRSession should use transformers AutoModel."""
    from scipy.io import wavfile

    from audiotrove.inference.asr import Qwen3ASRSession

    # Create a real audio file
    sr = 16000
    audio = np.zeros(sr, dtype=np.int16)
    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), sr, audio)

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(path):
            return _FakeProcessor()

        def __call__(self, audio, sampling_rate, return_tensors):
            return {"input_features": "fake"}

        def batch_decode(self, generated, skip_special_tokens):
            return [" transcribed text "]

    class _FakeModel:
        @staticmethod
        def from_pretrained(path):
            return _FakeModel()

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return ["fake_tokens"]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSpeechSeq2Seq = _FakeModel
    fake_transformers.AutoProcessor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    session = Qwen3ASRSession(model_path="Qwen/Qwen3-ASR", device="cpu")
    session.load()
    result = session.run(audio_path=str(wav_path))
    assert result.text == "transcribed text"
    assert result.audio is None


def test_parakeet_load_and_run_with_fake_backend(tmp_path, monkeypatch):
    """ParakeetSession should use nemo ASRModel."""
    from scipy.io import wavfile

    from audiotrove.inference.asr import ParakeetSession

    sr = 16000
    audio = np.zeros(sr, dtype=np.int16)
    wav_path = tmp_path / "test.wav"
    wavfile.write(str(wav_path), sr, audio)

    class _FakeASRModel:
        @staticmethod
        def from_pretrained(path):
            return _FakeASRModel()

        def transcribe(self, paths, **kwargs):
            return ["hello from parakeet"]

    class _FakeNeMoASR:
        class models:
            ASRModel = _FakeASRModel

    fake_nemo = types.ModuleType("nemo")
    fake_nemo_asr = types.ModuleType("nemo.collections.asr")
    fake_nemo_asr.models = _FakeNeMoASR.models
    fake_nemo_collections = types.ModuleType("nemo.collections")
    fake_nemo_collections.asr = fake_nemo_asr
    fake_nemo.collections = fake_nemo_collections

    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_nemo_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_nemo_asr)

    session = ParakeetSession(model_path="nvidia/parakeet-tdt-1.1b", device="cpu")
    session.load()
    result = session.run(audio_path=str(wav_path))
    assert result.text == "hello from parakeet"
