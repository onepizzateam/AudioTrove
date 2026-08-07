"""Tests for voice-conversion inference sessions."""

import sys
import types

import numpy as np
import pytest


def test_get_vc_session_families():
    from audiotrove.inference.vc import get_vc_session

    assert get_vc_session("seed_vc", model_path="/fake").__class__.__name__ == "SeedVCSession"
    assert get_vc_session("rvc", model_path="/fake").__class__.__name__ == "RVCSession"


def test_get_vc_session_unknown_raises():
    from audiotrove.inference.vc import get_vc_session

    with pytest.raises(ValueError, match="Unknown VC family"):
        get_vc_session("nope", model_path="/fake")


def test_package_level_vc_factory():
    from audiotrove.inference import get_vc_session

    session = get_vc_session("rvc", model_path="/fake")
    assert session.__class__.__name__ == "RVCSession"


def test_seed_vc_load_without_dep_raises():
    from audiotrove.inference.vc import SeedVCSession

    if "seed_vc" in sys.modules:
        pytest.skip("seed_vc installed; guard path not exercised")

    session = SeedVCSession(model_path="/fake", device="cpu")
    with pytest.raises(ImportError, match="SeedVC requires"):
        session.load()


def test_seed_vc_run_not_loaded_raises():
    from audiotrove.inference.vc import SeedVCSession

    session = SeedVCSession(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(source_audio_path="/a.wav", target_voice_path="/b.wav")


def test_seed_vc_load_and_run_with_fake_backend(monkeypatch):
    from audiotrove.inference.vc import SeedVCSession

    class _FakeModel:
        def convert(self, source, target, **kwargs):
            return np.zeros(4, dtype=np.float32), 16000

    fake_mod = types.ModuleType("seed_vc")
    fake_mod.load_model = lambda path, device: _FakeModel()
    monkeypatch.setitem(sys.modules, "seed_vc", fake_mod)

    session = SeedVCSession(model_path="/fake", device="cpu")
    session.load()
    result = session.run(source_audio_path="/a.wav", target_voice_path="/b.wav")
    assert result.sample_rate == 16000
    assert result.audio is not None
    session.unload()
    assert session._model is None


def test_rvc_run_not_loaded_raises():
    from audiotrove.inference.vc import RVCSession

    session = RVCSession(model_path="/fake", device="cpu")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        session.run(source_audio_path="/a.wav", target_voice_path="/b.wav")


def test_rvc_load_and_run_with_fake_backend(monkeypatch):
    from audiotrove.inference.vc import RVCSession

    class _FakeModel:
        def convert(self, source, target, **kwargs):
            return np.zeros(4, dtype=np.float32), 40000

    fake_mod = types.ModuleType("rvc")
    fake_mod.load_model = lambda path, device: _FakeModel()
    monkeypatch.setitem(sys.modules, "rvc", fake_mod)

    session = RVCSession(model_path="/fake", device="cpu")
    session.load()
    result = session.run(source_audio_path="/a.wav", target_voice_path="/b.wav")
    assert result.sample_rate == 40000
