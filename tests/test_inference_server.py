"""Tests for the AudioTrove inference HTTP server."""

import json
import sys
import types

import numpy as np
import pytest


def _write_config(tmp_path, models=None, host="127.0.0.1", port=8080):
    cfg = {"host": host, "port": port, "models": models or []}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_server_reads_config(tmp_path):
    from audiotrove.inference.server import AudioTroveServer

    cfg = _write_config(tmp_path, host="0.0.0.0", port=9000)
    server = AudioTroveServer(cfg)
    assert server.host == "0.0.0.0"
    assert server.port == 9000
    assert server._sessions == {}


def test_model_spec_lookup(tmp_path):
    from audiotrove.inference.server import AudioTroveServer

    models = [{"id": "voice1", "task": "tts", "family": "f5tts"}]
    server = AudioTroveServer(_write_config(tmp_path, models))
    spec = server._model_spec("voice1")
    assert spec["family"] == "f5tts"


def test_model_spec_unknown_raises(tmp_path):
    from audiotrove.inference.server import AudioTroveServer

    server = AudioTroveServer(_write_config(tmp_path, []))
    with pytest.raises(KeyError, match="Unknown model id"):
        server._model_spec("missing")


def test_get_session_lazy_loads_tts(tmp_path, monkeypatch):
    from audiotrove.inference.server import AudioTroveServer

    loaded = {"count": 0}

    class _FakeSession:
        def load(self):
            loaded["count"] += 1

    import audiotrove.inference.tts as tts_mod

    monkeypatch.setattr(
        tts_mod, "get_tts_session", lambda family, **kw: _FakeSession()
    )

    models = [
        {"id": "voice1", "task": "tts", "family": "f5tts", "model_path": "/m.pt"}
    ]
    server = AudioTroveServer(_write_config(tmp_path, models))
    session = server._get_session("voice1")
    assert isinstance(session, _FakeSession)
    # Cached on second call (no extra load).
    server._get_session("voice1")
    assert loaded["count"] == 1


def test_get_session_asr(tmp_path, monkeypatch):
    from audiotrove.inference.server import AudioTroveServer
    import audiotrove.inference.asr as asr_mod

    class _FakeSession:
        def load(self):
            pass

    monkeypatch.setattr(asr_mod, "get_asr_session", lambda family, **kw: _FakeSession())
    models = [{"id": "asr1", "task": "asr", "family": "faster_whisper"}]
    server = AudioTroveServer(_write_config(tmp_path, models))
    assert isinstance(server._get_session("asr1"), _FakeSession)


def test_get_session_vc(tmp_path, monkeypatch):
    from audiotrove.inference.server import AudioTroveServer
    import audiotrove.inference.vc as vc_mod

    class _FakeSession:
        def load(self):
            pass

    monkeypatch.setattr(vc_mod, "get_vc_session", lambda family, **kw: _FakeSession())
    models = [{"id": "vc1", "task": "vc", "family": "seed_vc"}]
    server = AudioTroveServer(_write_config(tmp_path, models))
    assert isinstance(server._get_session("vc1"), _FakeSession)


def test_get_session_unknown_task_raises(tmp_path):
    from audiotrove.inference.server import AudioTroveServer

    models = [{"id": "x", "task": "bogus", "family": "f5tts"}]
    server = AudioTroveServer(_write_config(tmp_path, models))
    with pytest.raises(ValueError, match="Unknown task"):
        server._get_session("x")


def test_encode_wav_roundtrip():
    from audiotrove.inference.server import _encode_wav
    import io
    import wave

    audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    data = _encode_wav(audio, 16000)
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1


def test_first_asr_model():
    from audiotrove.inference.server import _first_asr_model

    config = {"models": [{"id": "t", "task": "tts"}, {"id": "a", "task": "asr"}]}
    assert _first_asr_model(config) == "a"


def test_first_asr_model_none_raises():
    from audiotrove.inference.server import _first_asr_model

    with pytest.raises(ValueError, match="No ASR model"):
        _first_asr_model({"models": [{"id": "t", "task": "tts"}]})


def test_build_app_registers_routes(tmp_path, monkeypatch):
    """build_app() should register all five routes on a fake aiohttp app."""
    from audiotrove.inference.server import AudioTroveServer

    routes = []

    class _Router:
        def add_get(self, path, handler):
            routes.append(("GET", path))

        def add_post(self, path, handler):
            routes.append(("POST", path))

    class _App:
        def __init__(self):
            self.router = _Router()

    fake_web = types.SimpleNamespace(Application=lambda: _App())
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = fake_web
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    server = AudioTroveServer(_write_config(tmp_path, []))
    server.build_app()
    paths = {p for _, p in routes}
    assert "/health" in paths
    assert "/v1/models" in paths
    assert "/v1/audio/speech" in paths
    assert "/v1/audio/transcriptions" in paths
    assert "/v1/tasks/run" in paths


def test_run_without_aiohttp_raises(tmp_path, monkeypatch):
    from audiotrove.inference.server import AudioTroveServer

    # Force ``import aiohttp`` to fail even when aiohttp is installed, so the
    # import-guard path is exercised without ever starting a blocking event
    # loop. Setting a module to None in sys.modules makes import raise
    # ImportError.
    monkeypatch.setitem(sys.modules, "aiohttp", None)

    server = AudioTroveServer(_write_config(tmp_path, []))
    with pytest.raises(ImportError, match="Server requires aiohttp"):
        server.run()
