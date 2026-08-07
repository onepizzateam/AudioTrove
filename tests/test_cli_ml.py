"""CLI tests for the new ML commands: train / infer / run / serve.

These exercise ``audiotrove/cli/main.py`` without pulling in any heavy model
dependencies by monkeypatching the lazily-imported factories.
"""

import types

import numpy as np
import pytest
from click.testing import CliRunner

from audiotrove.cli.main import cli
from audiotrove.inference.base import InferenceResult


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------
class _FakeTrainer:
    def __init__(self, config):
        self.config = config
        self.validated = False

    def validate_manifest(self):
        self.validated = True

    def train(self):
        return {"loss": 0.01}

    def export(self, output_path):
        return output_path


class _FakeSession:
    def __init__(self, audio=None, text=None):
        self._audio = audio
        self._text = text
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()

    def run(self, **kwargs):
        return InferenceResult(
            audio=self._audio, text=self._text, sample_rate=22050, metadata={}
        )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def test_train_command(runner, tmp_path, monkeypatch):
    import audiotrove.training as training_mod

    manifest = tmp_path / "filelist.txt"
    manifest.write_text("x\n")
    out = tmp_path / "out"

    created = {}

    def fake_get_trainer(framework, config):
        t = _FakeTrainer(config)
        created["trainer"] = t
        created["framework"] = framework
        return t

    monkeypatch.setattr(training_mod, "get_trainer", fake_get_trainer)

    result = runner.invoke(
        cli,
        ["train", str(manifest), str(out), "--framework", "f5tts", "--epochs", "3"],
    )
    assert result.exit_code == 0, result.output
    assert created["framework"] == "f5tts"
    assert created["trainer"].validated is True
    assert created["trainer"].config.epochs == 3
    assert "Training complete" in result.output


def test_train_import_error_becomes_click_exception(runner, tmp_path, monkeypatch):
    import audiotrove.training as training_mod

    manifest = tmp_path / "filelist.txt"
    manifest.write_text("x\n")

    class _Boom(_FakeTrainer):
        def train(self):
            raise ImportError("pip install audiotrove[train-f5tts]")

    monkeypatch.setattr(training_mod, "get_trainer", lambda f, c: _Boom(c))

    result = runner.invoke(cli, ["train", str(manifest), str(tmp_path / "o")])
    assert result.exit_code != 0
    assert "train-f5tts" in result.output


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------
def test_infer_tts_writes_audio(runner, tmp_path, monkeypatch):
    import audiotrove.inference.tts as tts_mod

    out = tmp_path / "out.wav"
    session = _FakeSession(audio=np.zeros(2048, dtype=np.float32))
    monkeypatch.setattr(tts_mod, "get_tts_session", lambda family, **kw: session)

    result = runner.invoke(
        cli,
        ["infer", "--task", "tts", "--text", "hi", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Wrote" in result.output


def test_infer_tts_no_out_path(runner, monkeypatch):
    import audiotrove.inference.tts as tts_mod

    session = _FakeSession(audio=np.zeros(16, dtype=np.float32))
    monkeypatch.setattr(tts_mod, "get_tts_session", lambda family, **kw: session)

    result = runner.invoke(cli, ["infer", "--task", "tts", "--text", "hi"])
    assert result.exit_code == 0, result.output
    assert "not written" in result.output


def test_infer_asr_prints_transcript(runner, tmp_path, monkeypatch):
    import audiotrove.inference.asr as asr_mod

    session = _FakeSession(text="hello world")
    monkeypatch.setattr(asr_mod, "get_asr_session", lambda family, **kw: session)

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"\x00")
    result = runner.invoke(
        cli, ["infer", "--task", "asr", "--audio", str(audio)]
    )
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output


def test_infer_vc_writes_audio(runner, tmp_path, monkeypatch):
    import audiotrove.inference.vc as vc_mod

    out = tmp_path / "vc.wav"
    session = _FakeSession(audio=np.zeros(1024, dtype=np.float32))
    monkeypatch.setattr(vc_mod, "get_vc_session", lambda family, **kw: session)

    result = runner.invoke(
        cli,
        [
            "infer",
            "--task",
            "vc",
            "--audio",
            str(tmp_path / "src.wav"),
            "--voice-ref",
            str(tmp_path / "ref.wav"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_infer_import_error_becomes_click_exception(runner, monkeypatch):
    import audiotrove.inference.tts as tts_mod

    def boom(family, **kw):
        raise ImportError("pip install audiotrove[infer]")

    monkeypatch.setattr(tts_mod, "get_tts_session", boom)
    result = runner.invoke(cli, ["infer", "--task", "tts", "--text", "hi"])
    assert result.exit_code != 0
    assert "infer" in result.output


def test_infer_requires_task(runner):
    result = runner.invoke(cli, ["infer"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run (e2e)
# ---------------------------------------------------------------------------
def test_run_command(runner, tmp_path, monkeypatch):
    import audiotrove.pipelines.e2e as e2e_mod

    captured = {}

    def fake_e2e(config):
        captured["config"] = config
        return {
            "curate_summary": {"kept": 5, "filtered": 2},
            "train_summary": {"loss": 0.1},
            "validation_audio_path": str(tmp_path / "validation.wav"),
        }

    monkeypatch.setattr(e2e_mod, "e2e_pipeline", fake_e2e)

    result = runner.invoke(
        cli,
        ["run", str(tmp_path / "in"), str(tmp_path / "out"), "--no-train"],
    )
    assert result.exit_code == 0, result.output
    assert captured["config"].train is False
    assert "Curated: 5 clips kept" in result.output
    assert "Done." in result.output


def test_run_import_error_becomes_click_exception(runner, tmp_path, monkeypatch):
    import audiotrove.pipelines.e2e as e2e_mod

    def boom(config):
        raise ImportError("pip install audiotrove[train-f5tts]")

    monkeypatch.setattr(e2e_mod, "e2e_pipeline", boom)
    result = runner.invoke(cli, ["run", str(tmp_path / "in"), str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "train-f5tts" in result.output


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def test_serve_command(runner, tmp_path, monkeypatch):
    import audiotrove.inference.server as server_mod

    started = {}

    class _FakeServer:
        def __init__(self, config_path):
            started["config_path"] = config_path
            self.host = None
            self.port = None

        def run(self):
            started["host"] = self.host
            started["port"] = self.port

    monkeypatch.setattr(server_mod, "AudioTroveServer", _FakeServer)

    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    result = runner.invoke(
        cli, ["serve", str(cfg), "--host", "0.0.0.0", "--port", "9000"]
    )
    assert result.exit_code == 0, result.output
    assert started["host"] == "0.0.0.0"
    assert started["port"] == 9000


def test_serve_import_error_becomes_click_exception(runner, tmp_path, monkeypatch):
    import audiotrove.inference.server as server_mod

    class _FakeServer:
        def __init__(self, config_path):
            self.host = None
            self.port = None

        def run(self):
            raise ImportError("pip install aiohttp")

    monkeypatch.setattr(server_mod, "AudioTroveServer", _FakeServer)

    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    result = runner.invoke(cli, ["serve", str(cfg)])
    assert result.exit_code != 0
    assert "aiohttp" in result.output
