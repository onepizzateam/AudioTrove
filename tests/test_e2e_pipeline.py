"""Tests for the end-to-end pipeline orchestration."""

import pytest


def test_e2e_config_defaults():
    from audiotrove.pipelines.e2e import E2EConfig

    cfg = E2EConfig(input_path="in", output_path="out")
    assert cfg.train is False
    assert cfg.train_framework == "f5tts"
    assert cfg.device == "auto"


def test_e2e_config_validates_framework():
    from audiotrove.pipelines.e2e import E2EConfig

    with pytest.raises(ValueError, match="Unsupported train_framework"):
        E2EConfig(input_path="in", output_path="out", train_framework="bogus")


@pytest.mark.parametrize("framework", ["f5tts", "styletts2", "piper", "matcha"])
def test_e2e_config_accepts_supported_frameworks(framework):
    from audiotrove.pipelines.e2e import E2EConfig

    cfg = E2EConfig(input_path="in", output_path="out", train_framework=framework)
    assert cfg.train_framework == framework


def test_e2e_pipeline_curate_only(tmp_path, monkeypatch):
    """With train=False, e2e_pipeline should just return the curate summary."""
    import audiotrove.pipelines.tts as tts_mod
    from audiotrove.pipelines.e2e import E2EConfig, e2e_pipeline

    def _fake_tts_pipeline(**kwargs):
        return {"kept": 42, "filtered": 3, "total_duration_seconds": 100.0}

    monkeypatch.setattr(tts_mod, "tts_pipeline", _fake_tts_pipeline)

    cfg = E2EConfig(
        input_path=str(tmp_path / "in"),
        output_path=str(tmp_path / "out"),
        train=False,
    )
    result = e2e_pipeline(cfg)
    assert result["curate_summary"]["kept"] == 42
    assert result["train_summary"] is None
    assert result["validation_audio_path"] is None


def test_e2e_pipeline_zero_clips_raises(tmp_path, monkeypatch):
    import audiotrove.pipelines.tts as tts_mod
    from audiotrove.pipelines.e2e import E2EConfig, e2e_pipeline

    monkeypatch.setattr(
        tts_mod,
        "tts_pipeline",
        lambda **kw: {"kept": 0, "filtered": 5, "total_duration_seconds": 0.0},
    )
    cfg = E2EConfig(input_path=str(tmp_path / "in"), output_path=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="zero clips"):
        e2e_pipeline(cfg)


def test_e2e_pipeline_with_training(tmp_path, monkeypatch):
    """With train=True, e2e_pipeline should drive the trainer factory."""
    import audiotrove.pipelines.tts as tts_mod
    import audiotrove.training as training_mod
    from audiotrove.pipelines.e2e import E2EConfig, e2e_pipeline

    monkeypatch.setattr(
        tts_mod,
        "tts_pipeline",
        lambda **kw: {"kept": 20, "filtered": 1, "total_duration_seconds": 60.0},
    )

    events = []

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config

        def validate_manifest(self):
            events.append("validate")

        def train(self):
            events.append("train")
            return {"best_loss": 0.01}

        def export(self, path):
            events.append("export")
            return path

    monkeypatch.setattr(training_mod, "get_trainer", lambda fw, cfg: _FakeTrainer(cfg))

    cfg = E2EConfig(
        input_path=str(tmp_path / "in"),
        output_path=str(tmp_path / "out"),
        train=True,
        train_framework="f5tts",
    )
    result = e2e_pipeline(cfg)
    assert events == ["validate", "train", "export"]
    assert result["train_summary"]["best_loss"] == 0.01
    assert result["validation_audio_path"] is None


def test_e2e_pipeline_with_validation(tmp_path, monkeypatch):
    """validate_inference=True should synthesize and write validation.wav."""
    import numpy as np

    import audiotrove.pipelines.tts as tts_mod
    import audiotrove.training as training_mod
    import audiotrove.inference.tts as inf_tts_mod
    from audiotrove.inference.base import InferenceResult
    from audiotrove.pipelines.e2e import E2EConfig, e2e_pipeline

    monkeypatch.setattr(
        tts_mod,
        "tts_pipeline",
        lambda **kw: {"kept": 20, "filtered": 0, "total_duration_seconds": 60.0},
    )

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config

        def validate_manifest(self):
            pass

        def train(self):
            return {}

        def export(self, path):
            return path

    monkeypatch.setattr(training_mod, "get_trainer", lambda fw, cfg: _FakeTrainer(cfg))

    class _FakeSession:
        def load(self):
            pass

        def unload(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def run(self, text):
            return InferenceResult(
                audio=np.zeros(16, dtype=np.float32), sample_rate=22050
            )

    monkeypatch.setattr(
        inf_tts_mod, "get_tts_session", lambda fw, **kw: _FakeSession()
    )

    cfg = E2EConfig(
        input_path=str(tmp_path / "in"),
        output_path=str(tmp_path / "out"),
        train=True,
        validate_inference=True,
    )
    # Ensure output dir exists so soundfile can write validation.wav
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    result = e2e_pipeline(cfg)
    assert result["validation_audio_path"] is not None
    from pathlib import Path

    assert Path(result["validation_audio_path"]).exists()
