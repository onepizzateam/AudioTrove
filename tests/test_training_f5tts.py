"""Tests for the F5-TTS trainer and training factory."""

import sys
import types

import pytest


def _make_manifest(tmp_path, n_clips=10, make_wavs=True):
    """Create a filelist.txt with n_clips tab-separated lines."""
    lines = []
    for i in range(n_clips):
        wav = tmp_path / f"clip{i}.wav"
        if make_wavs:
            wav.write_bytes(b"RIFF")  # existence is all validate checks
        lines.append(f"{wav}\tspeaker\ttranscript {i}")
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("\n".join(lines))
    return manifest


def _config(manifest, out_dir, **kw):
    from audiotrove.training.base import TrainingConfig

    return TrainingConfig(
        manifest_path=str(manifest),
        output_dir=str(out_dir),
        model_name="f5tts",
        **kw,
    )


def test_training_config_defaults(tmp_path):
    from audiotrove.training.base import TrainingConfig

    cfg = TrainingConfig(
        manifest_path="m.txt", output_dir="out", model_name="f5tts"
    )
    assert cfg.epochs == 100
    assert cfg.batch_size == 16
    assert cfg.device == "auto"
    assert cfg.num_gpus == 1
    assert cfg.mixed_precision is True
    assert cfg.resume_from is None
    assert cfg.gradient_checkpointing is True
    assert cfg.save_every_n_epochs == 10
    assert cfg.torch_threads is None
    assert cfg.dataloader_workers == 0



def test_validate_manifest_missing_raises(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    trainer = F5TTSTrainer(_config(tmp_path / "nope.txt", tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        trainer.validate_manifest()


def test_validate_manifest_too_few_clips_raises(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    manifest = _make_manifest(tmp_path, n_clips=5)
    trainer = F5TTSTrainer(_config(manifest, tmp_path / "out"))
    with pytest.raises(ValueError, match=">= 10 clips"):
        trainer.validate_manifest()


def test_validate_manifest_missing_wav_raises(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    manifest = _make_manifest(tmp_path, n_clips=10, make_wavs=False)
    trainer = F5TTSTrainer(_config(manifest, tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="Missing WAV"):
        trainer.validate_manifest()


def test_validate_manifest_passes(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    manifest = _make_manifest(tmp_path, n_clips=10)
    trainer = F5TTSTrainer(_config(manifest, tmp_path / "out"))
    trainer.validate_manifest()  # should not raise


def test_train_invokes_owned_loop(tmp_path, monkeypatch):
    """train() should delegate to the owned train_f5tts loop, forwarding config."""
    import audiotrove.training.f5tts_loop as loop_mod
    from audiotrove.training.f5tts import F5TTSTrainer

    captured = {}

    def _fake_train(**cfg):
        captured.update(cfg)
        return {"best_loss": 0.04}

    monkeypatch.setattr(loop_mod, "train_f5tts", _fake_train)

    manifest = _make_manifest(tmp_path, n_clips=10)
    trainer = F5TTSTrainer(_config(manifest, tmp_path / "out", device="cpu"))
    metrics = trainer.train()
    assert metrics["best_loss"] == 0.04
    assert captured["device"] == "cpu"
    assert captured["epochs"] == 100
    assert captured["manifest_path"] == str(manifest)
    # The generic framework key "f5tts" is mapped to the base model preset.
    assert captured["model_name"] == "F5TTS_Base"
    # New TrainingConfig fields must be forwarded to the loop.
    assert captured["gradient_checkpointing"] is True
    assert captured["save_every_n_epochs"] == 10


def test_train_forwards_num_gpus(tmp_path, monkeypatch):
    import audiotrove.training.f5tts_loop as loop_mod
    from audiotrove.training.f5tts import F5TTSTrainer

    captured = {}

    monkeypatch.setattr(
        loop_mod, "train_f5tts", lambda **cfg: captured.update(cfg) or {}
    )

    manifest = _make_manifest(tmp_path, n_clips=10)
    trainer = F5TTSTrainer(_config(manifest, tmp_path / "out", device="cpu", num_gpus=4))
    trainer.train()
    assert captured["num_gpus"] == 4



def test_export_copies_newest_checkpoint(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    out_dir = tmp_path / "out"
    (out_dir / "ckpts").mkdir(parents=True)
    old = out_dir / "ckpts" / "old.pt"
    new = out_dir / "ckpts" / "new.pt"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    # Make `new` newer.
    import os
    import time

    os.utime(old, (time.time() - 100, time.time() - 100))

    trainer = F5TTSTrainer(_config(tmp_path / "filelist.txt", out_dir))
    dest = tmp_path / "final.pt"
    result = trainer.export(str(dest))
    assert result == str(dest)
    assert dest.read_bytes() == b"new"


def test_export_no_checkpoints_raises(tmp_path):
    from audiotrove.training.f5tts import F5TTSTrainer

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    trainer = F5TTSTrainer(_config(tmp_path / "filelist.txt", out_dir))
    with pytest.raises(FileNotFoundError, match="No .pt checkpoints"):
        trainer.export(str(tmp_path / "final.pt"))
