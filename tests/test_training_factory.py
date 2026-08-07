"""Tests for the training factory and non-F5 trainer wrappers."""

import sys
import types

import pytest


def _config(manifest, out_dir, **kw):
    from audiotrove.training.base import TrainingConfig

    return TrainingConfig(
        manifest_path=str(manifest),
        output_dir=str(out_dir),
        model_name="test",
        **kw,
    )


def _make_manifest(tmp_path, n_clips=10):
    lines = [f"clip{i}.wav\tspeaker\ttext {i}" for i in range(n_clips)]
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("\n".join(lines))
    return manifest


def test_get_trainer_dispatches():
    from audiotrove.training import get_trainer
    from audiotrove.training.base import TrainingConfig

    cfg = TrainingConfig(manifest_path="m", output_dir="o", model_name="x")
    assert get_trainer("f5tts", cfg).__class__.__name__ == "F5TTSTrainer"
    assert get_trainer("styletts2", cfg).__class__.__name__ == "StyleTTS2Trainer"
    assert get_trainer("piper", cfg).__class__.__name__ == "PiperTrainer"
    assert get_trainer("matcha", cfg).__class__.__name__ == "MatchaTrainer"


def test_get_trainer_case_insensitive():
    from audiotrove.training import get_trainer
    from audiotrove.training.base import TrainingConfig

    cfg = TrainingConfig(manifest_path="m", output_dir="o", model_name="x")
    assert get_trainer("F5TTS", cfg).__class__.__name__ == "F5TTSTrainer"


def test_get_trainer_unknown_raises():
    from audiotrove.training import get_trainer
    from audiotrove.training.base import TrainingConfig

    cfg = TrainingConfig(manifest_path="m", output_dir="o", model_name="x")
    with pytest.raises(ValueError, match="Unknown training framework"):
        get_trainer("bogus", cfg)


@pytest.mark.parametrize(
    "module,cls_name,needs",
    [
        ("audiotrove.training.styletts2", "StyleTTS2Trainer", "10 clips"),
        ("audiotrove.training.piper", "PiperTrainer", "10 clips"),
        ("audiotrove.training.matcha", "MatchaTrainer", "10 clips"),
    ],
)
def test_subprocess_trainers_validate_manifest(tmp_path, module, cls_name, needs):
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    # Missing manifest
    trainer = cls(_config(tmp_path / "nope.txt", tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        trainer.validate_manifest()

    # Too few clips
    manifest = _make_manifest(tmp_path, n_clips=3)
    trainer = cls(_config(manifest, tmp_path / "out"))
    with pytest.raises(ValueError):
        trainer.validate_manifest()

    # Enough clips
    manifest = _make_manifest(tmp_path, n_clips=10)
    trainer = cls(_config(manifest, tmp_path / "out"))
    trainer.validate_manifest()


def test_styletts2_train_runs_subprocess(tmp_path, monkeypatch):
    from audiotrove.training.styletts2 import StyleTTS2Trainer

    calls = {}

    def _fake_run(cmd, capture_output, text, check):
        calls["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("subprocess.run", _fake_run)
    trainer = StyleTTS2Trainer(_config(tmp_path / "m.txt", tmp_path / "out", device="cpu"))
    result = trainer.train()
    assert result["returncode"] == 0
    assert "styletts2.train" in calls["cmd"]


def test_piper_train_runs_subprocess(tmp_path, monkeypatch):
    from audiotrove.training.piper import PiperTrainer

    calls = {}

    def _fake_run(cmd, capture_output, text, check):
        calls["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("subprocess.run", _fake_run)
    trainer = PiperTrainer(_config(tmp_path / "m.txt", tmp_path / "out", device="cpu"))
    result = trainer.train()
    assert result["returncode"] == 0
    assert "piper_train" in calls["cmd"]


def test_matcha_train_runs_subprocess(tmp_path, monkeypatch):
    from audiotrove.training.matcha import MatchaTrainer

    calls = {}

    def _fake_run(cmd, capture_output, text, check):
        calls["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr("subprocess.run", _fake_run)
    trainer = MatchaTrainer(_config(tmp_path / "m.txt", tmp_path / "out", device="cpu"))
    result = trainer.train()
    assert result["returncode"] == 0
    assert "matcha.train" in calls["cmd"]


def test_styletts2_export(tmp_path):
    from audiotrove.training.styletts2 import StyleTTS2Trainer

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "model.pth").write_bytes(b"x")
    trainer = StyleTTS2Trainer(_config(tmp_path / "m.txt", out_dir))
    dest = tmp_path / "final.pth"
    assert trainer.export(str(dest)) == str(dest)
    assert dest.exists()


def test_piper_export_no_ckpt_raises(tmp_path):
    from audiotrove.training.piper import PiperTrainer

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    trainer = PiperTrainer(_config(tmp_path / "m.txt", out_dir))
    with pytest.raises(FileNotFoundError, match="No .ckpt"):
        trainer.export(str(tmp_path / "final.ckpt"))


def test_matcha_export(tmp_path):
    from audiotrove.training.matcha import MatchaTrainer

    out_dir = tmp_path / "out"
    (out_dir / "sub").mkdir(parents=True)
    (out_dir / "sub" / "last.ckpt").write_bytes(b"x")
    trainer = MatchaTrainer(_config(tmp_path / "m.txt", out_dir))
    dest = tmp_path / "final.ckpt"
    assert trainer.export(str(dest)) == str(dest)
    assert dest.exists()


def test_ddp_helpers_use_torch(monkeypatch):
    """launch_ddp/setup_ddp/cleanup_ddp should delegate to torch APIs.

    We patch the real ``torch.multiprocessing`` / ``torch.distributed`` module
    attributes directly (rather than replacing the sys.modules entries) so the
    ``import torch.x as y`` statements inside the helpers pick up the fakes and
    never launch real subprocesses.
    """
    torch_mp = pytest.importorskip("torch.multiprocessing")
    torch_dist = pytest.importorskip("torch.distributed")
    from audiotrove.training import gpu as gpu_mod

    spawned = {}

    def _fake_spawn(fn, args, nprocs, join):
        spawned.update({"nprocs": nprocs, "join": join})

    dist_calls = {}

    def _fake_init(backend, rank, world_size):
        dist_calls.update({"backend": backend, "rank": rank, "world_size": world_size})

    def _fake_destroy():
        dist_calls["destroyed"] = True

    monkeypatch.setattr(torch_mp, "spawn", _fake_spawn)
    monkeypatch.setattr(torch_dist, "init_process_group", _fake_init)
    monkeypatch.setattr(torch_dist, "destroy_process_group", _fake_destroy)

    def _noop_train(*a):
        return None

    gpu_mod.launch_ddp(_noop_train, num_gpus=2)
    assert spawned["nprocs"] == 2

    gpu_mod.setup_ddp(rank=0, world_size=2, backend="gloo")
    assert dist_calls["backend"] == "gloo"

    gpu_mod.cleanup_ddp()
    assert dist_calls["destroyed"] is True
