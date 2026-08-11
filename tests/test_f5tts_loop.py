"""Tests for the owned F5-TTS training loop.

The real F5-TTS model (``f5_tts.model.CFM``/``DiT``) is optional and heavy, so
these tests monkeypatch the two seams the loop exposes for exactly this purpose
-- ``_build_f5tts_model`` and ``_build_mel_transform`` -- with a tiny torch
model. That lets us exercise the loop mechanics (optimiser step, checkpointing,
resume, metrics, callback hooks) without the dependency.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

import audiotrove.training.f5tts_loop as loop_mod
from audiotrove.training.f5tts_loop import train_f5tts
from audiotrove.training.progress import TrainingProgressCallback

N_MEL = 100


class _TinyModel(torch.nn.Module):
    """Minimal stand-in for the F5-TTS CFM model.

    Consumes a ``(B, T', n_mel)`` mel and returns a scalar loss so the loop's
    backward/step path runs against real autograd.
    """

    def __init__(self, n_mel=N_MEL):
        super().__init__()
        self.proj = torch.nn.Linear(n_mel, 1)

    def forward(self, mel, text=None, lens=None):
        return self.proj(mel).pow(2).mean()


def _fake_mel_transform(_device):
    def _transform(audio):
        # (B, T) -> (B, T', n_mel); depends on audio so batches differ.
        b = audio.shape[0]
        base = audio.mean(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        return base.expand(b, 8, N_MEL).contiguous()

    return _transform


@pytest.fixture
def _patched_loop(monkeypatch):
    monkeypatch.setattr(loop_mod, "_build_f5tts_model", lambda name, dev: _TinyModel().to(dev))
    monkeypatch.setattr(loop_mod, "_build_mel_transform", _fake_mel_transform)


def _make_dataset(tmp_path, n=10, sr=24000, seconds=2.0):
    lines = []
    for i in range(n):
        wav = tmp_path / f"clip{i}.wav"
        t = np.arange(int(seconds * sr)) / sr
        sf.write(str(wav), (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
        lines.append(f"{wav}\t{seconds}\ttranscript {i}")
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def test_train_loop_runs_and_checkpoints(tmp_path, _patched_loop):
    manifest = _make_dataset(tmp_path)
    out_dir = tmp_path / "model"

    metrics = train_f5tts(
        manifest_path=str(manifest),
        output_dir=str(out_dir),
        model_name="F5TTS_Base",
        epochs=2,
        batch_size=4,
        device="cpu",
        mixed_precision=False,  # deterministic fp32 path
        save_every_n_epochs=1,
    )

    assert set(metrics) >= {
        "epochs",
        "final_loss",
        "best_loss",
        "best_checkpoint",
        "final_checkpoint",
        "total_seconds",
    }
    assert metrics["epochs"] == 2
    assert (out_dir / "model_final.pt").exists()
    assert (out_dir / "model_best.pt").exists()
    assert (out_dir / "model_epoch_1.pt").exists()


def test_train_loop_invokes_callback_hooks(tmp_path, _patched_loop):
    manifest = _make_dataset(tmp_path)

    class CountingCallback(TrainingProgressCallback):
        def __init__(self):
            self.epochs_started = 0
            self.steps = 0
            self.ended = False

        def on_epoch_start(self, epoch, total_epochs):
            self.epochs_started += 1

        def on_step(self, epoch, step, total_steps, loss):
            self.steps += 1

        def on_train_end(self, metrics):
            self.ended = True

    cb = CountingCallback()
    train_f5tts(
        manifest_path=str(manifest),
        output_dir=str(tmp_path / "model"),
        epochs=2,
        batch_size=4,
        device="cpu",
        mixed_precision=False,
        callback=cb,
    )
    assert cb.epochs_started == 2
    assert cb.steps > 0
    assert cb.ended is True


def test_train_loop_resume_starts_from_checkpoint(tmp_path, _patched_loop):
    manifest = _make_dataset(tmp_path)
    out_dir = tmp_path / "model"

    first = train_f5tts(
        manifest_path=str(manifest),
        output_dir=str(out_dir),
        epochs=1,
        batch_size=4,
        device="cpu",
        mixed_precision=False,
        save_every_n_epochs=1,
    )

    class CountingCallback(TrainingProgressCallback):
        def __init__(self):
            self.epochs_started = 0

        def on_epoch_start(self, epoch, total_epochs):
            self.epochs_started += 1

    cb = CountingCallback()
    second = train_f5tts(
        manifest_path=str(manifest),
        output_dir=str(out_dir),
        epochs=2,
        batch_size=4,
        device="cpu",
        mixed_precision=False,
        resume_from=first["final_checkpoint"],
        callback=cb,
    )
    # Resuming from epoch 1 and asking for 2 epochs means exactly one more epoch.
    assert cb.epochs_started == 1
    assert second["epochs"] == 2


def test_train_loop_rejects_multi_gpu(tmp_path, _patched_loop):
    manifest = _make_dataset(tmp_path)
    with pytest.raises(NotImplementedError, match="Multi-GPU"):
        train_f5tts(
            manifest_path=str(manifest),
            output_dir=str(tmp_path / "model"),
            epochs=1,
            device="cpu",
            num_gpus=2,
        )


def test_train_loop_empty_manifest_raises(tmp_path, _patched_loop):
    # Durations below min_duration (1.0) are all filtered out -> zero clips.
    lines = [f"{tmp_path / f'c{i}.wav'}\t0.1\ttext" for i in range(5)]
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="no usable clips"):
        train_f5tts(
            manifest_path=str(manifest),
            output_dir=str(tmp_path / "model"),
            epochs=1,
            device="cpu",
        )
