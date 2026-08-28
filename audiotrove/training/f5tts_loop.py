"""Owned F5-TTS training loop.

AudioTrove drives F5-TTS fine-tuning itself rather than shelling out to the
upstream ``f5_tts.train`` entry point. Owning the loop lets us:

* feed the Rust-accelerated :class:`~audiotrove.training.dataloader.RustAudioDataset`,
* pick the best dtype per device (bf16 > fp16 > fp32) via
  :func:`~audiotrove.gpu.device.optimal_dtype`,
* emit :class:`~audiotrove.training.progress.TrainingProgressCallback` events for
  Rich progress in the CLI,
* checkpoint/resume on our own schedule.

Only :class:`f5_tts.model.CFM`/``DiT`` are imported from the optional
``f5-tts`` package (``pip install audiotrove[train-f5tts]``); the loop itself,
the mel transform and the optimiser use core torch, so the surrounding module
imports without any optional dependency installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# F5-TTS mel/model presets. Sample rate + mel settings match the upstream
# ``F5TTS_Base`` recipe (Vocos mel, 24 kHz, 100 channels).
_MEL_KWARGS = {
    "target_sample_rate": 24000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
}

_MODEL_PRESETS = {
    "F5TTS_Base": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    },
    "F5TTS_Small": {
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    },
}

_PIP_HINT = (
    "F5-TTS is not installed. Install it with:\n"
    "    pip install audiotrove[train-f5tts]"
)


def _build_f5tts_model(model_name: str, device):
    """Construct an F5-TTS ``CFM`` model on ``device``.

    Imports the optional ``f5_tts`` package lazily. Factored out so tests can
    monkeypatch it with a lightweight stub instead of the full model.

    Raises:
        RuntimeError: ``f5_tts`` is not installed.
    """
    try:
        from f5_tts.model import CFM, DiT
    except ImportError as exc:  # pragma: no cover - exercised only without f5_tts
        raise RuntimeError(_PIP_HINT) from exc

    preset = _MODEL_PRESETS.get(model_name, _MODEL_PRESETS["F5TTS_Base"])
    n_mel = _MEL_KWARGS["n_mel_channels"]

    transformer = DiT(**preset, mel_dim=n_mel)
    model = CFM(transformer=transformer, mel_spec_kwargs=dict(_MEL_KWARGS))
    return model.to(device)


def _build_mel_transform(device):
    """Return a callable mapping ``(B, T)`` audio to ``(B, T', n_mel)`` mel.

    Uses ``torchaudio`` when available (it is a core dependency); otherwise the
    caller is expected to have provided a model that consumes raw audio.
    """
    import torch  # noqa: F401 - ensures torch present before torchaudio
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=_MEL_KWARGS["target_sample_rate"],
        n_fft=_MEL_KWARGS["n_fft"],
        win_length=_MEL_KWARGS["win_length"],
        hop_length=_MEL_KWARGS["hop_length"],
        n_mels=_MEL_KWARGS["n_mel_channels"],
        power=1.0,
    ).to(device)

    def _transform(audio):
        # audio: (B, T) -> mel: (B, n_mel, T') -> (B, T', n_mel)
        spec = mel(audio)
        spec = spec.clamp(min=1e-5).log()
        return spec.transpose(1, 2)

    return _transform


def _forward_loss(model, mel, texts, mel_lens):
    """Call the F5-TTS model and extract a scalar loss.

    F5-TTS ``CFM.forward`` returns ``(loss, cond, pred)``; we also tolerate a
    plain-tensor or ``.loss``-attribute return so stub models used in tests work.
    """
    out = model(mel, text=texts, lens=mel_lens)
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "loss"):
        return out.loss
    return out


def _make_grad_scaler(enabled: bool):
    """Return a GradScaler, preferring the non-deprecated ``torch.amp`` API."""
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_f5tts(
    manifest_path: str,
    output_dir: str,
    model_name: str = "F5TTS_Base",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "auto",
    num_gpus: int = 1,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    save_every_n_epochs: int = 10,
    resume_from: Optional[str] = None,
    torch_threads: Optional[int] = None,
    dataloader_workers: int = 0,
    callback=None,
) -> dict:
    """Fine-tune F5-TTS on an AudioTrove manifest.

    Args:
        manifest_path: ``filelist.txt`` produced by the curation pipeline.
        output_dir: Directory for checkpoints (created if missing).
        model_name: One of the keys in :data:`_MODEL_PRESETS`.
        epochs: Number of epochs to train.
        batch_size: DataLoader batch size.
        learning_rate: AdamW learning rate.
        device: ``"auto"`` | ``"cuda"`` | ``"mps"`` | ``"cpu"``.
        num_gpus: Multi-GPU DDP is not implemented here yet; must be 1.
        mixed_precision: Enable autocast in the best dtype for the device.
        gradient_checkpointing: Enable activation checkpointing when the model
            exposes it (saves memory at some speed cost).
        save_every_n_epochs: Checkpoint cadence.
        resume_from: Optional checkpoint to restore model/optimizer/epoch from.
        torch_threads: Intra-op thread count for CPU training.
        dataloader_workers: ``num_workers`` for the DataLoader.
        callback: Optional :class:`TrainingProgressCallback`. A
            :class:`RichProgressCallback` is created when ``None``.

    Returns:
        ``{"epochs", "final_loss", "best_loss", "best_checkpoint",
        "final_checkpoint", "total_seconds"}``.

    Raises:
        RuntimeError: torch or f5_tts unavailable.
        NotImplementedError: ``num_gpus > 1``.
        ValueError: The manifest yields no usable clips.
    """
    try:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - torch is a core dep
        raise RuntimeError("torch is required to train F5-TTS") from exc

    if num_gpus > 1:
        raise NotImplementedError(
            "Multi-GPU F5-TTS training is not implemented in the owned loop yet; "
            "use num_gpus=1. Track this at the DDP helpers in audiotrove.training.gpu."
        )

    from audiotrove.gpu.device import get_device, optimal_dtype
    from audiotrove.training.dataloader import RustAudioDataset
    from audiotrove.training.gpu import apply_ipex, configure_threads
    from audiotrove.training.progress import RichProgressCallback

    configure_threads(torch_threads)

    dev = get_device(device)
    amp_dtype = optimal_dtype(dev)
    use_amp = bool(mixed_precision) and amp_dtype != torch.float32
    # GradScaler is only needed (and only valid) for fp16 on CUDA; bf16 doesn't
    # need loss scaling.
    use_scaler = use_amp and amp_dtype == torch.float16 and dev.type == "cuda"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    dataset = RustAudioDataset(
        manifest_path,
        target_sr=_MEL_KWARGS["target_sample_rate"],
    )
    if len(dataset) == 0:
        raise ValueError(
            f"Manifest {manifest_path} yielded no usable clips for F5-TTS training."
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        collate_fn=RustAudioDataset.collate_fn,
        drop_last=False,
    )
    steps_per_epoch = len(loader)

    # --- Model / optimiser ---
    model = _build_f5tts_model(model_name, dev)
    if gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    model, optimizer = apply_ipex(model, optimizer, dtype=amp_dtype if use_amp else None)
    scaler = _make_grad_scaler(use_scaler)
    mel_transform = _build_mel_transform(dev)

    start_epoch = 0
    if resume_from:
        start_epoch = _load_checkpoint(resume_from, model, optimizer, scheduler, dev)
        logger.info("Resumed F5-TTS training from %s at epoch %d", resume_from, start_epoch)

    if callback is None:
        callback = RichProgressCallback(total_epochs=epochs, total_steps=steps_per_epoch)

    hop = _MEL_KWARGS["hop_length"]
    best_loss = float("inf")
    best_ckpt: Optional[str] = None
    final_loss = float("nan")
    started = time.time()

    model.train()
    for epoch in range(start_epoch, epochs):
        callback.on_epoch_start(epoch + 1, epochs)
        running = 0.0
        seen = 0
        for step, batch in enumerate(loader, start=1):
            audio = batch["audio"].to(dev)
            mel_lens = (batch["audio_lengths"].to(dev) // hop).clamp(min=1)
            texts = batch["texts"]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                mel = mel_transform(audio)
                loss = _forward_loss(model, mel, texts, mel_lens)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_val = float(loss.detach().float().cpu())
            running += loss_val
            seen += 1
            callback.on_step(epoch + 1, step, steps_per_epoch, loss_val)

        scheduler.step()
        avg_loss = running / max(1, seen)
        final_loss = avg_loss
        callback.on_epoch_end(epoch + 1, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = str(out_dir / "model_best.pt")
            _save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch + 1, avg_loss)

        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
            _save_checkpoint(
                str(out_dir / f"model_epoch_{epoch + 1}.pt"),
                model, optimizer, scheduler, epoch + 1, avg_loss,
            )

    final_ckpt = str(out_dir / "model_final.pt")
    _save_checkpoint(final_ckpt, model, optimizer, scheduler, epochs, final_loss)

    metrics = {
        "epochs": epochs,
        "final_loss": final_loss,
        "best_loss": best_loss if best_ckpt else final_loss,
        "best_checkpoint": best_ckpt or final_ckpt,
        "final_checkpoint": final_ckpt,
        "total_seconds": round(time.time() - started, 3),
    }
    callback.on_train_end(metrics)
    return metrics


def _enable_gradient_checkpointing(model) -> None:
    """Best-effort activation of gradient checkpointing on the F5-TTS model."""
    transformer = getattr(model, "transformer", model)
    for attr in ("gradient_checkpointing", "use_checkpoint", "checkpoint_activations"):
        if hasattr(transformer, attr):
            try:
                setattr(transformer, attr, True)
                logger.info("Enabled gradient checkpointing via %s.", attr)
                return
            except Exception:  # noqa: BLE001
                continue
    enable_fn = getattr(transformer, "gradient_checkpointing_enable", None)
    if callable(enable_fn):
        try:
            enable_fn()
            logger.info("Enabled gradient checkpointing via gradient_checkpointing_enable().")
        except Exception:  # noqa: BLE001
            logger.debug("gradient_checkpointing_enable() failed", exc_info=True)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, loss) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def _load_checkpoint(path, model, optimizer, scheduler, device) -> int:
    """Restore state from ``path``; returns the epoch to resume from."""
    import torch

    # weights_only defaults to True on torch >= 2.6 and would reject the
    # optimizer/scheduler state in our own checkpoints. These are files we wrote,
    # so loading the full pickle is safe here.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:  # noqa: BLE001 - scheduler shape mismatch is non-fatal
                logger.debug("Could not restore scheduler state", exc_info=True)
        return int(ckpt.get("epoch", 0))
    # Bare state_dict checkpoint.
    model.load_state_dict(ckpt)
    return 0
