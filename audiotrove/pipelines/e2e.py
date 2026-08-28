"""End-to-end pipeline: curate -> transcribe -> [train] -> [validate]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

_SUPPORTED_FRAMEWORKS = ("f5tts", "styletts2", "piper", "matcha")


@dataclass
class E2EConfig:
    """Configuration for the one-command end-to-end pipeline."""

    input_path: str
    output_path: str
    # Curation
    min_duration: float = 2.0
    max_duration: float = 15.0
    snr_min: float = 20.0
    extensions: Optional[list[str]] = None
    workers: int = 1
    # Device
    device: str = "auto"
    # Transcription
    transcribe: bool = True
    whisper_model: str = "base"
    # Training
    train: bool = False
    train_framework: Literal["f5tts", "styletts2", "piper", "matcha"] = "f5tts"
    epochs: int = 100
    batch_size: int = 16
    num_gpus: int = 1
    # Inference validation (optional smoke test after training)
    validate_inference: bool = False
    validate_text: str = "Hello, this is a test of the trained voice."
    validate_voice_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if self.train_framework not in _SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported train_framework: {self.train_framework!r}. "
                f"Choose from {list(_SUPPORTED_FRAMEWORKS)}"
            )


def e2e_pipeline(config: E2EConfig) -> dict:
    """Full pipeline: curate -> transcribe -> [train] -> [validate].

    Returns:
        Summary dict with keys ``curate_summary``, ``train_summary`` and
        ``validation_audio_path``.
    """
    from pathlib import Path

    from audiotrove.pipelines.tts import tts_pipeline

    output = Path(config.output_path)

    # --- Step 1: Curate ---
    curate_summary = tts_pipeline(
        input_path=config.input_path,
        output_path=str(output / "curated"),
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        snr_min=config.snr_min,
        extensions=config.extensions or ["wav"],
        workers=config.workers,
        transcribe=config.transcribe,
        whisper_model=config.whisper_model,
        device=config.device,
    )

    if curate_summary["kept"] == 0:
        raise RuntimeError(
            "Curation produced zero clips. Check input audio and filters."
        )

    result: dict = {
        "curate_summary": curate_summary,
        "train_summary": None,
        "validation_audio_path": None,
    }

    # --- Step 2: Train ---
    if config.train:
        from audiotrove.training import get_trainer
        from audiotrove.training.base import TrainingConfig

        manifest = str(output / "curated" / "filelist.txt")
        train_output = str(output / "model")

        trainer = get_trainer(
            config.train_framework,
            TrainingConfig(
                manifest_path=manifest,
                output_dir=train_output,
                model_name=config.train_framework,
                epochs=config.epochs,
                batch_size=config.batch_size,
                device=config.device,
                num_gpus=config.num_gpus,
            ),
        )
        trainer.validate_manifest()
        train_summary = trainer.train()
        model_path = trainer.export(str(output / "model" / "final.pt"))
        result["train_summary"] = train_summary

        # --- Step 3: Validate (optional smoke test) ---
        if config.validate_inference:
            import soundfile as sf

            from audiotrove.inference.tts import get_tts_session

            session = get_tts_session(
                config.train_framework,
                model_path=model_path,
                device=config.device,
                voice_ref=config.validate_voice_ref,
            )
            with session:
                r = session.run(text=config.validate_text)
            val_path = str(output / "validation.wav")
            sf.write(val_path, r.audio, r.sample_rate)
            result["validation_audio_path"] = val_path

    return result
