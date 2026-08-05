"""Training base classes and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for a TTS fine-tuning run.

    Attributes:
        manifest_path: filelist.txt or metadata.csv produced by TTSManifestExporter.
        output_dir: Directory for checkpoints and the final model.
        model_name: Base model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        device: ``"auto"`` | ``"cuda"`` | ``"mps"`` | ``"cpu"``.
        num_gpus: Number of GPUs (DDP when > 1).
        mixed_precision: Use fp16 training on CUDA when available.
        resume_from: Optional checkpoint path to resume from.
    """

    manifest_path: str
    output_dir: str
    model_name: str
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    device: str = "auto"
    num_gpus: int = 1
    mixed_precision: bool = True
    resume_from: Optional[str] = None


class BaseTrainer(ABC):
    """Base class for all AudioTrove training wrappers."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    @abstractmethod
    def validate_manifest(self) -> None:
        """Raise if the manifest is missing or malformed."""

    @abstractmethod
    def train(self) -> dict:
        """Run training. Returns a metrics dict."""

    @abstractmethod
    def export(self, output_path: str) -> str:
        """Export the final model to ``output_path``. Returns the path."""
