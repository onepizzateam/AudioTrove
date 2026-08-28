"""F5-TTS trainer wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

from audiotrove.training.base import BaseTrainer

logger = logging.getLogger(__name__)


class F5TTSTrainer(BaseTrainer):
    """Fine-tune F5-TTS on an AudioTrove-produced ``filelist.txt``."""

    def validate_manifest(self) -> None:
        """Validate the training manifest.

        Raises:
            FileNotFoundError: Manifest or a referenced WAV is missing.
            ValueError: Fewer than 10 usable clips.
        """
        path = Path(self.config.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 10:
            raise ValueError(f"F5-TTS needs >= 10 clips; found {len(lines)}")
        for line in lines:
            parts = line.split("\t")
            wav_path = parts[0]
            if not Path(wav_path).exists():
                raise FileNotFoundError(f"Missing WAV: {wav_path}")

    def train(self) -> dict:
        """Run F5-TTS training. Returns a metrics dict.

        Requires: pip install audiotrove[train-f5tts]
        """
        from f5_tts.train import train as f5_train

        from audiotrove.gpu.device import get_device

        device = get_device(self.config.device)
        cfg = {
            "dataset_path": str(Path(self.config.manifest_path).parent),
            "output_dir": self.config.output_dir,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "device": str(device),
            "mixed_precision": self.config.mixed_precision and device.type == "cuda",
        }
        if self.config.num_gpus > 1:
            cfg["num_gpus"] = self.config.num_gpus  # triggers DDP inside f5_train
        return f5_train(**cfg)

    def export(self, output_path: str) -> str:
        """Copy the newest checkpoint under output_dir to ``output_path``."""
        import shutil

        checkpoints = list(Path(self.config.output_dir).glob("**/*.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No .pt checkpoints found under {self.config.output_dir}"
            )
        best = max(checkpoints, key=lambda p: p.stat().st_mtime)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, output_path)
        return output_path
