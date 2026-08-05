"""Matcha-TTS trainer wrapper."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from audiotrove.training.base import BaseTrainer

logger = logging.getLogger(__name__)


class MatchaTrainer(BaseTrainer):
    """Fine-tune Matcha-TTS.

    Converts the manifest to Matcha-TTS format and invokes training with a
    Hydra config override for device and batch size.
    """

    def validate_manifest(self) -> None:
        """Validate the manifest exists and holds enough clips."""
        path = Path(self.config.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 10:
            raise ValueError(f"Matcha-TTS needs >= 10 clips; found {len(lines)}")

    def train(self) -> dict:
        """Launch Matcha-TTS training as a subprocess. Returns a metrics dict."""
        from audiotrove.gpu.device import get_device

        device = get_device(self.config.device)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        accelerator = "gpu" if device.type == "cuda" else "cpu"
        cmd = [
            "python",
            "-m",
            "matcha.train",
            f"trainer.max_epochs={self.config.epochs}",
            f"data.batch_size={self.config.batch_size}",
            f"trainer.accelerator={accelerator}",
            f"trainer.devices={self.config.num_gpus}",
            f"paths.output_dir={self.config.output_dir}",
        ]
        logger.info("Launching Matcha-TTS training: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"returncode": result.returncode, "stdout": result.stdout}

    def export(self, output_path: str) -> str:
        """Copy the newest checkpoint to ``output_path``."""
        import shutil

        checkpoints = list(Path(self.config.output_dir).glob("**/*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No .ckpt checkpoints found under {self.config.output_dir}"
            )
        best = max(checkpoints, key=lambda p: p.stat().st_mtime)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, output_path)
        return output_path
