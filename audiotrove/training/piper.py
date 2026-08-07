"""Piper trainer wrapper."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from audiotrove.training.base import BaseTrainer

logger = logging.getLogger(__name__)


class PiperTrainer(BaseTrainer):
    """Fine-tune Piper via the ``piper_train`` CLI.

    Converts ``filelist.txt`` to Piper's JSON manifest, then invokes
    ``python -m piper_train`` as a subprocess with GPU env vars.
    """

    def validate_manifest(self) -> None:
        """Validate the manifest exists and holds enough clips."""
        path = Path(self.config.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 10:
            raise ValueError(f"Piper needs >= 10 clips; found {len(lines)}")

    def train(self) -> dict:
        """Launch Piper training as a subprocess. Returns a metrics dict."""
        from audiotrove.gpu.device import get_device

        device = get_device(self.config.device)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        accelerator = "gpu" if device.type == "cuda" else "cpu"
        cmd = [
            "python",
            "-m",
            "piper_train",
            "--dataset-dir",
            str(Path(self.config.manifest_path).parent),
            "--accelerator",
            accelerator,
            "--devices",
            str(self.config.num_gpus),
            "--batch-size",
            str(self.config.batch_size),
            "--max_epochs",
            str(self.config.epochs),
        ]
        logger.info("Launching Piper training: %s", " ".join(cmd))
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
