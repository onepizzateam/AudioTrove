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
        """Run F5-TTS training via AudioTrove's owned loop. Returns a metrics dict.

        Delegates to :func:`audiotrove.training.f5tts_loop.train_f5tts`, which
        drives the ``f5_tts.model`` classes directly (rather than shelling out to
        the upstream ``f5_tts.train`` entry point) so it can feed the
        Rust-accelerated dataloader, pick the best dtype per device, and emit
        progress events.

        Requires: pip install audiotrove[train-f5tts]
        """
        from audiotrove.training.f5tts_loop import train_f5tts

        model_name = self.config.model_name or "F5TTS_Base"
        # The generic framework key "f5tts" is not a model preset; map it to the
        # default base recipe.
        if model_name == "f5tts":
            model_name = "F5TTS_Base"

        return train_f5tts(
            manifest_path=self.config.manifest_path,
            output_dir=self.config.output_dir,
            model_name=model_name,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            num_gpus=self.config.num_gpus,
            mixed_precision=self.config.mixed_precision,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_every_n_epochs=self.config.save_every_n_epochs,
            resume_from=self.config.resume_from,
            torch_threads=self.config.torch_threads,
            dataloader_workers=self.config.dataloader_workers,
        )


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
