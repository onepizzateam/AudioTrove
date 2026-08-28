"""CPU-first Piper training adapter."""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


class PiperTrainer:
    """Stream a curated F5-TTS filelist into the optional Piper trainer."""

    framework = "piper"

    def __init__(self, manifest: str, output_dir: str, batch_size: int = 1,
                 resume: bool = True, max_epochs: int = 1, checkpoint_epochs: int = 1):
        self.manifest = Path(manifest)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.resume = resume
        self.max_epochs = max_epochs
        self.checkpoint_epochs = checkpoint_epochs

    def iter_records(self):
        with self.manifest.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t", 2)
                if len(fields) == 3:
                    yield fields

    def train(self):
        """Prepare Piper metadata and run its CPU Lightning trainer."""
        try:
            import piper.train  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Piper training requires the optional extra: pip install audiotrove[train-piper]"
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "metadata.csv"
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="|")
            for audio_path, _duration, text in self.iter_records():
                source = Path(audio_path)
                destination = audio_dir / source.name
                if source.resolve() != destination.resolve():
                    shutil.copyfile(source, destination)
                writer.writerow([destination.name, text])
        config_path = self.output_dir / "config.json"
        checkpoint = self.output_dir / "last.ckpt"
        command = [sys.executable, "-m", "piper.train", "fit",
                   "--data.voice_name", self.output_dir.name,
                   "--data.csv_path", str(csv_path), "--data.audio_dir", str(audio_dir),
                   "--data.config_path", str(config_path), "--data.batch_size", str(self.batch_size),
                   "--trainer.max_epochs", str(self.max_epochs),
                   "--trainer.check_val_every_n_epoch", str(self.checkpoint_epochs),
                   "--trainer.accelerator", "cpu", "--trainer.devices", "1"]
        if self.resume and checkpoint.exists():
            command.extend(["--ckpt_path", str(checkpoint)])
        env = os.environ.copy()
        checkout = env.get("AUDIOTROVE_PIPER_TRAIN_PATH")
        if checkout:
            env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, check=True, env=env)
        return config_path
