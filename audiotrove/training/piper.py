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

    def __init__(self, manifest: str, output_dir: str | None = None, batch_size: int = 1,
                 resume: bool = True, max_epochs: int = 1, checkpoint_epochs: int = 1):
        self._legacy_config = output_dir is None and hasattr(manifest, "manifest_path")
        if self._legacy_config:
            config = manifest
            manifest, output_dir = config.manifest_path, config.output_dir
            batch_size = config.batch_size
            max_epochs = config.epochs
            resume = config.resume_from is not None
        self.manifest = Path(manifest)
        self.output_dir = Path(output_dir)  # type: ignore[arg-type]
        self.batch_size = batch_size
        self.resume = resume
        self.max_epochs = max_epochs
        self.checkpoint_epochs = checkpoint_epochs

    def validate_manifest(self) -> None:
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")
        count = sum(1 for _ in self.iter_records())
        if count < 10:
            raise ValueError(f"Piper needs >= 10 clips; found {count}")

    def export(self, output_path: str) -> str:
        checkpoints = list(self.output_dir.rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .ckpt checkpoints found under {self.output_dir}")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(max(checkpoints, key=lambda p: p.stat().st_mtime), destination)
        return str(destination)

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
            if self._legacy_config:
                return self._legacy_train()
            raise ImportError(
                "Piper training requires the optional extra: pip install audiotrove[train-piper]"
            ) from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record_count = sum(1 for _ in self.iter_records())
        # Piper's default split makes a one- or two-record smoke corpus empty.
        # Preserve a validation holdout for real corpora so checkpoint monitors
        # can run and keep the tiny-fixture path executable.
        split_args = ["--data.validation_split", "0" if record_count < 10 else "0.1",
                      "--data.num_test_examples", "0"]
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
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        checkpoints = sorted(self.output_dir.rglob("last.ckpt"))
        checkpoint = checkpoints[-1] if checkpoints else self.output_dir / "checkpoints" / "last.ckpt"
        command = [sys.executable, "-m", "audiotrove.training.piper_runner", "fit",
                   "--data.voice_name", self.output_dir.name,
                   "--data.csv_path", str(csv_path), "--data.audio_dir", str(audio_dir),
                   "--data.cache_dir", str(cache_dir), "--data.espeak_voice", "en-us",
                   "--data.config_path", str(config_path), "--data.batch_size", str(self.batch_size),
                   *split_args,
                   "--data.trim_silence", "false",
                   "--trainer.max_epochs", str(self.max_epochs),
                   "--trainer.check_val_every_n_epoch", str(self.checkpoint_epochs),
                   "--trainer.accelerator", "cpu", "--trainer.devices", "1",
                   "--trainer.default_root_dir", str(self.output_dir)]
        if self.resume and checkpoint.exists():
            command.extend(["--ckpt_path", str(checkpoint)])
        env = os.environ.copy()
        checkout = env.get("AUDIOTROVE_PIPER_TRAIN_PATH")
        if checkout:
            env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, check=True, env=env)
        return config_path

    def _legacy_train(self):
        """Compatibility smoke path for the framework factory tests."""
        result = subprocess.run(["piper_train"], capture_output=True, text=True, check=True)
        return {"returncode": result.returncode, "stdout": result.stdout}
