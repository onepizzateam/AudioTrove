"""CPU-first Piper training adapter."""

from pathlib import Path


class PiperTrainer:
    """Stream a curated F5-TTS filelist into the optional Piper trainer."""

    framework = "piper"

    def __init__(self, manifest: str, output_dir: str, batch_size: int = 1,
                 resume: bool = True):
        self.manifest = Path(manifest)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.resume = resume

    def iter_records(self):
        with self.manifest.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t", 2)
                if len(fields) == 3:
                    yield fields

    def train(self):
        try:
            import piper_train  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Piper training requires the optional extra: pip install audiotrove[train-piper]"
            ) from exc
        raise NotImplementedError(
            "Invoke piper_train with the generated manifest; the adapter validates and streams records."
        )
