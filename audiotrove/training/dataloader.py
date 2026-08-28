"""Rust-accelerated audio dataset for training.

``RustAudioDataset`` reads a curation manifest (the ``filelist.txt`` produced by
:class:`~audiotrove.exporters.tts_manifest.TTSManifestExporter`) and yields
decoded audio tensors ready for a training ``DataLoader``.

Decoding prefers the optional ``audiotrove_core`` Rust extension, which decodes
and sinc-resamples an entire batch of files off the GIL in parallel. When the
extension is not installed it transparently falls back to a pure-Python
``soundfile`` + numpy path, so training works on a base ``pip install`` too.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch optional for base install
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

    class Dataset:  # type: ignore[no-redef]
        """Minimal stand-in so the module imports without torch."""

try:
    import soundfile as _soundfile
except Exception:  # noqa: BLE001 - optional
    _soundfile = None

logger = logging.getLogger(__name__)


class RustAudioDataset(Dataset):
    """A ``torch.utils.data.Dataset`` over an AudioTrove manifest.

    Args:
        manifest_path: Path to a ``filelist.txt`` (``path\\tduration\\ttext`` per
            line). Lines whose duration falls outside ``[min_duration,
            max_duration]`` are dropped.
        target_sr: Sample rate every clip is resampled to.
        max_duration: Upper duration bound (seconds) for included clips.
        min_duration: Lower duration bound (seconds) for included clips.
        preload: When True, decode every clip up front (via Rust when available)
            so ``__getitem__`` is a cheap dictionary lookup. When False, clips
            are decoded lazily on first access.
        cache_tensors: When True (and lazy), cache decoded tensors after first
            access to avoid repeated decode across epochs.
    """

    def __init__(
        self,
        manifest_path: str,
        target_sr: int = 22050,
        max_duration: float = 15.0,
        min_duration: float = 1.0,
        preload: bool = True,
        cache_tensors: bool = False,
    ):
        if not HAS_TORCH:
            raise RuntimeError("torch is required to use RustAudioDataset")

        self.manifest_path = str(manifest_path)
        self.target_sr = int(target_sr)
        self.max_duration = float(max_duration)
        self.min_duration = float(min_duration)
        self.preload = bool(preload)
        self.cache_tensors = bool(cache_tensors)

        self.entries = self._parse_manifest()
        self._audio: dict[int, np.ndarray] = {}

        if self.preload:
            self._preload_rust()

    def _parse_manifest(self) -> list[dict]:
        """Parse the manifest into a list of ``{path, duration, text}`` dicts."""
        path = Path(self.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        entries: list[dict] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            wav_path = parts[0]
            try:
                duration = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
            except ValueError:
                duration = 0.0
            text = parts[2] if len(parts) > 2 else ""

            # Filter by duration when the manifest records it. A duration of 0
            # (unknown) is kept; the decode step establishes the real length.
            if duration:
                if duration < self.min_duration or duration > self.max_duration:
                    continue

            entries.append({"path": wav_path, "duration": duration, "text": text})

        if not entries:
            logger.warning("Manifest %s produced 0 usable entries", self.manifest_path)
        return entries

    def _preload_rust(self) -> None:
        """Decode all clips up front, preferring the Rust batch decoder."""
        paths = [e["path"] for e in self.entries]
        if not paths:
            return

        try:
            from audiotrove_core import decode_audio_batch
        except ImportError:
            logger.info(
                "audiotrove_core not installed; RustAudioDataset will decode "
                "lazily via soundfile."
            )
            self.preload = False
            return

        try:
            results = decode_audio_batch(paths, self.target_sr)
        except Exception as e:  # noqa: BLE001
            logger.warning("Rust batch decode failed (%s); falling back to lazy decode.", e)
            self.preload = False
            return

        by_path = {path: (samples, sr) for samples, sr, path in results}
        for idx, entry in enumerate(self.entries):
            samples, sr = by_path.get(entry["path"], (None, 0))
            if samples is None or sr == 0 or len(samples) == 0:
                # Decode the straggler lazily rather than dropping it silently.
                self._audio[idx] = self._load_fallback(entry["path"])
            else:
                self._audio[idx] = np.asarray(samples, dtype=np.float32)

    def _load_fallback(self, path: str) -> np.ndarray:
        """Pure-Python decode + resample for a single file."""
        if _soundfile is None:
            raise RuntimeError("soundfile is required to decode audio without audiotrove_core")

        audio, sr = _soundfile.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.target_sr:
            num = int(len(audio) * float(self.target_sr) / float(sr))
            audio = np.interp(
                np.linspace(0, len(audio), num, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return np.asarray(audio, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        if idx in self._audio:
            audio = self._audio[idx]
        else:
            audio = self._load_fallback(entry["path"])
            if self.cache_tensors:
                self._audio[idx] = audio

        return {
            "audio": torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
            "text": entry["text"],
            "path": entry["path"],
            "sample_rate": self.target_sr,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Pad a list of items into a right-zero-padded ``(B, T)`` batch."""
        audios = [item["audio"] for item in batch]
        lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(audios) > 0 else 0

        padded = torch.zeros((len(audios), max_len), dtype=torch.float32)
        for i, a in enumerate(audios):
            padded[i, : a.shape[0]] = a

        return {
            "audio": padded,
            "audio_lengths": lengths,
            "texts": [item["text"] for item in batch],
            "paths": [item["path"] for item in batch],
            "sample_rate": batch[0]["sample_rate"] if batch else 0,
        }
