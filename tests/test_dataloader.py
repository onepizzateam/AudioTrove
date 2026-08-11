"""Tests for the training RustAudioDataset (soundfile fallback path).

These exercise the pure-Python decode path; the Rust ``audiotrove_core``
extension is optional and, when absent, the dataset falls back to
``soundfile`` + numpy transparently.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from audiotrove.training.dataloader import RustAudioDataset


def _write_wav(path, seconds=2.0, sr=16000, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


def _write_manifest(tmp_path, entries):
    """entries: list of (path, duration, text) tuples -> filelist.txt."""
    lines = [f"{p}\t{d}\t{txt}" for p, d, txt in entries]
    manifest = tmp_path / "filelist.txt"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        RustAudioDataset(str(tmp_path / "nope.txt"))


def test_parse_filters_out_of_range_durations(tmp_path):
    good = _write_wav(tmp_path / "good.wav")
    short = _write_wav(tmp_path / "short.wav", seconds=0.5)
    manifest = _write_manifest(
        tmp_path,
        [(good, 2.0, "keep me"), (short, 0.5, "too short")],
    )
    ds = RustAudioDataset(str(manifest), min_duration=1.0, max_duration=15.0, preload=False)
    assert len(ds) == 1
    assert ds.entries[0]["text"] == "keep me"


def test_getitem_returns_expected_fields_and_resamples(tmp_path):
    wav = _write_wav(tmp_path / "clip.wav", seconds=2.0, sr=16000)
    manifest = _write_manifest(tmp_path, [(wav, 2.0, "hello world")])
    ds = RustAudioDataset(str(manifest), target_sr=8000, preload=False)

    item = ds[0]
    assert set(item) == {"audio", "text", "path", "sample_rate"}
    assert item["text"] == "hello world"
    assert item["sample_rate"] == 8000
    assert isinstance(item["audio"], torch.Tensor)
    # 2 s at 8 kHz after resampling from 16 kHz.
    assert abs(item["audio"].shape[0] - 16000) <= 2


def test_collate_fn_right_pads_batch(tmp_path):
    long_wav = _write_wav(tmp_path / "long.wav", seconds=2.0)
    short_wav = _write_wav(tmp_path / "shorter.wav", seconds=1.5)
    manifest = _write_manifest(
        tmp_path,
        [(long_wav, 2.0, "long clip"), (short_wav, 1.5, "short clip")],
    )
    ds = RustAudioDataset(str(manifest), target_sr=16000, preload=False)

    batch = RustAudioDataset.collate_fn([ds[0], ds[1]])
    assert batch["audio"].shape[0] == 2
    assert batch["audio"].shape[1] == int(batch["audio_lengths"].max().item())
    assert batch["texts"] == ["long clip", "short clip"]
    assert len(batch["paths"]) == 2
    assert batch["sample_rate"] == 16000
    # The shorter clip must be right-zero-padded to the batch width.
    shorter_len = int(batch["audio_lengths"].min().item())
    assert torch.count_nonzero(batch["audio"][1, shorter_len:]) == 0
