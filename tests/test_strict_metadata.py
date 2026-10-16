"""
Tests for --strict-metadata CLI flag and pipeline fail-fast validation.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from audiotrove.cli.main import cli
from audiotrove.document import AudioDocument
from audiotrove.executor.local import LocalExecutor
from audiotrove.io.writers import JSONLWriter
from audiotrove.utils.hashing import make_doc_id


def _create_wav_file(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    """Create a dummy WAV file for testing."""
    samples = int(duration_s * sr)
    data = np.zeros(samples, dtype=np.float32)
    half = samples // 2
    data[:half] = (np.sin(2 * np.pi * 440 * np.arange(half) / sr) * 0.5).astype(np.float32)
    sf.write(str(path), data, sr)
    return path


def test_cli_strict_metadata_missing_fails_fast(tmp_path):
    """Pipeline fails fast when a required metadata key is missing from output."""
    runner = CliRunner()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    _create_wav_file(input_dir / "clip1.wav")

    output_dir = tmp_path / "out"

    # Run without SNR/VAD so snr_db and vad_speech_ratio are not in metadata
    result = runner.invoke(
        cli,
        [
            "curate",
            str(input_dir),
            str(output_dir),
            "--vad-threshold", "0.0",
            "--snr-min", "0.0",
            "--strict-metadata", "snr_db,vad_speech_ratio",
        ],
    )

    assert result.exit_code != 0
    assert "missing required metadata field" in result.output
    assert "snr_db" in result.output or "vad_speech_ratio" in result.output


def test_cli_strict_metadata_present_succeeds(tmp_path):
    """Pipeline succeeds when all required metadata keys are populated."""
    runner = CliRunner()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    _create_wav_file(input_dir / "clip1.wav")

    output_dir = tmp_path / "out"

    # Run with SNR filter enabled (so snr_db is computed and added to metadata)
    result = runner.invoke(
        cli,
        [
            "curate",
            str(input_dir),
            str(output_dir),
            "--vad-threshold", "0.0",
            "--snr-min", "1.0",
            "--strict-metadata", "snr_db",
        ],
    )

    assert result.exit_code == 0
    assert "Manifest written to" in result.output
    assert (output_dir / "manifest.jsonl").exists()


class DummyReader:
    """Mock reader yielding synthetic audio documents."""

    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        return iter(self.docs)


def test_executor_strict_metadata_fail_fast_parallel(tmp_path):
    """Parallel executor propagates missing metadata ValueError immediately."""
    import pytest

    doc = AudioDocument(
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        source_path="fake.wav",
        duration_seconds=1.0,
        doc_id=make_doc_id("fake.wav"),
        metadata={"other_key": "val"},
    )
    reader = DummyReader([doc])
    writer = JSONLWriter(str(tmp_path / "manifest.jsonl"), required_metadata=["required_field"])

    executor = LocalExecutor(pipeline=[], num_workers=2)
    with pytest.raises(ValueError, match="missing required metadata field\\(s\\): required_field"):
        executor.run(reader, writer)
