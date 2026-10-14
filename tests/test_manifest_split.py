"""
Tests for the manifest split functionality and CLI command.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from audiotrove.cli.main import cli
from audiotrove.manifest.split import split_manifest


def _create_sample_manifest(path: Path, num_rows: int = 100, num_speakers: int = 4) -> list[dict]:
    """Helper to generate a synthetic manifest with speaker metadata."""
    records = []
    with path.open("w", encoding="utf-8") as f:
        for i in range(num_rows):
            speaker_id = f"speaker_{i % num_speakers}"
            record = {
                "doc_id": f"doc_{i:04d}",
                "source_path": f"/audio/clip_{i:04d}.wav",
                "sample_rate": 16000,
                "duration_seconds": 3.5,
                "metadata": {
                    "speaker": speaker_id,
                    "gender": "female" if (i % num_speakers) % 2 == 0 else "male",
                },
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")
    return records


def test_fraction_validation():
    """Fractions must sum to 1.0 and be non-negative."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        _create_sample_manifest(manifest_file, num_rows=10)
        out_dir = tmp_p / "splits"

        # Not summing to 1.0
        with pytest.raises(ValueError, match="Fractions must sum to 1.0"):
            split_manifest(manifest_file, out_dir, train_fraction=0.8, val_fraction=0.1, test_fraction=0.2)

        # Negative fraction
        with pytest.raises(ValueError, match="non-negative"):
            split_manifest(manifest_file, out_dir, train_fraction=1.2, val_fraction=-0.1, test_fraction=-0.1)


def test_unstratified_100_row_manifest():
    """Unstratified split divides 100 rows into exact requested proportions without duplicates."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        original_records = _create_sample_manifest(manifest_file, num_rows=100)
        out_dir = tmp_p / "splits"

        counts = split_manifest(
            manifest_file,
            out_dir,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            seed=42,
        )

        assert counts["total"] == 100
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

        train_file = out_dir / "train.jsonl"
        val_file = out_dir / "val.jsonl"
        test_file = out_dir / "test.jsonl"

        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        train_docs = [json.loads(line)["doc_id"] for line in train_file.read_text(encoding="utf-8").splitlines() if line]
        val_docs = [json.loads(line)["doc_id"] for line in val_file.read_text(encoding="utf-8").splitlines() if line]
        test_docs = [json.loads(line)["doc_id"] for line in test_file.read_text(encoding="utf-8").splitlines() if line]

        assert len(train_docs) == 80
        assert len(val_docs) == 10
        assert len(test_docs) == 10

        # Disjoint partitions
        assert len(set(train_docs) & set(val_docs)) == 0
        assert len(set(train_docs) & set(test_docs)) == 0
        assert len(set(val_docs) & set(test_docs)) == 0

        # All original doc IDs preserved
        all_split_docs = set(train_docs) | set(val_docs) | set(test_docs)
        assert all_split_docs == {r["doc_id"] for r in original_records}


def test_stratified_100_row_manifest():
    """Stratified split maintains proportional distribution across strata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        # 4 speakers, 25 rows each
        _create_sample_manifest(manifest_file, num_rows=100, num_speakers=4)
        out_dir = tmp_p / "splits"

        counts = split_manifest(
            manifest_file,
            out_dir,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            stratify_field="speaker",
            seed=123,
        )

        assert counts["total"] == 100
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

        train_lines = [json.loads(l) for l in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
        val_lines = [json.loads(l) for l in (out_dir / "val.jsonl").read_text(encoding="utf-8").splitlines() if l]
        test_lines = [json.loads(l) for l in (out_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if l]

        # For 25 items per stratum at 0.8 / 0.1 / 0.1:
        # 25 * 0.8 = 20 train, 25 * 0.1 = 2.5 -> 2 or 3 val and test
        for spk_idx in range(4):
            spk = f"speaker_{spk_idx}"
            spk_train = sum(1 for r in train_lines if r["metadata"]["speaker"] == spk)
            spk_val = sum(1 for r in val_lines if r["metadata"]["speaker"] == spk)
            spk_test = sum(1 for r in test_lines if r["metadata"]["speaker"] == spk)
            assert spk_train == 20
            assert spk_train + spk_val + spk_test == 25


def test_stratified_proposed_command_fractions():
    """Test the proposed command: 0.9 train, 0.05 val, 0.05 test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        _create_sample_manifest(manifest_file, num_rows=100, num_speakers=2)
        out_dir = tmp_p / "splits"

        counts = split_manifest(
            manifest_file,
            out_dir,
            train_fraction=0.9,
            val_fraction=0.05,
            test_fraction=0.05,
            stratify_field="speaker",
            seed=42,
        )

        assert counts["total"] == 100
        assert counts["train"] == 90
        assert counts["val"] == 5
        assert counts["test"] == 5


def test_seed_reproducibility():
    """Identical seeds produce identical outputs, different seeds produce different outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        _create_sample_manifest(manifest_file, num_rows=50)

        out1 = tmp_p / "splits1"
        out2 = tmp_p / "splits2"
        out3 = tmp_p / "splits3"

        split_manifest(manifest_file, out1, seed=999)
        split_manifest(manifest_file, out2, seed=999)
        split_manifest(manifest_file, out3, seed=111)

        # Same seed must be byte-for-byte identical
        assert (out1 / "train.jsonl").read_bytes() == (out2 / "train.jsonl").read_bytes()
        assert (out1 / "val.jsonl").read_bytes() == (out2 / "val.jsonl").read_bytes()
        assert (out1 / "test.jsonl").read_bytes() == (out2 / "test.jsonl").read_bytes()

        # Different seed produces different ordering
        assert (out1 / "train.jsonl").read_bytes() != (out3 / "train.jsonl").read_bytes()


def test_empty_manifest():
    """Empty manifest yields zero rows without raising errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        manifest_file.write_text("", encoding="utf-8")
        out_dir = tmp_p / "splits"

        counts = split_manifest(manifest_file, out_dir)
        assert counts == {"train": 0, "val": 0, "test": 0, "total": 0}
        assert (out_dir / "train.jsonl").read_text(encoding="utf-8") == ""


def test_missing_stratify_field_gracefully_handled():
    """Rows lacking the stratify field are grouped into missing bucket and still split."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        with manifest_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": "1", "metadata": {"speaker": "alice"}}) + "\n")
            f.write(json.dumps({"doc_id": "2"}) + "\n")
            f.write(json.dumps({"doc_id": "3", "metadata": {}}) + "\n")
            f.write(json.dumps({"doc_id": "4", "speaker": "bob"}) + "\n")

        out_dir = tmp_p / "splits"
        counts = split_manifest(
            manifest_file,
            out_dir,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            stratify_field="speaker",
        )
        assert counts["total"] == 4


def test_cli_split_command_success():
    """Test audiotrove split CLI end-to-end with flags."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        _create_sample_manifest(manifest_file, num_rows=100)
        out_dir = tmp_p / "splits"

        result = runner.invoke(
            cli,
            [
                "split",
                str(manifest_file),
                "--train", "0.9",
                "--val", "0.05",
                "--test", "0.05",
                "--stratify", "speaker",
                "--output-dir", str(out_dir),
                "--seed", "42",
            ],
        )

        assert result.exit_code == 0
        assert "Manifest Split Results" in result.output
        assert "Successfully split manifest" in result.output
        assert (out_dir / "train.jsonl").exists()
        assert (out_dir / "val.jsonl").exists()
        assert (out_dir / "test.jsonl").exists()


def test_cli_split_command_fraction_mismatch():
    """CLI rejects fractions that do not sum to 1.0."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        manifest_file = tmp_p / "manifest.jsonl"
        _create_sample_manifest(manifest_file, num_rows=10)

        result = runner.invoke(
            cli,
            [
                "split",
                str(manifest_file),
                "--train", "0.5",
                "--val", "0.2",
                "--test", "0.2",
            ],
        )
        assert result.exit_code != 0
        assert "Fractions must sum to 1.0" in result.output
