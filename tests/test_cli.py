"""
Tests for the CLI interface.
"""
import tempfile
from pathlib import Path
import numpy as np
import json
import pytest

from click.testing import CliRunner

from audiotrove.cli.main import cli
from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


def test_curate_command_help():
    """Test curate command help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ['curate', '--help'])
    assert result.exit_code == 0
    assert 'vad-threshold' in result.output
    assert 'snr-min' in result.output
    assert 'extensions' in result.output  # New --extensions flag


def test_inspect_command_help():
    """Test inspect command help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', '--help'])
    assert result.exit_code == 0
    assert 'INPUT_PATH' in result.output


def test_cli_group_version():
    """Test CLI version flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.0.1' in result.output or 'version' in result.output.lower()


def test_cli_group_help():
    """Test main CLI help."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'curate' in result.output
    assert 'inspect' in result.output


def test_curate_multi_format():
    """Test curate with --extensions wav,flac to pick up multiple formats."""
    try:
        import torchaudio
        import torch
    except ImportError:
        pytest.skip("torchaudio not available")
    
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create synthetic WAV and FLAC files
        sr = 16000
        duration_s = 1.0
        samples = int(sr * duration_s)
        
        # WAV file: simple sine wave (speech-like signal for VAD)
        wav_audio = torch.sin(2 * torch.pi * 440 * torch.arange(samples).float() / sr)
        wav_path = input_dir / "audio1.wav"
        torchaudio.save(str(wav_path), wav_audio.unsqueeze(0), sr)
        
        # FLAC file: different sine wave
        flac_audio = torch.sin(2 * torch.pi * 220 * torch.arange(samples).float() / sr)
        flac_path = input_dir / "audio2.flac"
        torchaudio.save(str(flac_path), flac_audio.unsqueeze(0), sr)
        
        # Run curate with --extensions wav,flac and high SNR threshold to avoid filtering
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
            '--extensions', 'wav,flac',
            '--snr-min', '0',  # No SNR filtering
            '--vad-threshold', '0.0',  # Accept all audio
        ])
        
        assert result.exit_code == 0, f"CLI exited with code {result.exit_code}: {result.output}"
        
        # Verify manifest exists
        manifest_path = output_dir / "manifest.jsonl"
        assert manifest_path.exists(), "manifest.jsonl not created"
        
        # Check that both files were processed
        with open(manifest_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        # Should have 2 documents (one from WAV, one from FLAC)
        assert len(lines) == 2, f"Expected 2 documents in manifest, got {len(lines)}"
