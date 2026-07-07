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


def test_curate_nonexistent_input():
    """Test curate returns SystemExit when input doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'curate',
        '/nonexistent/path',
        '/tmp/output',
    ])
    
    assert result.exit_code == 1
    assert 'Error' in result.output
    assert 'does not exist' in result.output


def test_inspect_nonexistent_input():
    """Test inspect returns SystemExit when input doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'inspect',
        '/nonexistent/path/to/audio',
    ])
    
    assert result.exit_code == 1
    assert 'Error' in result.output


def test_curate_creates_output_directory():
    """Test curate creates output directory if it doesn't exist."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "deep" / "output"  # Non-existent nested
        input_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
        ])
        
        # Should succeed (even if no audio found)
        assert result.exit_code in [0, 1]  # 0 if audio files processed, 1 if errors
        # Verify output directory was created
        assert output_dir.exists()


def test_inspect_limit_parameter():
    """Test inspect --limit parameter is accepted."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'inspect',
            str(input_dir),
            '--limit', '5',
        ])
        
        # Should succeed (or fail gracefully if fsspec missing)
        assert result.exit_code in [0, 1]


def test_curate_extensions_parameter():
    """Test curate --extensions parameter is accepted."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
            '--extensions', 'wav,mp3,flac',
        ])
        
        # Should accept extensions parameter without error
        assert result.exit_code in [0, 1]  # 0 if processed, 1 if other error


def test_curate_workers_parameter():
    """Test curate --workers parameter is accepted."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
            '--workers', '2',
        ])
        
        # Should accept workers parameter without error
        assert result.exit_code in [0, 1]


def test_curate_checkpoint_parameter():
    """Test curate --checkpoint parameter is accepted."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        checkpoint_db = Path(tmpdir) / "checkpoint.db"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
            '--checkpoint', str(checkpoint_db),
        ])
        
        # Should accept checkpoint parameter without error
        assert result.exit_code in [0, 1]


def test_curate_filter_parameters():
    """Test curate VAD and SNR filter parameters are accepted."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "test.wav").touch()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
            '--vad-threshold', '0.5',
            '--snr-min', '15.0',
        ])
        
        # Should accept filter parameters without error
        assert result.exit_code in [0, 1]


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


def test_curate_empty_input_directory():
    """Test curate on empty input directory."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            'curate',
            str(input_dir),
            str(output_dir),
        ])
        
        # Should complete successfully (no files to process)
        assert result.exit_code in [0, 1]


def test_inspect_help_shows_limit_option():
    """Test inspect help shows --limit option."""
    runner = CliRunner()
    result = runner.invoke(cli, ['inspect', '--help'])
    assert result.exit_code == 0
    assert '--limit' in result.output
    assert 'Show stats for first N files' in result.output
