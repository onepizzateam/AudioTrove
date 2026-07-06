"""
Tests for the CLI interface.
"""
import tempfile
from pathlib import Path
import numpy as np

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
