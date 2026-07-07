"""Tests for audio readers."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from audiotrove.io.readers import LocalAudioReader
from audiotrove.document import AudioDocument


def test_reader_accepts_string_pattern():
    """Test LocalAudioReader initializes with string pattern (backward compatibility)."""
    pattern = "/path/to/*.wav"
    reader = LocalAudioReader(pattern)
    assert reader.path_patterns == [pattern]
    assert reader.target_sr == 16000


def test_reader_accepts_list_patterns():
    """Test LocalAudioReader initializes with list of patterns."""
    patterns = ["/path/to/*.wav", "/path/to/*.flac", "/path/to/*.mp3"]
    reader = LocalAudioReader(patterns)
    assert reader.path_patterns == patterns
    assert reader.target_sr == 16000


def test_reader_target_sr_parameter():
    """Test LocalAudioReader respects target sample rate parameter."""
    pattern = "/path/to/*.wav"
    reader = LocalAudioReader(pattern, target_sample_rate=22050)
    assert reader.target_sr == 22050


def test_reader_duration_parameters():
    """Test LocalAudioReader respects duration parameters."""
    pattern = "/path/to/*.wav"
    reader = LocalAudioReader(
        pattern,
        min_duration_seconds=1.0,
        max_duration_seconds=30.0
    )
    assert reader.min_duration == 1.0
    assert reader.max_duration == 30.0

