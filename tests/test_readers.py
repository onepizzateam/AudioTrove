"""Tests for audio readers with 85%+ coverage target."""
from unittest.mock import Mock, patch
import numpy as np
import pytest

from audiotrove.io.readers import LocalAudioReader
from audiotrove.document import AudioDocument


# ========== Initialization Tests ==========

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


# ========== Iteration Tests ==========

def _make_mock_tensor(samples):
    """Helper to create mock tensor for mocking torchaudio."""
    tensor = Mock()
    tensor.ndim = 1
    tensor.shape = (samples,)
    tensor.numpy.return_value = np.array([0.1] * samples, dtype=np.float32)
    tensor.squeeze.return_value = tensor
    tensor.mean.return_value = tensor
    return tensor


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_iter_processes_files(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader.__iter__ finds and processes files from glob."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/audio.wav"]
    
    audio_tensor = _make_mock_tensor(16000)
    mock_torchaudio.load.return_value = (audio_tensor, 16000)
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    mock_resampler = Mock()
    mock_resampler.return_value = audio_tensor
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    reader = LocalAudioReader("/path/to/*.wav")
    docs = [doc for doc in reader if doc is not None]
    
    assert len(docs) == 1
    assert isinstance(docs[0], AudioDocument)


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_iter_multiple_files(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader.__iter__ handles multiple files."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/file1.wav", "/path/to/file2.wav"]
    
    audio_tensor = _make_mock_tensor(8000)
    mock_torchaudio.load.return_value = (audio_tensor, 16000)
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    mock_resampler = Mock()
    mock_resampler.return_value = audio_tensor
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    reader = LocalAudioReader("/path/to/*.wav")
    docs = [doc for doc in reader if doc is not None]
    
    assert len(docs) == 2


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_stereo_downmix(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader downmixes stereo to mono."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/stereo.wav"]
    
    # Stereo: 2 channels
    stereo_tensor = Mock()
    stereo_tensor.ndim = 2
    stereo_tensor.shape = (2, 8000)
    
    # After downmix (mean)
    mono_tensor = _make_mock_tensor(8000)
    stereo_tensor.mean.return_value = mono_tensor
    
    mock_torchaudio.load.return_value = (stereo_tensor, 16000)
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    mock_resampler = Mock()
    mock_resampler.return_value = mono_tensor
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    reader = LocalAudioReader("/path/to/*.wav")
    docs = [doc for doc in reader if doc is not None]
    
    assert len(docs) == 1
    assert docs[0].audio.ndim == 1  # Mono


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_resampling(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader resamples audio."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/audio.wav"]
    
    # Original at 44.1kHz
    source_tensor = _make_mock_tensor(44100)
    mock_torchaudio.load.return_value = (source_tensor, 44100)
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    # Resampled to 16kHz
    target_tensor = _make_mock_tensor(16000)
    mock_resampler = Mock()
    mock_resampler.return_value = target_tensor
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    reader = LocalAudioReader("/path/to/*.wav", target_sample_rate=16000)
    docs = [doc for doc in reader if doc is not None]
    
    assert len(docs) == 1
    assert docs[0].sample_rate == 16000
    mock_torchaudio.transforms.Resample.assert_called_with(44100, 16000)


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_min_duration_filter(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader filters short audio."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/short.wav", "/path/to/long.wav"]
    
    # Short: 100 samples < 1s minimum
    short_tensor = _make_mock_tensor(100)
    # Long: 32000 samples > 1s minimum
    long_tensor = _make_mock_tensor(32000)
    
    mock_torchaudio.load.side_effect = [(short_tensor, 16000), (long_tensor, 16000)]
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    mock_resampler = Mock()
    mock_resampler.side_effect = [short_tensor, long_tensor]
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    reader = LocalAudioReader("/path/to/*.wav", min_duration_seconds=1.0)
    docs = [doc for doc in reader if doc is not None]
    
    # Only long file should pass
    assert len(docs) == 1
    assert len(docs[0].audio) == 32000


@patch('audiotrove.io.readers.fsspec')
@patch('audiotrove.io.readers.torchaudio')
def test_reader_max_duration_truncate(mock_torchaudio, mock_fsspec):
    """Test LocalAudioReader truncates long audio."""
    mock_fs = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, "/path/to/*.wav")
    mock_fs.glob.return_value = ["/path/to/long.wav"]
    
    # Long audio: 160000 samples (10s at 16kHz)
    long_tensor = _make_mock_tensor(160000)
    mock_torchaudio.load.return_value = (long_tensor, 16000)
    mock_fs.open.return_value.__enter__ = Mock(return_value=Mock())
    mock_fs.open.return_value.__exit__ = Mock(return_value=None)
    
    mock_resampler = Mock()
    mock_resampler.return_value = long_tensor
    mock_torchaudio.transforms.Resample.return_value = mock_resampler
    
    # Max 5 seconds
    reader = LocalAudioReader("/path/to/*.wav", max_duration_seconds=5.0)
    docs = [doc for doc in reader if doc is not None]
    
    assert len(docs) == 1
    doc = docs[0]
    # Truncated to 5s = 80000 samples
    assert len(doc.audio) == 80000
    assert doc.duration_seconds == 5.0


# ========== Dependency Tests ==========

@patch('audiotrove.io.readers.fsspec', None)
def test_reader_error_no_fsspec():
    """Test LocalAudioReader requires fsspec."""
    reader = LocalAudioReader("/path/to/*.wav")
    
    with pytest.raises(RuntimeError, match="fsspec is required"):
        list(reader)
