"""Quality guard for the Rust sinc resampler in audiotrove_core.decode_audio_batch.

The whole point of the Rust decode path is that it must not regress audio
quality relative to the Python path -- in particular it must sinc-resample, not
naively linear-interpolate. The entire module is skipped when the optional
``audiotrove_core`` extension is not built, so a base checkout still passes.
"""

import numpy as np
import pytest

audiotrove_core = pytest.importorskip("audiotrove_core")
sf = pytest.importorskip("soundfile")


def _dominant_freq(signal, sr):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    return freqs[int(np.argmax(spectrum))]


def test_decode_batch_returns_expected_shape_and_sr(tmp_path):
    src_sr, target_sr = 48000, 24000
    seconds = 1.0
    t = np.arange(int(src_sr * seconds)) / src_sr
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), src_sr)

    results = audiotrove_core.decode_audio_batch([str(wav)], target_sr)
    assert len(results) == 1
    samples, sr, path = results[0]
    samples = np.asarray(samples, dtype=np.float32)

    assert sr == target_sr
    assert path == str(wav)
    expected_len = int(len(t) * target_sr / src_sr)
    assert abs(len(samples) - expected_len) <= 8


def test_decode_batch_preserves_tone_after_resample(tmp_path):
    """A 440 Hz tone downsampled 48k->24k must remain a clean 440 Hz tone.

    A broken linear resampler would smear energy across the spectrum and shift
    the dominant bin; sinc resampling preserves it.

    """
    src_sr, target_sr, freq = 48000, 24000, 440.0
    t = np.arange(int(src_sr * 1.0)) / src_sr
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), src_sr)

    samples, sr, _ = audiotrove_core.decode_audio_batch([str(wav)], target_sr)[0]
    samples = np.asarray(samples, dtype=np.float32)

    # Dominant frequency preserved within one FFT bin.
    dom = _dominant_freq(samples, sr)
    assert abs(dom - freq) < (sr / len(samples)) * 2

    # RMS roughly preserved (a pure tone's RMS ~ amplitude / sqrt(2) = 0.354).
    rms = float(np.sqrt(np.mean(samples**2)))
    assert 0.28 < rms < 0.42


def test_decode_batch_downmixes_to_mono(tmp_path):
    src_sr = 24000
    t = np.arange(int(src_sr * 0.5)) / src_sr
    left = 0.5 * np.sin(2 * np.pi * 300 * t)
    right = 0.5 * np.sin(2 * np.pi * 300 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    wav = tmp_path / "stereo.wav"
    sf.write(str(wav), stereo, src_sr)

    samples, sr, _ = audiotrove_core.decode_audio_batch([str(wav)], src_sr)[0]
    samples = np.asarray(samples, dtype=np.float32)
    assert samples.ndim == 1  # mono
