"""
Generate audio test fixtures for AudioTrove.
Run once: python scripts/generate_fixtures.py
Outputs to tests/fixtures/
"""
import numpy as np
from pathlib import Path
import wave

FIXTURES_DIR = Path("tests/fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
SR = 16000

def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

def main():
    duration = 5.0
    n = int(duration * SR)
    t = np.linspace(0, duration, n, endpoint=False)

    # speech_clean.wav — voiced sine at 150Hz + harmonics (speech-like tonal content) + quiet noise
    speech = (
        0.4 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t) +
        0.05 * np.random.randn(n)
    )
    speech = speech / np.max(np.abs(speech)) * 0.8
    write_wav(FIXTURES_DIR / "speech_clean.wav", speech)
    print("✓ speech_clean.wav")

    # speech_noisy.wav — same signal + heavy additive noise (SNR ~8dB)
    noise = np.random.randn(n) * 0.35
    speech_noisy = speech + noise
    speech_noisy = speech_noisy / np.max(np.abs(speech_noisy)) * 0.8
    write_wav(FIXTURES_DIR / "speech_noisy.wav", speech_noisy)
    print("✓ speech_noisy.wav")

    # silence.wav — true silence (near-zero)
    silence = np.zeros(n, dtype=np.float32) + np.random.randn(n) * 1e-5
    write_wav(FIXTURES_DIR / "silence.wav", silence)
    print("✓ silence.wav")

    # music.wav — complex multi-tone (not speech-like, no 150Hz fundamental)
    music = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 554 * t) +
        0.3 * np.sin(2 * np.pi * 659 * t) +
        0.05 * np.random.randn(n)
    )
    music = music / np.max(np.abs(music)) * 0.8
    write_wav(FIXTURES_DIR / "music.wav", music)
    print("✓ music.wav")

    # short_clip.wav — 0.3s (below min_duration_seconds threshold)
    short = np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(0.3 * SR), endpoint=False)).astype(np.float32) * 0.5
    write_wav(FIXTURES_DIR / "short_clip.wav", short)
    print("✓ short_clip.wav")

    # corrupt.wav — truncated header (write a broken WAV)
    corrupt_path = FIXTURES_DIR / "corrupt.wav"
    with open(corrupt_path, 'wb') as f:
        f.write(b'RIFF\x00\x00\x00\x00WAVEfmt ')
    print("✓ corrupt.wav")

    # stereo.wav — stereo file (should be downmixed to mono by reader)
    stereo_path = FIXTURES_DIR / "stereo.wav"
    left = speech
    right = music[:len(speech)]
    stereo = np.stack([left, right], axis=0)
    stereo_int16 = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(stereo_path), 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_int16.T.tobytes())
    print("✓ stereo.wav")

    print(f"\nAll fixtures written to {FIXTURES_DIR}/")

if __name__ == '__main__':
    main()
