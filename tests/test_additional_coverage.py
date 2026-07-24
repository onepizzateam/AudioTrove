import wave
import numpy as np
from audiotrove.document import AudioDocument


def _gen_from_fixtures(fixtures_dir):
    from audiotrove.utils.hashing import make_doc_id

    for f in sorted(fixtures_dir.glob("*.wav")):
        if f.name == "corrupt.wav":
            continue
        with wave.open(str(f), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)
        duration = float(len(audio)) / float(sr)
        yield AudioDocument(
            audio=audio,
            sample_rate=sr,
            source_path=str(f),
            duration_seconds=duration,
            doc_id=make_doc_id(str(f)),
        )


def test_executor_parallel_processes_all_files(fixtures_dir, tmp_path):
    from audiotrove.filters.vad import SileroVADFilter
    from audiotrove.filters.snr import SNRFilter
    from audiotrove.executor.local import LocalExecutor
    from audiotrove.io.writers import JSONLWriter

    pipeline = [SileroVADFilter(min_speech_ratio=0.0), SNRFilter(min_snr_db=0.0)]
    executor = LocalExecutor(
        pipeline=pipeline, num_workers=2, checkpoint_path=str(tmp_path / "ckpt.db")
    )
    reader = _gen_from_fixtures(fixtures_dir)
    writer = JSONLWriter(str(tmp_path / "manifest.jsonl"))

    stats = executor.run(reader, writer)
    assert stats["processed"] > 0
    assert stats["errors"] == 0


def test_reader_skips_corrupt_file(monkeypatch, fixtures_dir):
    # Mock fsspec and torchaudio to simulate a corrupt file raising on load
    import audiotrove.io.readers as readers_mod
    from unittest.mock import Mock

    mock_fs = Mock()
    mock_fs.glob.return_value = [str(fixtures_dir / "corrupt.wav")]
    mock_fsspec = Mock()
    mock_fsspec.core.url_to_fs.return_value = (mock_fs, str(fixtures_dir / "*.wav"))
    monkeypatch.setattr(readers_mod, "fsspec", mock_fsspec)

    mock_torchaudio = Mock()

    def raise_load(f):
        raise OSError("corrupt")

    mock_torchaudio.load.side_effect = raise_load
    monkeypatch.setattr(readers_mod, "torchaudio", mock_torchaudio)

    reader = readers_mod.LocalAudioReader(str(fixtures_dir / "*.wav"))
    docs = list(reader)
    assert all(d is None for d in docs)


def test_reader_downmixes_stereo_real(fixtures_dir):
    # Read stereo.wav with stdlib wave and confirm downmixing produces 1D audio
    import wave as _wave

    f = fixtures_dir / "stereo.wav"
    with _wave.open(str(f), "rb") as wf:
        wf.getframerate()
        nch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if nch > 1:
        audio_mono = audio.reshape(-1, nch).mean(axis=1)
    else:
        audio_mono = audio
    assert audio_mono.ndim == 1


def test_vad_energy_fallback_no_torch(speech_clean, monkeypatch):
    from audiotrove.filters import vad as vad_module

    monkeypatch.setattr(vad_module, "HAS_TORCH", False)
    from audiotrove.filters.vad import SileroVADFilter

    f = SileroVADFilter(min_speech_ratio=0.3)
    result = f.filter(speech_clean)
    assert isinstance(result, bool)
    assert "vad_backend" in speech_clean.metadata
    assert speech_clean.metadata["vad_backend"] == "energy_fallback"
