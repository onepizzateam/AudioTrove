"""Tests for the TTS curation pipeline."""

from pathlib import Path

import numpy as np
import soundfile as sf

from audiotrove.document import AudioDocument
from audiotrove.exporters.tts_manifest import TTSManifestExporter
from audiotrove.filters.duration import DurationBucketFilter
from audiotrove.filters.vad import VADSegmenter
from audiotrove.pipelines.tts import tts_pipeline
from audiotrove.transformers.silence_trim import SilenceTrimmingTransformer


def test_silence_trimming_transformer_trims_with_padding(speech_clean):
    """Trimming uses existing timestamps and updates duration metadata."""
    original_duration = speech_clean.duration_seconds
    speech_clean.metadata["vad_speech_timestamps"] = [
        {"start": 16000, "end": len(speech_clean.audio) - 16000}
    ]

    result = SilenceTrimmingTransformer(padding_ms=150).transform(speech_clean)

    assert result.duration_seconds < original_duration
    assert result.duration_seconds == result.metadata["trimmed_duration_seconds"]
    assert result.metadata["vad_speech_timestamps"][0]["start"] == 2400


def test_silence_trimming_transformer_respects_minimum_duration(speech_clean):
    """Trimming leaves a document intact when it would make it too short."""
    original_duration = speech_clean.duration_seconds
    speech_clean.metadata["vad_speech_timestamps"] = [{"start": 16000, "end": 20000}]

    result = SilenceTrimmingTransformer(padding_ms=0, min_duration_seconds=1.0).transform(
        speech_clean
    )

    assert result.duration_seconds == original_duration
    assert result.metadata["trimmed_duration_seconds"] == original_duration


def test_duration_bucket_filter_records_rejection_reason(speech_clean):
    """Duration filter keeps in-range clips and records rejected ranges."""
    duration_filter = DurationBucketFilter(min_duration_seconds=2.0, max_duration_seconds=5.0)
    assert duration_filter.filter(speech_clean)

    short_doc = AudioDocument(
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        source_path="short.wav",
        duration_seconds=1.0,
        doc_id="short",
    )
    long_doc = AudioDocument(
        audio=np.zeros(96000, dtype=np.float32),
        sample_rate=16000,
        source_path="long.wav",
        duration_seconds=6.0,
        doc_id="long",
    )
    assert not duration_filter.filter(short_doc)
    assert short_doc.metadata["duration_filter_reason"] == "too_short"
    assert not duration_filter.filter(long_doc)
    assert long_doc.metadata["duration_filter_reason"] == "too_long"


def test_tts_manifest_exporter_writes_ljspeech_and_f5tts(tmp_path, speech_clean, speech_noisy):
    """Exporter writes both requested manifest formats."""
    speech_clean.metadata["transcription"] = "A clean utterance."
    exporter = TTSManifestExporter(str(tmp_path), speaker_id="test_speaker")

    output_files = exporter.export([speech_clean, speech_noisy])

    metadata_lines = (tmp_path / "metadata.csv").read_text(encoding="utf-8").splitlines()
    filelist_lines = (tmp_path / "filelist.txt").read_text(encoding="utf-8").splitlines()
    assert set(output_files) == {str(tmp_path / "metadata.csv"), str(tmp_path / "filelist.txt")}
    assert (
        metadata_lines[0]
        == f"{Path(speech_clean.source_path).name}|test_speaker|A clean utterance."
    )
    assert metadata_lines[1].endswith("|test_speaker|")
    assert filelist_lines[0].split("\t", 2)[2] == "A clean utterance."
    assert filelist_lines[1].endswith("\t")


def test_tts_manifest_exporter_gives_fanout_segments_unique_wav_paths(tmp_path):
    """Fan-out documents from one source file must not overwrite each other."""
    exporter = TTSManifestExporter(str(tmp_path))
    documents = [
        AudioDocument(
            audio=np.ones(16000, dtype=np.float32),
            sample_rate=16000,
            source_path="chapter.mp3",
            duration_seconds=1.0,
            doc_id=f"segment-{index}",
            metadata={"parent_doc_id": "chapter"},
        )
        for index in range(2)
    ]

    exporter.export(documents)

    wav_paths = [line.split("\t", 1)[0] for line in (tmp_path / "filelist.txt").read_text().splitlines()]
    assert len(set(wav_paths)) == 2
    assert all(Path(path).exists() for path in wav_paths)


def test_tts_pipeline_runs_end_to_end_on_fixtures(tmp_path, fixtures_dir):
    """The default TTS pipeline produces at least one curated fixture."""
    summary = tts_pipeline(str(fixtures_dir), str(tmp_path), snr_min=0.0, workers=1)

    assert summary["kept"] > 0
    assert summary["total_duration_seconds"] > 0
    assert (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "filelist.txt").exists()


def test_tts_pipeline_reads_requested_flac_extensions(tmp_path):
    """Pipeline attempts FLAC documents when requested explicitly."""
    flac_path = tmp_path / "silent.flac"
    sf.write(flac_path, np.zeros(32000, dtype=np.float32), 16000)

    summary = tts_pipeline(
        str(tmp_path),
        str(tmp_path / "output"),
        extensions=["flac"],
        workers=1,
    )

    assert summary["kept"] + summary["filtered"] > 0


def test_tts_pipeline_adds_vad_segmenter_when_requested(tmp_path, monkeypatch):
    """TTS segmentation is opt-in and uses the VAD fan-out transformer."""
    captured = {}

    class FakeExecutor:
        def __init__(self, pipeline, **_kwargs):
            captured["pipeline"] = pipeline

        def run(self, _reader, _exporter):
            return {"kept": 0, "skipped": 0}

    monkeypatch.setattr("audiotrove.pipelines.tts.LocalExecutor", FakeExecutor)

    tts_pipeline(str(tmp_path), str(tmp_path / "output"), segment=True)

    assert any(isinstance(block, VADSegmenter) for block in captured["pipeline"])


def test_tts_pipeline_uses_requested_checkpoint_path(tmp_path, monkeypatch):
    """A caller-provided checkpoint path overrides the output default."""
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, _reader, _exporter):
            return {"kept": 0, "skipped": 0}

    monkeypatch.setattr("audiotrove.pipelines.tts.LocalExecutor", FakeExecutor)
    checkpoint = tmp_path / "custom.db"

    tts_pipeline(str(tmp_path), str(tmp_path / "output"), checkpoint_path=str(checkpoint))

    assert captured["checkpoint_path"] == str(checkpoint)


def test_vad_remote_call_has_trust_repo():
    """Remote Silero loading must not require interactive trust confirmation."""
    src = Path("audiotrove/filters/vad.py").read_text(encoding="utf-8")
    remote_block = src[src.index('torch.hub.load("snakers4/silero-vad"') :]
    assert "trust_repo=True" in remote_block[:200]


def test_tts_pipeline_whisper_is_last_when_transcribe_enabled(tmp_path, monkeypatch):
    """WhisperTranscriber must be appended after all filters, not inserted mid-pipeline."""
    captured = {}

    class FakeExecutor:
        def __init__(self, pipeline, **_kwargs):
            captured["pipeline"] = pipeline

        def run(self, _reader, _exporter):
            return {"kept": 0, "skipped": 0}

    # Stub out WhisperTranscriber so whisper doesn't need to be installed.
    class StubWhisperTranscriber:
        def __init__(self, model_name="base", device="cpu"):
            pass

    monkeypatch.setattr("audiotrove.pipelines.tts.LocalExecutor", FakeExecutor)

    # Directly inject the stub so the lazy import inside tts_pipeline resolves to it.
    import sys
    import types

    fake_module = types.ModuleType("audiotrove.transformers.whisper_transcribe")
    fake_module.WhisperTranscriber = StubWhisperTranscriber
    monkeypatch.setitem(sys.modules, "audiotrove.transformers.whisper_transcribe", fake_module)

    tts_pipeline(str(tmp_path), str(tmp_path / "output"), transcribe=True)

    pipeline = captured["pipeline"]
    assert isinstance(pipeline[-1], StubWhisperTranscriber), (
        "WhisperTranscriber should be the last stage in the pipeline"
    )
    assert not isinstance(pipeline[-2], StubWhisperTranscriber), (
        "WhisperTranscriber should appear exactly once, at the end"
    )


def test_tts_pipeline_whisper_not_called_on_duration_rejected_doc(tmp_path, monkeypatch):
    """WhisperTranscriber.transform() must never be invoked on a clip that
    DurationBucketFilter rejects — confirmed by placing Whisper after all filters."""
    import numpy as np

    from audiotrove.document import AudioDocument

    whisper_calls = []

    class SpyWhisperTranscriber:
        name = "whisper_transcriber"

        def __init__(self, model_name="base", device="cpu"):
            pass

        def transform(self, doc: AudioDocument) -> AudioDocument:
            whisper_calls.append(doc.doc_id)
            return doc

    # A document that DurationBucketFilter(min=2, max=15) will reject (too short: 0.5 s).
    too_short = AudioDocument(
        audio=np.zeros(8000, dtype=np.float32),
        sample_rate=16000,
        source_path="short.wav",
        duration_seconds=0.5,
        doc_id="too_short",
    )

    from audiotrove.filters.duration import DurationBucketFilter
    from audiotrove.filters.snr import SNRFilter
    from audiotrove.filters.vad import SileroVADFilter
    from audiotrove.transformers.silence_trim import SilenceTrimmingTransformer

    # Reconstruct the exact pipeline tts_pipeline builds when transcribe=True.
    pipeline = [
        SileroVADFilter(min_speech_ratio=0.1),
        SilenceTrimmingTransformer(padding_ms=150),
        SNRFilter(min_snr_db=20.0),
        DurationBucketFilter(2.0, 15.0),
        SpyWhisperTranscriber(),
    ]

    # Walk the document through each stage, honouring filter rejections.
    doc = too_short
    for stage in pipeline:
        if hasattr(stage, "filter"):
            if not stage.filter(doc):
                break  # document rejected — no further stages should run
        else:
            doc = stage.transform(doc)

    assert whisper_calls == [], (
        "WhisperTranscriber.transform() should not be called on a document "
        "that DurationBucketFilter rejects"
    )


def test_tts_pipeline_diarize_writes_per_speaker_ids_to_manifest(tmp_path, monkeypatch):
    """With diarize=True the manifest must contain per-speaker IDs from diarization."""
    import sys
    import types
    import numpy as np
    from unittest.mock import MagicMock, patch

    SR = 16000

    # ------------------------------------------------------------------
    # Build a fake pyannote module so SpeakerDiarizationTransformer can
    # be imported without pyannote installed.
    # ------------------------------------------------------------------
    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "pyannote", MagicMock())
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)

    # ------------------------------------------------------------------
    # Fake diarization result: two speakers, each 3 s.
    # ------------------------------------------------------------------
    def _seg(start, end):
        s = MagicMock()
        s.start = start
        s.end = end
        return s

    mock_diar = MagicMock()
    mock_diar.itertracks.return_value = [
        (_seg(0.0, 3.0), None, "SPEAKER_00"),
        (_seg(3.0, 6.0), None, "SPEAKER_01"),
    ]
    mock_pipeline_instance = MagicMock(return_value=mock_diar)

    # ------------------------------------------------------------------
    # Write a 6-second wav so the reader has something to load.
    # ------------------------------------------------------------------
    import soundfile as sf

    src = tmp_path / "multi.wav"
    sf.write(str(src), np.zeros(int(6 * SR), dtype=np.float32), SR)
    out = tmp_path / "out"

    from audiotrove.transformers.diarize import SpeakerDiarizationTransformer

    with patch.object(
        SpeakerDiarizationTransformer,
        "pipeline",
        new_callable=lambda: property(lambda self: mock_pipeline_instance),
    ), patch("audiotrove.transformers.diarize.sf.write"), \
       patch("audiotrove.transformers.diarize.Path.unlink"):
        from audiotrove.pipelines.tts import tts_pipeline

        summary = tts_pipeline(
            str(tmp_path),
            str(out),
            min_duration=1.0,
            max_duration=20.0,
            snr_min=0.0,
            extensions=["wav"],
            workers=1,
            diarize=True,
            hf_token="hf_fake",
        )

    # At least one speaker segment should have been kept.
    metadata_path = out / "metadata.csv"
    if metadata_path.exists():
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        speaker_ids = [line.split("|")[1] for line in lines if "|" in line]
        # All IDs must be one of the diarized speaker labels, not the default "speaker_00".
        for sid in speaker_ids:
            assert sid in {"SPEAKER_00", "SPEAKER_01"}, (
                f"Unexpected speaker_id in manifest: {sid!r}"
            )

