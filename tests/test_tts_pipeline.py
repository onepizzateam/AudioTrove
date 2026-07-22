"""Tests for the TTS curation pipeline."""

from pathlib import Path

import numpy as np
import soundfile as sf

from audiotrove.document import AudioDocument
from audiotrove.exporters.tts_manifest import TTSManifestExporter
from audiotrove.filters.duration import DurationBucketFilter
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


def test_vad_remote_call_has_trust_repo():
    """Remote Silero loading must not require interactive trust confirmation."""
    src = Path("audiotrove/filters/vad.py").read_text(encoding="utf-8")
    remote_block = src[src.index('torch.hub.load("snakers4/silero-vad"') :]
    assert "trust_repo=True" in remote_block[:200]
