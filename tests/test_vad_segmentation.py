"""Tests for VAD Segmentation (AudioFanOutTransformer)."""

import numpy as np
from audiotrove.filters.vad import VADSegmenter
from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id


def test_vad_segmenter_basic():
    """VADSegmenter should split audio into speech segments."""
    # Create a simple 10-second fixture: 5s speech + 5s silence
    sr = 16000
    duration = 10
    sr * duration

    # 0-5s: speech-like (sine wave)
    speech_part = 0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 5) / sr)
    # 5-10s: near-silence
    silence_part = np.random.normal(0, 0.001, sr * 5)
    audio = np.concatenate([speech_part, silence_part]).astype(np.float32)

    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="test_speech_silence.wav",
        duration_seconds=duration,
        doc_id=make_doc_id("test_speech_silence.wav"),
    )

    segmenter = VADSegmenter()
    segments = segmenter.transform(doc)

    # Should have at least one speech segment
    assert len(segments) > 0

    # All segments should be shorter than original
    for seg in segments:
        assert seg.duration_seconds < doc.duration_seconds
        assert len(seg.audio) < len(doc.audio)

    # Segments should have correct metadata
    for seg in segments:
        assert "parent_doc_id" in seg.metadata
        assert seg.metadata["parent_doc_id"] == doc.doc_id
        assert "segment_start_sample" in seg.metadata
        assert "segment_end_sample" in seg.metadata
        assert "vad_backend" in seg.metadata


def test_vad_segmenter_speech_detection():
    """VADSegmenter should segment detected speech regions correctly."""
    sr = 16000

    # Create 10-second audio: first 5s is strong speech-like signal
    speech_samples = sr * 5
    silence_samples = sr * 5

    # Strong speech signal (similar to test_vad_segmenter_basic which passes)
    speech = 0.3 * np.sin(2 * np.pi * 300 * np.arange(speech_samples) / sr)
    silence = np.random.normal(0, 0.001, silence_samples)

    audio = np.concatenate([speech, silence]).astype(np.float32)

    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="speech_detection.wav",
        duration_seconds=10.0,
        doc_id=make_doc_id("speech_detection.wav"),
    )

    segmenter = VADSegmenter()
    segments = segmenter.transform(doc)

    # Should detect at least one speech segment
    assert len(segments) >= 1

    # Each segment should be a valid AudioDocument
    for seg in segments:
        assert isinstance(seg, AudioDocument)
        assert seg.audio.dtype == np.float32
        assert seg.sample_rate == sr
        assert "parent_doc_id" in seg.metadata


def test_vad_segmenter_deterministic_doc_id():
    """VADSegmenter should generate deterministic doc_ids for segments."""
    sr = 16000
    audio = 0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr).astype(np.float32)
    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="test_deterministic.wav",
        duration_seconds=3.0,
        doc_id=make_doc_id("test_deterministic.wav"),
    )

    segmenter = VADSegmenter()

    # Process the same doc twice
    segments1 = segmenter.transform(doc)
    segments2 = segmenter.transform(doc)

    # Should get same segments in same order
    assert len(segments1) == len(segments2)
    for s1, s2 in zip(segments1, segments2):
        assert s1.doc_id == s2.doc_id
        assert s1.metadata["segment_start_sample"] == s2.metadata["segment_start_sample"]
        assert s1.metadata["segment_end_sample"] == s2.metadata["segment_end_sample"]


def test_vad_segmenter_no_speech():
    """VADSegmenter should return empty list when no speech detected."""
    sr = 16000
    # Pure silence
    audio = np.random.normal(0, 0.0001, sr * 5).astype(np.float32)

    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="silence.wav",
        duration_seconds=5.0,
        doc_id=make_doc_id("silence.wav"),
    )

    segmenter = VADSegmenter()
    segments = segmenter.transform(doc)

    # Should return empty list for silence
    assert len(segments) == 0


def test_vad_segmenter_metadata_preservation():
    """VADSegmenter should preserve parent metadata in segments."""
    sr = 16000
    audio = 0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 2) / sr).astype(np.float32)

    parent_metadata = {"source": "test_dataset", "language": "en", "speaker_id": "12345"}

    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="test_metadata.wav",
        duration_seconds=2.0,
        doc_id=make_doc_id("test_metadata.wav"),
        metadata=parent_metadata,
    )

    segmenter = VADSegmenter()
    segments = segmenter.transform(doc)

    # All segments should include parent metadata
    for seg in segments:
        for key, val in parent_metadata.items():
            assert seg.metadata.get(key) == val


def test_vad_segmenter_audio_integrity():
    """VADSegmenter segments should contain correct audio data."""
    sr = 16000

    # Create 4-second audio: 2s sine wave, 2s silence
    speech_samples = int(sr * 2)
    silence_samples = int(sr * 2)

    freq = 400
    speech = 0.3 * np.sin(2 * np.pi * freq * np.arange(speech_samples) / sr)
    silence = np.zeros(silence_samples)
    audio = np.concatenate([speech, silence]).astype(np.float32)

    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="test_integrity.wav",
        duration_seconds=4.0,
        doc_id=make_doc_id("test_integrity.wav"),
    )

    segmenter = VADSegmenter()
    segments = segmenter.transform(doc)

    assert len(segments) > 0

    # Each segment should have the correct audio data
    for seg in segments:
        start = seg.metadata["segment_start_sample"]
        end = seg.metadata["segment_end_sample"]

        expected_audio = audio[start:end]
        np.testing.assert_array_almost_equal(seg.audio, expected_audio)


def test_vad_segmenter_with_executor():
    """VADSegmenter should work correctly in executor pipeline."""
    from audiotrove.executor.local import LocalExecutor
    from audiotrove.base import AudioFilter

    # Create a filter that counts segments
    class SegmentCountingFilter(AudioFilter):
        @property
        def name(self):
            return "segment_counter"

        def filter(self, doc):
            # Count all documents as speech (for this test)
            return True

    sr = 16000
    audio = 0.3 * np.sin(2 * np.pi * 300 * np.arange(sr * 3) / sr).astype(np.float32)
    doc = AudioDocument(
        audio=audio,
        sample_rate=sr,
        source_path="test_executor.wav",
        duration_seconds=3.0,
        doc_id=make_doc_id("test_executor.wav"),
    )

    # Mock reader and writer
    class MockReader:
        def __iter__(self):
            yield doc

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, d):
            self.written.append(d)

    reader = MockReader()
    writer = MockWriter()

    # Build pipeline: segment -> count
    pipeline = [VADSegmenter(), SegmentCountingFilter()]
    executor = LocalExecutor(pipeline=pipeline, num_workers=1)

    stats = executor.run(reader, writer)

    # Should process the original doc but write multiple segments
    assert stats["processed"] == 1
    assert len(writer.written) > 1  # Multiple segments written

    # All written docs should be segments
    for written_doc in writer.written:
        assert "parent_doc_id" in written_doc.metadata


def test_executor_keeps_passing_fanout_documents_when_later_one_fails():
    """A rejected fan-out child must not discard earlier accepted children."""
    from audiotrove.base import AudioFanOutTransformer, AudioFilter
    from audiotrove.executor.local import LocalExecutor

    class FanOut(AudioFanOutTransformer):
        name = "fan_out"

        def transform(self, document):
            return [
                AudioDocument(
                    audio=document.audio,
                    sample_rate=document.sample_rate,
                    source_path=document.source_path,
                    duration_seconds=document.duration_seconds,
                    doc_id=f"{document.doc_id}-{index}",
                )
                for index in range(3)
            ]

    class RejectLast(AudioFilter):
        name = "reject_last"

        def filter(self, document):
            return not document.doc_id.endswith("-2")

    class Reader:
        def __iter__(self):
            yield AudioDocument(
                audio=np.ones(16, dtype=np.float32),
                sample_rate=16,
                source_path="fanout.wav",
                duration_seconds=1.0,
                doc_id="parent",
            )

    class Writer:
        def __init__(self):
            self.written = []

        def write(self, document):
            self.written.append(document)

    writer = Writer()
    stats = LocalExecutor([FanOut(), RejectLast()], num_workers=1).run(Reader(), writer)

    assert stats["kept"] == 2
    assert [document.doc_id for document in writer.written] == ["parent-0", "parent-1"]
