"""Tests for SpeakerDiarizationTransformer."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audiotrove.document import AudioDocument

try:
    import pyannote.audio  # noqa: F401
    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PYANNOTE_AVAILABLE,
    reason="pyannote.audio optional diarization extra is not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000

def _make_transformer(hf_token: str = "hf_test", **kwargs):
    # Diarization is an optional extra; skip (don't fail) when pyannote is
    # absent, matching how the rest of the suite guards optional deps.
    pytest.importorskip("pyannote.audio")
    from audiotrove.transformers.diarize import SpeakerDiarizationTransformer
    return SpeakerDiarizationTransformer(hf_token=hf_token, **kwargs)

def _make_doc(duration: float = 4.0, doc_id: str = "test_doc") -> AudioDocument:
    """Return a synthetic AudioDocument for the given duration."""
    n = int(duration * SR)
    return AudioDocument(
        audio=np.random.default_rng(0).standard_normal(n).astype(np.float32),
        sample_rate=SR,
        source_path="recording.wav",
        duration_seconds=duration,
        doc_id=doc_id,
        metadata={"session": "s1"},
    )


def _fake_segment(start: float, end: float):
    """Return a minimal object that looks like a pyannote Segment."""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    return seg


def _make_diarization_result(turns: list[tuple[float, float, str]]):
    """Return a mock diarization object whose itertracks() yields the given turns."""
    mock_diar = MagicMock()
    mock_diar.itertracks.return_value = [
        (_fake_segment(start, end), None, label) for start, end, label in turns
    ]
    return mock_diar

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpeakerDiarizationTransformer:
    def test_three_speaker_turns_produce_three_children(self, tmp_path):
        """3-speaker mock diarization → 3 child AudioDocuments."""
        transformer = _make_transformer()

        doc = _make_doc(duration=9.0, doc_id="parent")
        turns = [
            (0.0, 3.0, "SPEAKER_00"),
            (3.0, 6.0, "SPEAKER_01"),
            (6.0, 9.0, "SPEAKER_02"),
        ]
        mock_result = _make_diarization_result(turns)

        # Patch the pipeline property so no HF download occurs.
        with patch.object(type(transformer), "pipeline", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_result))):
            # Also patch soundfile.write and Path.unlink so no real disk I/O needed.
            with patch("audiotrove.transformers.diarize.sf.write"), \
                 patch("audiotrove.transformers.diarize.Path.unlink"):
                children = transformer.transform(doc)

        assert len(children) == 3

    def test_child_doc_ids_are_deterministic(self):
        """doc_id format: {parent}__{speaker}__{start_ms}_{end_ms}."""
        transformer = _make_transformer()
        doc = _make_doc(duration=6.0, doc_id="parent42")

        turns = [(1.0, 3.5, "SPEAKER_00")]
        mock_result = _make_diarization_result(turns)

        with patch.object(type(transformer), "pipeline", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_result))), \
             patch("audiotrove.transformers.diarize.sf.write"), \
             patch("audiotrove.transformers.diarize.Path.unlink"):
            children = transformer.transform(doc)

        expected_id = "parent42__SPEAKER_00__1000_3500"
        assert children[0].doc_id == expected_id

    def test_child_metadata_carries_speaker_id_and_provenance(self):
        """Children must have speaker_id, parent_doc_id, segment_start, segment_end."""
        transformer = _make_transformer()
        doc = _make_doc(duration=4.0, doc_id="p1")
        turns = [(0.5, 2.5, "SPEAKER_01")]
        mock_result = _make_diarization_result(turns)

        with patch.object(type(transformer), "pipeline", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_result))), \
             patch("audiotrove.transformers.diarize.sf.write"), \
             patch("audiotrove.transformers.diarize.Path.unlink"):
            children = transformer.transform(doc)

        child = children[0]
        assert child.metadata["speaker_id"] == "SPEAKER_01"
        assert child.metadata["parent_doc_id"] == "p1"
        assert child.metadata["segment_start"] == pytest.approx(0.5)
        assert child.metadata["segment_end"] == pytest.approx(2.5)
        # Parent metadata should be inherited.
        assert child.metadata["session"] == "s1"

    def test_child_audio_is_sliced_correctly(self):
        """Child audio must be the sample-accurate slice of the parent array."""
        transformer = _make_transformer()
        doc = _make_doc(duration=4.0, doc_id="slice_test")
        # Turn covers samples 16000-48000 (1.0 s – 3.0 s at 16 kHz).
        turns = [(1.0, 3.0, "SPEAKER_00")]
        mock_result = _make_diarization_result(turns)

        with patch.object(type(transformer), "pipeline", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_result))), \
             patch("audiotrove.transformers.diarize.sf.write"), \
             patch("audiotrove.transformers.diarize.Path.unlink"):
            children = transformer.transform(doc)

        expected_audio = doc.audio[16000:48000]
        np.testing.assert_array_equal(children[0].audio, expected_audio)
        assert children[0].duration_seconds == pytest.approx(2.0)
        assert children[0].sample_rate == SR
        assert children[0].source_path == "recording.wav"

    def test_empty_diarization_returns_empty_list(self):
        """No detected speech → transform() returns []."""
        transformer = _make_transformer()
        doc = _make_doc(duration=2.0)
        mock_result = _make_diarization_result([])  # no turns

        with patch.object(type(transformer), "pipeline", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_result))), \
             patch("audiotrove.transformers.diarize.sf.write"), \
             patch("audiotrove.transformers.diarize.Path.unlink"):
            result = transformer.transform(doc)

        assert result == []

    def test_getstate_excludes_pipeline(self):
        """Pickling must not serialise the loaded pyannote pipeline."""
        transformer = _make_transformer()
        # Simulate a loaded pipeline.
        transformer._pipeline = MagicMock(name="loaded_pipeline")

        state = transformer.__getstate__()
        assert state["_pipeline"] is None

    def test_setstate_restores_config_but_not_pipeline(self):
        """After unpickling, hf_token/model_name are present; _pipeline is None."""
        transformer = _make_transformer(hf_token="hf_abc", model_name="pyannote/speaker-diarization-3.1")
        transformer._pipeline = MagicMock()

        data = pickle.dumps(transformer)
        restored = pickle.loads(data)

        assert restored.hf_token == "hf_abc"
        assert restored.model_name == "pyannote/speaker-diarization-3.1"
        assert restored._pipeline is None

    def test_missing_hf_token_raises_in_tts_pipeline(self, tmp_path, monkeypatch):
        """tts_pipeline(diarize=True) without hf_token must raise ValueError immediately."""
        from audiotrove.pipelines.tts import tts_pipeline

        with pytest.raises(ValueError, match="hf_token"):
            tts_pipeline(
                str(tmp_path),
                str(tmp_path / "out"),
                diarize=True,
                hf_token=None,
            )
