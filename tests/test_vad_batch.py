"""Tests for SileroVADFilter.filter_batch.

``filter_batch`` amortises the Silero inference lock across a batch of docs; it
must produce the same decision and metadata as calling ``filter`` per document.
These run on the energy-based fallback when Silero weights aren't cached, which
is exactly the CI default, so the equivalence must hold on either backend.
"""

import copy

from audiotrove.filters.vad import SileroVADFilter


def _clone(doc):
    return copy.deepcopy(doc)


def test_filter_batch_empty_returns_empty():
    vad = SileroVADFilter(min_speech_ratio=0.1)
    assert vad.filter_batch([]) == []


def test_filter_batch_matches_per_doc_filter(speech_clean, silence):
    docs = [speech_clean, silence]

    batch_vad = SileroVADFilter(min_speech_ratio=0.1)
    batch_results = batch_vad.filter_batch([_clone(d) for d in docs])

    single_vad = SileroVADFilter(min_speech_ratio=0.1)
    single_results = [single_vad.filter(_clone(d)) for d in docs]

    assert len(batch_results) == len(docs)
    assert all(isinstance(r, bool) for r in batch_results)
    assert batch_results == single_results


def test_filter_batch_silence_is_rejected(silence):
    vad = SileroVADFilter(min_speech_ratio=0.1)
    (result,) = vad.filter_batch([silence])
    assert result is False


def test_filter_batch_populates_metadata(speech_clean, silence):
    vad = SileroVADFilter(min_speech_ratio=0.1)
    docs = [_clone(speech_clean), _clone(silence)]
    vad.filter_batch(docs)
    for doc in docs:
        assert "vad_speech_ratio" in doc.metadata
        assert "vad_speech_timestamps" in doc.metadata
        assert "vad_backend" in doc.metadata


def test_filter_batch_one_failure_does_not_sink_batch(speech_clean, silence):
    """A malformed document degrades to False without breaking the batch."""

    class Broken:
        source_path = "broken"
        # Missing .audio / .metadata so _run_silero raises internally.

    vad = SileroVADFilter(min_speech_ratio=0.1)
    results = vad.filter_batch([_clone(silence), Broken(), _clone(speech_clean)])
    assert len(results) == 3
    assert results[1] is False  # the broken doc
