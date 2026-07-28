import sqlite3
from click.testing import CliRunner

from audiotrove.base import AudioFanOutTransformer
from audiotrove.document import AudioDocument
from audiotrove.cli import main as cli


class DummyFilter:
    name = "dummy_filter"

    def __init__(self, raise_on=False, return_value=True):
        self._raise = raise_on
        self._ret = return_value

    def filter(self, doc):
        if self._raise:
            raise RuntimeError("filter-error")
        return self._ret


class DummyTransformer:
    name = "dummy_transformer"

    def __init__(self, raise_on=False):
        self._raise = raise_on

    def transform(self, doc):
        if self._raise:
            raise RuntimeError("transform-error")
        # return same doc
        return doc


class DummyFanOut(AudioFanOutTransformer):
    name = "dummy_fanout"

    def transform(self, doc):
        # emit two docs
        d1 = AudioDocument(
            audio=doc.audio,
            sample_rate=doc.sample_rate,
            source_path=doc.source_path,
            duration_seconds=doc.duration_seconds,
            doc_id=doc.doc_id + "-1",
        )
        d2 = AudioDocument(
            audio=doc.audio,
            sample_rate=doc.sample_rate,
            source_path=doc.source_path,
            duration_seconds=doc.duration_seconds,
            doc_id=doc.doc_id + "-2",
        )
        return [d1, d2]


def _make_doc():
    import numpy as np

    audio = np.zeros(1600, dtype=float)
    return AudioDocument(
        audio=audio, sample_rate=16000, source_path="x.wav", duration_seconds=0.1, doc_id="doc-123"
    )


def test_worker_process_doc_filter_exception():
    from audiotrove.executor.local import _worker_process_doc

    doc = _make_doc()
    pipeline = [DummyFilter(raise_on=True)]
    docs, keep, error, error_block, _ = _worker_process_doc(doc, pipeline)
    assert not keep
    assert error is not None
    assert "filter-error" in error


def test_worker_process_doc_fanout_and_transform():
    from audiotrove.executor.local import _worker_process_doc

    doc = _make_doc()
    pipeline = [DummyFanOut(), DummyTransformer()]
    docs, keep, error, error_block, _ = _worker_process_doc(doc, pipeline)
    assert keep
    assert error is None
    assert isinstance(docs, list)
    assert len(docs) == 2


def test_worker_process_doc_transform_exception():
    from audiotrove.executor.local import _worker_process_doc

    class BadTransform:
        name = "bad_transform"

        def transform(self, doc):
            raise RuntimeError("boom")

    doc = _make_doc()
    docs, keep, error, error_block, _ = _worker_process_doc(doc, [BadTransform()])
    assert not keep
    assert error_block == "bad_transform"


def test_run_sequential_marks_processed_and_handles_errors(tmp_path):
    from audiotrove.executor.local import LocalExecutor

    # reader yields one doc
    doc = _make_doc()

    def reader_gen():
        yield doc

    # writer that records writes
    class W:
        def __init__(self):
            self.writes = []

        def write(self, d):
            self.writes.append(d)

    writer = W()

    # Pipeline with a filter that raises -> should increment errors
    pipeline = [DummyFilter(raise_on=True)]
    ckpt = str(tmp_path / "ckpt.db")
    ex = LocalExecutor(pipeline=pipeline, checkpoint_path=ckpt, num_workers=1)
    stats = ex.run(reader_gen(), writer)

    assert stats["processed"] == 1
    assert stats["errors"] >= 1
    # DB should exist and contain the doc id as processed (marked even on error)
    conn = sqlite3.connect(ckpt)
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM processed")
    rows = cur.fetchall()
    conn.close()
    assert ("doc-123",) in rows


def test_cli_curate_inserts_segmenter_and_prints(monkeypatch, fixtures_dir, tmp_path):
    # Monkeypatch filters to predictable simple classes
    class FakeVAD:
        def __init__(self, min_speech_ratio=0.0):
            self.name = "silero_vad"

    class FakeSNR:
        def __init__(self, min_snr_db=0.0):
            self.name = "snr_filter"

    monkeypatch.setattr("audiotrove.filters.vad.SileroVADFilter", FakeVAD)
    monkeypatch.setattr("audiotrove.filters.snr.SNRFilter", FakeSNR)

    # Monkeypatch LocalExecutor to avoid heavy processing
    class FakeExecutor:
        def __init__(self, pipeline, checkpoint_path, num_workers):
            self.pipeline = pipeline
            self.checkpoint_path = checkpoint_path
            self.num_workers = num_workers

        def run(self, reader, writer):
            return {"processed": 1, "kept": 1, "skipped": 0}

    monkeypatch.setattr("audiotrove.executor.local.LocalExecutor", FakeExecutor)

    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        cli.cli, ["curate", str(fixtures_dir), str(out_dir), "--segment", "--workers", "1"]
    )
    assert result.exit_code == 0
    assert (
        "Segmentation: enabled" in result.output
        or "Segmentation: enabled (VADSegmenter)" in result.output
    )


def test_cli_inspect_shows_stats(fixtures_dir, monkeypatch):
    # Monkeypatch LocalAudioReader to avoid fsspec/torchaudio dependencies
    from audiotrove.io import readers as readers_mod

    class FakeReader:
        def __init__(self, patterns, min_duration_seconds=0.0, max_duration_seconds=None):
            import numpy as np

            self._docs = [
                AudioDocument(
                    audio=np.zeros(1600),
                    sample_rate=16000,
                    source_path="a.wav",
                    duration_seconds=0.1,
                    doc_id="a",
                ),
                AudioDocument(
                    audio=np.zeros(3200),
                    sample_rate=16000,
                    source_path="b.wav",
                    duration_seconds=0.2,
                    doc_id="b",
                ),
            ]

        def __iter__(self):
            return iter(self._docs)

    monkeypatch.setattr(readers_mod, "LocalAudioReader", FakeReader)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["inspect", str(fixtures_dir), "--limit", "2"])
    assert result.exit_code == 0
    assert "Audio Directory Stats" in result.output
