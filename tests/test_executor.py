"""Tests for LocalExecutor."""

from audiotrove.executor.local import LocalExecutor
from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument


class TestFilter(AudioFilter):
    """Simple test filter that keeps documents with metadata flag."""

    @property
    def name(self) -> str:
        return "test_filter"

    def filter(self, doc: AudioDocument) -> bool:
        doc.metadata["filtered"] = True
        return doc.metadata.get("keep", True)


class CountingFilter(AudioFilter):
    """A simple stateless filter for testing parallel execution."""

    @property
    def name(self) -> str:
        return "counting_filter"

    def filter(self, doc: AudioDocument) -> bool:
        doc.metadata["count"] = 1
        return True


class FailingFilter(AudioFilter):
    """Raises an exception on even-numbered documents."""

    @property
    def name(self) -> str:
        return "failing_filter"

    def filter(self, doc: AudioDocument) -> bool:
        # Fail on even-numbered docs
        if int(doc.source_path.replace("test", "").replace(".wav", "")) % 2 == 0:
            raise ValueError(f"Intentional error on {doc.source_path}")
        return True


def test_executor_runs_pipeline():
    """LocalExecutor should run a pipeline over documents."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    # Create mock reader and writer
    class MockReader:
        def __iter__(self):
            audio = np.zeros(16000, dtype=np.float32)
            for i in range(3):
                yield AudioDocument(
                    audio=audio,
                    sample_rate=16000,
                    source_path=f"test{i}.wav",
                    duration_seconds=1.0,
                    doc_id=make_doc_id(f"test{i}.wav"),
                )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            self.written.append(doc)

    reader = MockReader()
    writer = MockWriter()
    pipeline = [TestFilter()]
    executor = LocalExecutor(pipeline=pipeline)

    stats = executor.run(reader, writer)

    # All 3 documents should be processed and kept
    assert stats["processed"] == 3
    assert stats["kept"] == 3
    assert len(writer.written) == 3


def test_executor_with_checkpoint(tmp_path):
    """LocalExecutor should checkpoint processed doc_ids."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    checkpoint_db = tmp_path / "checkpoint.db"

    class MockReader:
        def __init__(self, count=3):
            self.count = count

        def __iter__(self):
            audio = np.zeros(16000, dtype=np.float32)
            for i in range(self.count):
                yield AudioDocument(
                    audio=audio,
                    sample_rate=16000,
                    source_path=f"test{i}.wav",
                    duration_seconds=1.0,
                    doc_id=make_doc_id(f"test{i}.wav"),
                )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            self.written.append(doc)

    # First run: process all 3 documents
    reader1 = MockReader(count=3)
    writer1 = MockWriter()
    executor1 = LocalExecutor(pipeline=[], checkpoint_path=str(checkpoint_db))
    stats1 = executor1.run(reader1, writer1)

    assert stats1["processed"] == 3
    assert checkpoint_db.exists()

    # Second run: same 3 documents should be skipped (already in checkpoint)
    reader2 = MockReader(count=3)
    writer2 = MockWriter()
    executor2 = LocalExecutor(pipeline=[], checkpoint_path=str(checkpoint_db))
    stats2 = executor2.run(reader2, writer2)

    assert stats2["processed"] == 0
    assert stats2["skipped"] == 3


def test_executor_skips_none_documents():
    """LocalExecutor should skip None documents from reader."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    class MockReaderWithNone:
        def __iter__(self):
            audio = np.zeros(16000, dtype=np.float32)
            yield AudioDocument(
                audio=audio,
                sample_rate=16000,
                source_path="test0.wav",
                duration_seconds=1.0,
                doc_id=make_doc_id("test0.wav"),
            )
            yield None  # Reader returns None on error
            yield AudioDocument(
                audio=audio,
                sample_rate=16000,
                source_path="test1.wav",
                duration_seconds=1.0,
                doc_id=make_doc_id("test1.wav"),
            )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            self.written.append(doc)

    reader = MockReaderWithNone()
    writer = MockWriter()
    executor = LocalExecutor(pipeline=[])

    stats = executor.run(reader, writer)

    # Should process 2 documents, skip 1 None
    assert stats["processed"] == 2
    assert stats["skipped"] == 1


def test_executor_parallel_matches_sequential():
    """Parallel execution should produce identical results to sequential."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    class MockReader:
        def __init__(self):
            self.call_count = 0

        def __iter__(self):
            audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
            for i in range(10):
                yield AudioDocument(
                    audio=audio.copy(),
                    sample_rate=16000,
                    source_path=f"test{i}.wav",
                    duration_seconds=1.0,
                    doc_id=make_doc_id(f"test{i}.wav"),
                )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            # Store a copy with audio for comparison
            self.written.append((doc.doc_id, doc.metadata.copy()))

    # Run sequentially
    reader_seq = MockReader()
    writer_seq = MockWriter()
    executor_seq = LocalExecutor(pipeline=[CountingFilter()], num_workers=1)
    stats_seq = executor_seq.run(reader_seq, writer_seq)

    # Run in parallel
    reader_par = MockReader()
    writer_par = MockWriter()
    executor_par = LocalExecutor(pipeline=[CountingFilter()], num_workers=4)
    stats_par = executor_par.run(reader_par, writer_par)

    # Stats should match
    assert stats_seq["processed"] == stats_par["processed"]
    assert stats_seq["kept"] == stats_par["kept"]
    assert stats_seq["skipped"] == stats_par["skipped"]
    assert stats_seq["errors"] == stats_par["errors"]

    # Written docs should match (same set of doc_ids)
    seq_ids = {doc_id for doc_id, _ in writer_seq.written}
    par_ids = {doc_id for doc_id, _ in writer_par.written}
    assert seq_ids == par_ids


def test_executor_parallel_checkpoint_still_works(tmp_path):
    """Checkpoint should work correctly with parallel execution."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    checkpoint_db = tmp_path / "checkpoint.db"

    class MockReader:
        def __init__(self, count=5):
            self.count = count

        def __iter__(self):
            audio = np.zeros(16000, dtype=np.float32)
            for i in range(self.count):
                yield AudioDocument(
                    audio=audio,
                    sample_rate=16000,
                    source_path=f"test{i}.wav",
                    duration_seconds=1.0,
                    doc_id=make_doc_id(f"test{i}.wav"),
                )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            self.written.append(doc.doc_id)

    # First run with num_workers=2
    reader1 = MockReader(count=5)
    writer1 = MockWriter()
    executor1 = LocalExecutor(pipeline=[], checkpoint_path=str(checkpoint_db), num_workers=2)
    stats1 = executor1.run(reader1, writer1)

    assert stats1["processed"] == 5
    assert len(writer1.written) == 5

    # Second run: same documents should be skipped
    reader2 = MockReader(count=5)
    writer2 = MockWriter()
    executor2 = LocalExecutor(pipeline=[], checkpoint_path=str(checkpoint_db), num_workers=2)
    stats2 = executor2.run(reader2, writer2)

    assert stats2["processed"] == 0
    assert stats2["skipped"] == 5
    assert len(writer2.written) == 0


def test_executor_parallel_error_isolation():
    """Errors in one doc should not prevent processing of others."""
    import numpy as np
    from audiotrove.utils.hashing import make_doc_id

    class MockReader:
        def __iter__(self):
            audio = np.zeros(16000, dtype=np.float32)
            for i in range(5):
                yield AudioDocument(
                    audio=audio,
                    sample_rate=16000,
                    source_path=f"test{i}.wav",
                    duration_seconds=1.0,
                    doc_id=make_doc_id(f"test{i}.wav"),
                )

    class MockWriter:
        def __init__(self):
            self.written = []

        def write(self, doc):
            self.written.append(doc.source_path)

    reader = MockReader()
    writer = MockWriter()
    executor = LocalExecutor(pipeline=[FailingFilter()], num_workers=2)
    stats = executor.run(reader, writer)

    # With 5 docs (0-4), even-indexed (0, 2, 4) fail = 3 errors
    # Odd-indexed (1, 3) pass = 2 kept
    assert stats["processed"] == 5
    assert stats["errors"] == 3
    assert stats["kept"] == 2
    assert stats["skipped"] == 3
    assert "failing_filter" in stats["errors_by_filter"]
    assert stats["errors_by_filter"]["failing_filter"] == 3
    # Only non-erroring docs (1, 3) should be written
    assert len(writer.written) == 2
