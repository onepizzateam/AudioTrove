"""Tests for LocalExecutor."""
import pytest
from pathlib import Path
import tempfile
from audiotrove.executor.local import LocalExecutor
from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument


class TestFilter(AudioFilter):
    """Simple test filter that keeps documents with metadata flag."""
    
    @property
    def name(self) -> str:
        return "test_filter"
    
    def filter(self, doc: AudioDocument) -> bool:
        doc.metadata['filtered'] = True
        return doc.metadata.get('keep', True)


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
    assert stats['processed'] == 3
    assert stats['kept'] == 3
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
    
    assert stats1['processed'] == 3
    assert checkpoint_db.exists()
    
    # Second run: same 3 documents should be skipped (already in checkpoint)
    reader2 = MockReader(count=3)
    writer2 = MockWriter()
    executor2 = LocalExecutor(pipeline=[], checkpoint_path=str(checkpoint_db))
    stats2 = executor2.run(reader2, writer2)
    
    assert stats2['processed'] == 0
    assert stats2['skipped'] == 3


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
    assert stats['processed'] == 2
    assert stats['skipped'] == 1
