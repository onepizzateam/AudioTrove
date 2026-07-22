from audiotrove.executor.local import LocalExecutor, _worker_process_doc
from audiotrove.document import AudioDocument
import numpy as np
import sqlite3


class DummyTransformer2:
    name = "dummy_transformer2"

    def transform(self, doc):
        return doc


def _make_doc(num=1):
    audio = np.zeros(1600, dtype=float)
    return AudioDocument(
        audio=audio,
        sample_rate=16000,
        source_path=f"x{num}.wav",
        duration_seconds=0.1,
        doc_id=f"doc-{num}",
    )


def test_parallel_run_with_fake_executor(monkeypatch, tmp_path):
    # Fake ProcessPoolExecutor that executes tasks synchronously
    from audiotrove import executor as executor_mod

    class FakeFuture:
        def __init__(self, res):
            self._res = res

        def result(self):
            return self._res

    class FakeExecutor:
        def __init__(self, max_workers, **kwargs):
            self.max_workers = max_workers
            self._futures = []

        def submit(self, fn, doc, pipeline):
            res = fn(doc, pipeline)
            fut = FakeFuture(res)
            self._futures.append(fut)
            return fut

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(executor_mod.local, "ProcessPoolExecutor", FakeExecutor)
    # as_completed should iterate over the dict of futures
    monkeypatch.setattr(executor_mod.local, "as_completed", lambda futs: iter(futs.keys()))

    docs = [_make_doc(1), _make_doc(2)]

    def reader_gen():
        for d in docs:
            yield d

    writer_writes = []

    class W:
        def write(self, d):
            writer_writes.append(d)

    ex = LocalExecutor(
        pipeline=[DummyTransformer2()], checkpoint_path=str(tmp_path / "ckpt.db"), num_workers=2
    )
    stats = ex.run(reader_gen(), W())

    assert stats["processed"] == 2
    assert stats["kept"] >= 2
    # ensure checkpoint db has both entries
    conn = sqlite3.connect(str(tmp_path / "ckpt.db"))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM processed")
    cnt = cur.fetchone()[0]
    conn.close()
    assert cnt >= 2
