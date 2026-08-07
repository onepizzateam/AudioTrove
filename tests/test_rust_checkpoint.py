"""Tests for the optional Rust ``audiotrove_core`` extension.

The entire module is skipped when the Rust extension is not built, so the pure
Python package continues to test/pass without it.
"""

import threading

import pytest

audiotrove_core = pytest.importorskip("audiotrove_core")


def test_checkpoint_store_creates_db(tmp_path):
    db = tmp_path / "checkpoint.db"
    store = audiotrove_core.CheckpointStore(str(db))
    assert store is not None
    assert db.exists()


def test_is_processed_false_for_unknown(tmp_path):
    store = audiotrove_core.CheckpointStore(str(tmp_path / "c.db"))
    assert store.is_processed("never-seen") is False


def test_mark_processed_roundtrip(tmp_path):
    store = audiotrove_core.CheckpointStore(str(tmp_path / "c.db"))
    assert store.is_processed("doc-1") is False
    store.mark_processed("doc-1")
    assert store.is_processed("doc-1") is True


def test_mark_batch_records_all(tmp_path):
    store = audiotrove_core.CheckpointStore(str(tmp_path / "c.db"))
    ids = [f"doc-{i}" for i in range(50)]
    store.mark_batch(ids)
    for doc_id in ids:
        assert store.is_processed(doc_id) is True


def test_mark_batch_empty(tmp_path):
    store = audiotrove_core.CheckpointStore(str(tmp_path / "c.db"))
    # Should not raise on an empty batch.
    store.mark_batch([])


def test_concurrent_writes_do_not_corrupt(tmp_path):
    store = audiotrove_core.CheckpointStore(str(tmp_path / "c.db"))

    def worker(offset):
        for i in range(100):
            store.mark_processed(f"doc-{offset}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for offset in range(4):
        for i in range(100):
            assert store.is_processed(f"doc-{offset}-{i}") is True
