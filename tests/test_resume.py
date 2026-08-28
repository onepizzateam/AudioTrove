"""Interruption/resume coverage for the local checkpoint contract."""

import multiprocessing
import time

import numpy as np

from audiotrove.document import AudioDocument
from audiotrove.executor.local import LocalExecutor


class SlowPass:
    name = "slow_pass"

    def filter(self, doc):
        time.sleep(0.15)
        return True


class Records:
    def __init__(self, count):
        self.docs = [
            AudioDocument(np.zeros(1600, dtype=np.float32), 16000, f"{i}.wav", 0.1, str(i))
            for i in range(count)
        ]

    def __iter__(self):
        return iter(self.docs)


class Collect:
    def __init__(self):
        self.ids = []

    def write(self, doc):
        self.ids.append(doc.doc_id)


def _run_interrupted(db_path):
    LocalExecutor([SlowPass()], str(db_path)).run(Records(8), Collect())


def test_checkpoint_survives_hard_interruption_and_resume(tmp_path):
    db = tmp_path / "checkpoint.db"
    process = multiprocessing.Process(target=_run_interrupted, args=(str(db),))
    process.start()
    time.sleep(0.35)
    process.terminate()
    process.join(10)
    assert process.exitcode != 0

    writer = Collect()
    stats = LocalExecutor([], str(db)).run(Records(8), writer)
    assert stats["processed"] + stats["skipped"] == 8
    assert len(writer.ids) == stats["processed"]
    assert len(writer.ids) == len(set(writer.ids))
