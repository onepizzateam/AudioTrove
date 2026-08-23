import numpy as np

from audiotrove.document import AudioDocument
from audiotrove.filters.fingerprint import FingerprintDeduplicator
from audiotrove.qc import QCReport


def _doc(audio, name):
    return AudioDocument(audio, 16000, name, len(audio) / 16000, name)


def test_fingerprint_flags_duplicate_pair():
    audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    dedup = FingerprintDeduplicator()
    assert dedup.filter(_doc(audio, "one"))
    duplicate = _doc(audio.copy(), "two")
    assert not dedup.filter(duplicate)
    assert duplicate.metadata["duplicate_of"] == "one"


def test_qc_report_flags_clipped_and_silent(tmp_path):
    report = QCReport(str(tmp_path / "qc_report.json"))
    report.add(_doc(np.ones(1600, dtype=np.float32), "clipped"))
    report.add(_doc(np.zeros(1600, dtype=np.float32), "silent"))
    data = (tmp_path / "qc_report.json").read_text(encoding="utf-8")
    assert '"clipped": 1' in data
    assert '"silence_only": 1' in data
