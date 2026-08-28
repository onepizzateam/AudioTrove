import numpy as np
import json

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


def test_tts_curation_emits_qc_report_for_clipped_and_silent(tmp_path, monkeypatch):
    from audiotrove.document import AudioDocument
    from audiotrove.pipelines import tts as tts_module

    docs = [
        AudioDocument(np.ones(32000, dtype=np.float32), 16000, "clipped.wav", 2.0, "clipped",
                      metadata={"snr_db": 20.0}),
        AudioDocument(np.zeros(32000, dtype=np.float32), 16000, "silent.wav", 2.0, "silent",
                      metadata={"snr_db": 20.0}),
    ]

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            return iter(docs)

    class PassBlock:
        name = "pass"

        def __init__(self, *args, **kwargs):
            pass

        def filter(self, doc):
            doc.metadata["snr_db"] = 20.0
            return True

        def transform(self, doc):
            return doc

    class Executor:
        def __init__(self, pipeline, checkpoint_path, num_workers):
            self.pipeline = pipeline

        def run(self, reader, writer):
            for doc in reader:
                writer.write(doc)
            return {"kept": 2, "skipped": 0}

    monkeypatch.setattr(tts_module, "LocalAudioReader", Reader)
    monkeypatch.setattr(tts_module, "SileroVADFilter", PassBlock)
    monkeypatch.setattr(tts_module, "SilenceTrimmingTransformer", PassBlock)
    monkeypatch.setattr(tts_module, "SNRFilter", PassBlock)
    monkeypatch.setattr(tts_module, "DurationBucketFilter", PassBlock)
    monkeypatch.setattr(tts_module, "LocalExecutor", Executor)

    summary = tts_module.tts_pipeline("input", str(tmp_path))
    data = json.loads((tmp_path / "qc_report.json").read_text(encoding="utf-8"))
    assert summary["qc_report"] == str(tmp_path / "qc_report.json")
    assert data["clipped"] == 1
    assert data["silence_only"] == 1
    assert data["snr_db"] == [20.0, 20.0]
