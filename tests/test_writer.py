import json
import numpy as np
from pathlib import Path

from audiotrove.document import AudioDocument
from audiotrove.io.writers import JSONLWriter
from audiotrove.utils.hashing import make_doc_id


def _make_audio(duration_s=1.0, sr=16000):
    import numpy as _np
    return (_np.zeros(int(duration_s * sr), dtype=_np.float32), sr)


def test_jsonl_writer_file_creation(tmp_path):
    output_path = tmp_path / "test_output.jsonl"
    writer = JSONLWriter(str(output_path))
    assert not output_path.exists()

    audio, sr = _make_audio()
    doc = AudioDocument(audio=audio, sample_rate=sr, source_path="test.mp3", duration_seconds=1.0, doc_id=make_doc_id("test.mp3"), metadata={"key": "value"})
    writer.write(doc)

    assert output_path.exists()
    with open(output_path) as f:
        content = f.read().strip()
        assert len(content.splitlines()) == 1


def test_jsonl_writer_valid_json_output(tmp_path):
    output_path = tmp_path / "test_output.jsonl"
    writer = JSONLWriter(str(output_path))
    audio, sr = _make_audio()
    doc = AudioDocument(audio=audio, sample_rate=sr, source_path="test.mp3", duration_seconds=1.0, doc_id=make_doc_id("test.mp3"), metadata={"key": "value"})
    writer.write(doc)

    with open(output_path) as f:
        content = json.loads(f.read())
        assert content == {
            "doc_id": make_doc_id("test.mp3"),
            "source_path": "test.mp3",
            "sample_rate": sr,
            "duration_seconds": 1.0,
            "metadata": {"key": "value"}
        }


def test_jsonl_writer_multiple_writes(tmp_path):
    output_path = tmp_path / "test_output.jsonl"
    writer = JSONLWriter(str(output_path))
    audio, sr = _make_audio()
    doc1 = AudioDocument(audio=audio, sample_rate=sr, source_path="test1.mp3", duration_seconds=1.0, doc_id=make_doc_id("test1.mp3"), metadata={"key": "value"})
    doc2 = AudioDocument(audio=audio, sample_rate=sr, source_path="test2.mp3", duration_seconds=1.0, doc_id=make_doc_id("test2.mp3"), metadata={"key": "value2"})

    writer.write(doc1)
    writer.write(doc2)

    with open(output_path) as f:
        content = f.read().strip()
        assert len(content.splitlines()) == 2


def test_jsonl_writer_metadata_preservation(tmp_path):
    output_path = tmp_path / "test_output.jsonl"
    writer = JSONLWriter(str(output_path))
    audio, sr = _make_audio()
    doc = AudioDocument(audio=audio, sample_rate=sr, source_path="test.mp3", duration_seconds=1.0, doc_id=make_doc_id("test.mp3"), metadata={"key": "value"})

    writer.write(doc)

    with open(output_path) as f:
        content = json.loads(f.read())
        assert content["metadata"] == {"key": "value"}
