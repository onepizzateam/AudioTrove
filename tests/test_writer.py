import pytest
from audiotrove.document import AudioDocument
from audiotrove.io.writers import JSONLWriter
from pathlib import Path
import json
from audiotrove.utils.hashing import make_doc_id

def test_jsonl_writer_file_creation():
    output_path = "test_output.jsonl"
    writer = JSONLWriter(output_path)
    assert not Path(output_path).exists()

    doc = AudioDocument(path="test.mp3", duration=120, sample_rate=44100, metadata={"key": "value"})
    writer.write(doc)

    assert Path(output_path).exists()
    with open(output_path) as f:
        content = f.read().strip()
        assert len(content.splitlines()) == 1

def test_jsonl_writer_valid_json_output():
    output_path = "test_output.jsonl"
    writer = JSONLWriter(output_path)
    doc = AudioDocument(path="test.mp3", duration=120, sample_rate=44100, metadata={"key": "value"})
    writer.write(doc)

    with open(output_path) as f:
        content = json.load(f)
        assert content == {
            "doc_id": make_doc_id("test.mp3"),
            "source_path": "test.mp3",
            "sample_rate": 44100,
            "duration_seconds": 120,
            "metadata": {"key": "value"}
        }

def test_jsonl_writer_multiple_writes():
    output_path = "test_output.jsonl"
    writer = JSONLWriter(output_path)
    doc1 = AudioDocument(path="test1.mp3", duration=120, sample_rate=44100, metadata={"key": "value"})
    doc2 = AudioDocument(path="test2.mp3", duration=180, sample_rate=44100, metadata={"key": "value2"})

    writer.write(doc1)
    writer.write(doc2)

    with open(output_path) as f:
        content = f.read().strip()
        assert len(content.splitlines()) == 2

def test_jsonl_writer_metadata_preservation():
    output_path = "test_output.jsonl"
    writer = JSONLWriter(output_path)
    doc = AudioDocument(path="test.mp3", duration=120, sample_rate=44100, metadata={"key": "value"})

    writer.write(doc)

    with open(output_path) as f:
        content = json.load(f)
        assert content["metadata"] == {"key": "value"}
