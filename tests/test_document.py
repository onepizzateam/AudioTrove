import pytest
from audiotrove.document import AudioDocument

def test_audio_document_creation():
    doc = AudioDocument(path="test.mp3", duration=120)
    assert doc.path == "test.mp3"
    assert doc.duration == 120

def test_audio_document_str_representation():
    doc = AudioDocument(path="test.mp3", duration=120)
    assert str(doc) == "AudioDocument(path='test.mp3', duration=120)"

def test_audio_document_eq_method():
    doc1 = AudioDocument(path="test.mp3", duration=120)
    doc2 = AudioDocument(path="test.mp3", duration=120)
    assert doc1 == doc2

    doc3 = AudioDocument(path="test2.mp3", duration=120)
    assert doc1 != doc3
