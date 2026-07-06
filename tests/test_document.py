import numpy as np
from audiotrove.document import AudioDocument


def test_audio_document_creation():
    audio = np.zeros(16000, dtype=np.float32)
    doc = AudioDocument(audio=audio, sample_rate=16000, source_path='test.mp3', duration_seconds=1.0, doc_id='abc')
    assert doc.source_path == 'test.mp3'
    assert doc.duration_seconds == 1.0


def test_audio_document_eq_and_repr():
    audio = np.zeros(16000, dtype=np.float32)
    doc1 = AudioDocument(audio=audio, sample_rate=16000, source_path='a.wav', duration_seconds=1.0, doc_id='a')
    doc2 = AudioDocument(audio=audio, sample_rate=16000, source_path='a.wav', duration_seconds=1.0, doc_id='a')
    assert doc1 == doc2
    doc3 = AudioDocument(audio=audio, sample_rate=16000, source_path='b.wav', duration_seconds=1.0, doc_id='b')
    assert doc1 != doc3
