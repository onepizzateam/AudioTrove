import pytest
from abc import ABC, abstractmethod
from audiotrove.base import AudioFilter, AudioTransformer

class MockAudioDocument:
    def __init__(self):
        pass

def test_audio_filter_interface():
    import pytest

    class MyFilter(AudioFilter):
        pass

    with pytest.raises(TypeError):
        MyFilter()

    class ConcreteFilter(AudioFilter):
        def filter(self, doc: MockAudioDocument) -> bool:
            return True

        @property
        def name(self) -> str:
            return "concrete"

    assert issubclass(ConcreteFilter, AudioFilter)
    assert hasattr(ConcreteFilter, 'filter')
    assert hasattr(ConcreteFilter, 'name')

def test_audio_transformer_interface():
    import pytest

    class MyTransformer(AudioTransformer):
        pass

    with pytest.raises(TypeError):
        MyTransformer()
    class ConcreteTransformer(AudioTransformer):
        def transform(self, doc: MockAudioDocument) -> MockAudioDocument:
            return doc

        @property
        def name(self) -> str:
            return "concrete"

    assert issubclass(ConcreteTransformer, AudioTransformer)
    assert hasattr(ConcreteTransformer, 'transform')
    assert hasattr(ConcreteTransformer, 'name')
