import pytest
from abc import ABC, abstractmethod
from audiotrove.base import AudioFilter, AudioTransformer

class MockAudioDocument:
    def __init__(self):
        pass

def test_audio_filter_interface():
    with pytest.raises(TypeError) as exc_info:
        class MyFilter(AudioFilter):
            pass

    assert "Can't instantiate abstract class MyFilter with abstract methods filter, name" in str(exc_info.value)

    class ConcreteFilter(AudioFilter):
        @abstractmethod
        def filter(self, doc: MockAudioDocument) -> bool:
            pass

        @property
        @abstractmethod
        def name(self) -> str:
            pass

    assert issubclass(ConcreteFilter, AudioFilter)
    assert hasattr(ConcreteFilter, 'filter')
    assert hasattr(ConcreteFilter, 'name')

def test_audio_transformer_interface():
    with pytest.raises(TypeError) as exc_info:
        class MyTransformer(AudioTransformer):
            pass

    assert "Can't instantiate abstract class MyTransformer with abstract methods transform, name" in str(exc_info.value)

    class ConcreteTransformer(AudioTransformer):
        @abstractmethod
        def transform(self, doc: MockAudioDocument) -> MockAudioDocument:
            pass

        @property
        @abstractmethod
        def name(self) -> str:
            pass

    assert issubclass(ConcreteTransformer, AudioTransformer)
    assert hasattr(ConcreteTransformer, 'transform')
    assert hasattr(ConcreteTransformer, 'name')
