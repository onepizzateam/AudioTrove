from abc import ABC, abstractmethod
from audiotrove.document import AudioDocument


class AudioFilter(ABC):

    @abstractmethod
    def filter(self, doc: AudioDocument) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class AudioTransformer(ABC):

    @abstractmethod
    def transform(
        self,
        doc: AudioDocument
    ) -> AudioDocument:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
