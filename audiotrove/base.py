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
    def transform(self, doc: AudioDocument) -> AudioDocument:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class AudioFanOutTransformer(ABC):
    """Transformer that can emit zero, one, or many documents from a single input.

    Unlike AudioTransformer (which always returns exactly one document), a fan-out
    transformer enables segmentation and expansion operations. For example, VAD
    segmentation splits a long file into multiple speech segments, each becoming
    its own AudioDocument for independent filtering and processing.

    Use cases:
    - Segmentation: split a file by detected speech regions (VADSegmenter)
    - Augmentation: expand a document into multiple variants
    - Filtering that returns variable-length output without discarding

    The returned documents should have deterministic doc_ids derived from the
    parent doc_id plus segment metadata (e.g., start/end times) so re-running
    the pipeline is idempotent and checkpoint-safe.
    """

    @abstractmethod
    def transform(self, doc: AudioDocument) -> list[AudioDocument]:
        """Transform one document into zero, one, or many documents.

        Args:
            doc: The input AudioDocument.

        Returns:
            A list of AudioDocument objects (may be empty).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
