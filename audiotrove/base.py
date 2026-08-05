from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from audiotrove.document import AudioDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


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


class GPUFilter(AudioFilter):
    """AudioFilter that receives a device and can operate on GPU tensors.

    Subclasses run on a :class:`torch.device`. The default :meth:`to`
    implementation is a no-op so simple GPU filters that keep no persistent
    model state still satisfy the interface.
    """

    @property
    @abstractmethod
    def device(self) -> "torch.device":
        """Return the device this filter currently operates on."""

    def to(self, device: "torch.device") -> "GPUFilter":
        """Move internal models to ``device``. Returns ``self``."""
        return self


class GPUTransformer(AudioTransformer):
    """AudioTransformer that operates on GPU tensors."""

    @property
    @abstractmethod
    def device(self) -> "torch.device":
        """Return the device this transformer currently operates on."""

    def to(self, device: "torch.device") -> "GPUTransformer":
        """Move internal models to ``device``. Returns ``self``."""
        return self


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
