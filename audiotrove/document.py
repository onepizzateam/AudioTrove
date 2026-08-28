from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AudioDocument:
    audio: np.ndarray
    sample_rate: int
    source_path: str
    duration_seconds: float
    doc_id: str
    metadata: dict = field(default_factory=dict)
    # Optional cache of the audio already resident on a GPU device. This is set
    # by GPUFilter/GPUTransformer components to avoid redundant host<->device
    # transfers across adjacent GPU steps. The CPU ``audio`` field always
    # remains the source of truth; ``gpu_tensor`` is only a cache and is never
    # pickled (workers reload/recompute it on demand).
    gpu_tensor: Optional["object"] = field(default=None, repr=False, compare=False)

    def __getstate__(self):
        """Exclude the GPU tensor cache from pickling.

        The tensor lives on a device that may not exist in a worker process,
        and it is always reproducible from the CPU ``audio`` array, so we drop
        it before serialisation.
        """
        state = self.__dict__.copy()
        state["gpu_tensor"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling; gpu_tensor stays cleared."""
        self.__dict__.update(state)
        self.gpu_tensor = None

    def __eq__(self, other):
        """Custom equality to handle numpy array comparison.

        ``gpu_tensor`` is intentionally excluded — it is a device-local cache,
        not part of the document's identity.
        """
        if not isinstance(other, AudioDocument):
            return False
        return (
            np.array_equal(self.audio, other.audio)
            and self.sample_rate == other.sample_rate
            and self.source_path == other.source_path
            and self.duration_seconds == other.duration_seconds
            and self.doc_id == other.doc_id
            and self.metadata == other.metadata
        )
