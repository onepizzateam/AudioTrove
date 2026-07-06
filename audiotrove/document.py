from dataclasses import dataclass, field
import numpy as np


@dataclass
class AudioDocument:
    audio: np.ndarray
    sample_rate: int
    source_path: str
    duration_seconds: float
    doc_id: str
    metadata: dict = field(default_factory=dict)

    def __eq__(self, other):
        """Custom equality to handle numpy array comparison."""
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
