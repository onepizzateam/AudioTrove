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
