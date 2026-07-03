"""
Audio writers.
"""

import os
from pathlib import Path
from typing import Dict

from audiotrove.document import AudioDocument
from audiotrove.utils.hashing import make_doc_id

class JSONLWriter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, doc: AudioDocument) -> None:
        data = {
            "doc_id": make_doc_id(doc.path),
            "source_path": doc.path,
            "sample_rate": doc.sample_rate,
            "duration_seconds": doc.duration,
            "metadata": doc.metadata
        }
        with self.output_path.open('a') as f:
            json.dump(data, f)
            f.write('\n')
