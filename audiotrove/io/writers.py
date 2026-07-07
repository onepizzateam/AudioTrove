"""
Audio writers.
"""

from pathlib import Path
import json

from audiotrove.document import AudioDocument

class JSONLWriter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, doc: AudioDocument) -> None:
        data = {
            "doc_id": doc.doc_id,
            "source_path": doc.source_path,
            "sample_rate": doc.sample_rate,
            "duration_seconds": doc.duration_seconds,
            "metadata": doc.metadata
        }
        with self.output_path.open('a') as f:
            json.dump(data, f)
            f.write('\n')
