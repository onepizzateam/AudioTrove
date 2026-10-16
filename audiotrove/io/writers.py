"""
Audio writers.
"""

import datetime
import json
from collections.abc import Iterable
from pathlib import Path

from audiotrove import __version__
from audiotrove.document import AudioDocument


class JSONLWriter:
    def __init__(
        self,
        output_path: str,
        required_metadata: str | Iterable[str] | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(required_metadata, str):
            self.required_metadata = [k.strip() for k in required_metadata.split(",") if k.strip()]
        elif required_metadata:
            self.required_metadata = list(required_metadata)
        else:
            self.required_metadata = []

    def write(self, doc: AudioDocument) -> None:
        if self.required_metadata:
            missing = [k for k in self.required_metadata if k not in doc.metadata]
            if missing:
                raise ValueError(
                    f"Document '{doc.doc_id}' is missing required metadata field(s): {', '.join(missing)}"
                )

        data = {
            "doc_id": doc.doc_id,
            "source_path": doc.source_path,
            "sample_rate": doc.sample_rate,
            "duration_seconds": round(doc.duration_seconds, 4),
            "metadata": doc.metadata,
            "pipeline_version": __version__,
            "processed_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            json.dump(data, f)
            f.write("\n")
