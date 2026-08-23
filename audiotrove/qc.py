"""Streaming quality-control summary for exported clips."""

import json
from pathlib import Path


class QCReport:
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.durations = []
        self.clipped = 0
        self.silent = 0
        self.snr_values = []

    def add(self, doc):
        self.durations.append(doc.duration_seconds)
        if abs(doc.audio).max(initial=0) >= 0.999:
            self.clipped += 1
        if (abs(doc.audio) > 1e-4).mean() < 0.01:
            self.silent += 1
        if "snr_db" in doc.metadata:
            self.snr_values.append(float(doc.metadata["snr_db"]))
        self.write()

    def write(self):
        data = {"clips": len(self.durations), "clipped": self.clipped,
                "silence_only": self.silent, "durations_seconds": self.durations,
                "snr_db": self.snr_values}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
