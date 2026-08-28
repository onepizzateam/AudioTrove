"""Streaming quality-control summary for exported clips."""

import json
from pathlib import Path


class QCReport:
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.clip_count = 0
        self.duration_min = None
        self.duration_max = None
        self.duration_total = 0.0
        self.clipped = 0
        self.silent = 0
        self.snr_values = []

    def add(self, doc):
        duration = float(doc.duration_seconds)
        self.clip_count += 1
        self.duration_total += duration
        self.duration_min = duration if self.duration_min is None else min(self.duration_min, duration)
        self.duration_max = duration if self.duration_max is None else max(self.duration_max, duration)
        if abs(doc.audio).max(initial=0) >= 0.999:
            self.clipped += 1
        if (abs(doc.audio) > 1e-4).mean() < 0.01:
            self.silent += 1
        if "snr_db" in doc.metadata:
            self.snr_values.append(float(doc.metadata["snr_db"]))
        self.write()

    def write(self):
        data = {"clips": self.clip_count, "clipped": self.clipped,
                "silence_only": self.silent,
                "duration_seconds": {"min": self.duration_min, "max": self.duration_max,
                                     "mean": self.duration_total / self.clip_count
                                     if self.clip_count else None},
                "snr_db": self.snr_values}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
