"""Lightweight deterministic audio fingerprint deduplication."""

import hashlib
import numpy as np


class FingerprintDeduplicator:
    """Reject near-identical audio using a compact spectral fingerprint."""

    name = "fingerprint_dedup"

    def __init__(self, threshold: int = 8):
        self.threshold = threshold
        self._fingerprints: list[tuple[int, str]] = []

    @staticmethod
    def _fingerprint(audio: np.ndarray) -> int:
        spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float32), n=2048))[1:]
        bands = np.array_split(spectrum, 64)
        mean = float(np.mean(spectrum))
        return sum((float(np.mean(band)) > mean) << i for i, band in enumerate(bands))

    def filter(self, doc) -> bool:
        fingerprint = self._fingerprint(doc.audio)
        for previous, doc_id in self._fingerprints:
            if (fingerprint ^ previous).bit_count() <= self.threshold:
                doc.metadata["duplicate_of"] = doc_id
                return False
        doc.metadata["audio_fingerprint"] = hashlib.sha256(
            fingerprint.to_bytes(8, "little")
        ).hexdigest()[:16]
        self._fingerprints.append((fingerprint, doc.doc_id))
        return True
