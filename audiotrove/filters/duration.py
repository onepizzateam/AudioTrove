"""Duration filtering."""

from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument


class DurationBucketFilter(AudioFilter):
    """Keep audio documents whose duration falls within a configured range."""

    name = "duration_bucket"

    def __init__(self, min_duration_seconds: float = 2.0, max_duration_seconds: float = 15.0):
        """Initialize the duration bucket filter.

        Args:
            min_duration_seconds: Minimum clip duration to keep.
            max_duration_seconds: Maximum clip duration to keep.
        """
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds

    def filter(self, doc: AudioDocument) -> bool:
        """Keep documents with durations inside the configured interval."""
        if doc.duration_seconds < self.min_duration_seconds:
            doc.metadata["duration_filter_reason"] = "too_short"
            return False
        if doc.duration_seconds > self.max_duration_seconds:
            doc.metadata["duration_filter_reason"] = "too_long"
            return False
        return True
