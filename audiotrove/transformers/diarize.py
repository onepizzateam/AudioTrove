"""Speaker diarization fan-out transformer using pyannote.audio."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from audiotrove.base import AudioFanOutTransformer
from audiotrove.document import AudioDocument

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpeakerDiarizationTransformer(AudioFanOutTransformer):
    """Fan-out transformer that segments audio by speaker using pyannote.audio.

    Each detected speaker turn becomes its own ``AudioDocument`` whose audio
    contains only that turn's samples.  The child ``doc_id`` is deterministic
    (parent + speaker label + millisecond-precision timestamps) so pipeline
    re-runs are idempotent and checkpoint-safe.

    Requires: ``pip install audiotrove[diarize]``

    The pyannote/speaker-diarization-3.1 model requires accepting the licence
    at https://hf.co/pyannote/speaker-diarization-3.1 before use.
    """

    name = "speaker_diarization"

    def __init__(
        self,
        hf_token: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        """Initialise the diarization transformer.

        Args:
            hf_token: HuggingFace access token with access to the pyannote model.
            model_name: pyannote pipeline identifier on the HF hub.
            min_speakers: Optional lower bound on number of speakers.
            max_speakers: Optional upper bound on number of speakers.
        """
        try:
            import pyannote.audio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Speaker diarization requires the 'diarize' extra: "
                "pip install audiotrove[diarize]"
            ) from exc

        self.hf_token = hf_token
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    # ------------------------------------------------------------------
    # Pickle safety — exclude the loaded pyannote pipeline so that the
    # object can be transferred to worker processes via pickling; the
    # pipeline is re-loaded lazily in each worker on first use.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Exclude the lazy-loaded pyannote pipeline from pickling."""
        state = self.__dict__.copy()
        state["_pipeline"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling; pipeline will reload on first use."""
        self.__dict__.update(state)

    def __reduce__(self):
        return (
            self.__class__,
            (self.hf_token, self.model_name, self.min_speakers, self.max_speakers),
        )

    # ------------------------------------------------------------------
    # Lazy pipeline property
    # ------------------------------------------------------------------

    @property
    def pipeline(self):
        """Load the pyannote pipeline on first access (once per process)."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            logger.debug("Loading pyannote pipeline %s …", self.model_name)
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
        return self._pipeline

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def transform(self, doc: AudioDocument) -> list[AudioDocument]:
        """Diarize *doc* and return one child ``AudioDocument`` per speaker turn.

        The input audio is written to a temporary WAV file so that pyannote
        can read it without requiring in-memory format conversion.  The temp
        file is deleted after diarization.

        Args:
            doc: Input document (any duration, mono float32).

        Returns:
            List of ``AudioDocument`` objects — one per detected speaker turn.
            Returns ``[]`` when no speech is found.
        """
        sr = doc.sample_rate
        audio = doc.audio.astype(np.float32)

        # Write to a temp wav so pyannote can load it via its own reader.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, sr)

            # Build keyword args for optional speaker count bounds.
            run_kwargs: dict = {}
            if self.min_speakers is not None:
                run_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                run_kwargs["max_speakers"] = self.max_speakers

            diarization = self.pipeline(tmp_path, **run_kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        children: list[AudioDocument] = []
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)

            # Guard against edge cases where pyannote returns out-of-bounds times.
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(start_sample, min(end_sample, len(audio)))

            segment_audio = audio[start_sample:end_sample]
            duration = segment.end - segment.start

            # Deterministic doc_id: parent + speaker + ms-precision timestamps.
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            child_doc_id = f"{doc.doc_id}__{speaker_label}__{start_ms}_{end_ms}"

            child_meta = {
                **doc.metadata,
                "speaker_id": speaker_label,
                "parent_doc_id": doc.doc_id,
                "segment_start": segment.start,
                "segment_end": segment.end,
            }

            child = AudioDocument(
                audio=segment_audio,
                sample_rate=sr,
                source_path=doc.source_path,
                duration_seconds=duration,
                doc_id=child_doc_id,
                metadata=child_meta,
            )
            children.append(child)

        logger.debug(
            "SpeakerDiarizationTransformer: %d turn(s) from %s",
            len(children),
            doc.source_path,
        )
        return children
