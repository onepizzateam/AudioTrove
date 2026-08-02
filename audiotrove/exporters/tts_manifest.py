"""TTS training manifest export."""

import logging
from pathlib import Path

import soundfile as sf

from audiotrove.document import AudioDocument

logger = logging.getLogger(__name__)


class TTSManifestExporter:
    """Write curated documents as LJSpeech and F5-TTS manifests."""

    def __init__(
        self,
        output_dir: str,
        speaker_id: str = "speaker_00",
        export_format: list[str] | None = None,
    ):
        """Initialize the TTS manifest exporter.

        Args:
            output_dir: Directory where manifest files are written.
            speaker_id: Speaker identifier for LJSpeech metadata.
            export_format: Requested formats: ``ljspeech`` and/or ``f5tts``.
        """
        self.output_dir = Path(output_dir)
        self.speaker_id = speaker_id
        self.export_format = export_format or ["ljspeech", "f5tts"]
        invalid_formats = set(self.export_format) - {"ljspeech", "f5tts"}
        if invalid_formats:
            raise ValueError(f"Unsupported TTS export formats: {sorted(invalid_formats)}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.csv"
        self.filelist_path = self.output_dir / "filelist.txt"
        self.total_duration_seconds = 0.0
        self._initialize_files()

    @property
    def output_files(self) -> list[str]:
        """Return paths to the requested manifest files."""
        files = []
        if "ljspeech" in self.export_format:
            files.append(str(self.metadata_path))
        if "f5tts" in self.export_format:
            files.append(str(self.filelist_path))
        return files

    def _initialize_files(self) -> None:
        """Create requested manifest files without discarding resumed output."""
        for path in self.output_files:
            Path(path).touch(exist_ok=True)

    def write(self, doc: AudioDocument) -> None:
        """Append one audio document to each requested manifest."""
        transcription = str(doc.metadata.get("transcription", ""))
        # Per-clip speaker_id takes precedence over the exporter-level default.
        # This lets diarized clips carry their own speaker label automatically.
        effective_speaker_id = doc.metadata.get("speaker_id", self.speaker_id)
        self.total_duration_seconds += doc.duration_seconds
        stem = Path(doc.source_path).stem
        if "parent_doc_id" in doc.metadata:
            stem = f"{stem}_{doc.doc_id}"
        audio_path = self.output_dir / f"{stem}.wav"
        if audio_path.exists():
            logger.warning(
                "Stem collision: %s already exists and will be overwritten. "
                "Rename source files with duplicate stems to avoid data loss.",
                audio_path,
            )
        sf.write(audio_path, doc.audio, doc.sample_rate)
        if "ljspeech" in self.export_format:
            filename = audio_path.name
            with self.metadata_path.open("a", encoding="utf-8", newline="") as metadata_file:
                metadata_file.write(f"{filename}|{effective_speaker_id}|{transcription}\n")
        if "f5tts" in self.export_format:
            with self.filelist_path.open("a", encoding="utf-8", newline="") as filelist_file:
                filelist_file.write(f"{audio_path}\t{doc.duration_seconds:.4f}\t{transcription}\n")

    def export(self, documents: list[AudioDocument]) -> list[str]:
        """Write a list of audio documents and return the generated file paths."""
        for doc in documents:
            self.write(doc)
        return self.output_files
