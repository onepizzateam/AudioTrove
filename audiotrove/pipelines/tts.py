"""TTS-specific curation pipeline."""

from pathlib import Path

from audiotrove.executor.local import LocalExecutor
from audiotrove.exporters.tts_manifest import TTSManifestExporter
from audiotrove.filters.duration import DurationBucketFilter
from audiotrove.filters.snr import SNRFilter
from audiotrove.filters.vad import SileroVADFilter, VADSegmenter
from audiotrove.io.readers import LocalAudioReader
from audiotrove.transformers.silence_trim import SilenceTrimmingTransformer


def tts_pipeline(
    input_path: str,
    output_path: str,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
    snr_min: float = 20.0,
    padding_ms: int = 150,
    export_format: list[str] | None = None,
    extensions: list[str] | None = None,
    workers: int = 1,
    segment: bool = False,
    checkpoint_path: str | None = None,
    transcribe: bool = False,
    whisper_model: str = "base",
    device: str = "cpu",
) -> dict:

    """Curate audio files and export TTS-ready training manifests.

    Args:
        input_path: Audio file or directory of audio files.
        output_path: Directory for manifests and checkpoint data.
        min_duration: Minimum duration of curated clips in seconds.
        max_duration: Maximum duration of curated clips in seconds.
        snr_min: Minimum signal-to-noise ratio in dB.
        padding_ms: Silence retained on each side of speech.
        export_format: Requested manifest formats: ``ljspeech`` and/or ``f5tts``.
        extensions: Audio file extensions to read from a directory.
        workers: Number of local worker processes.
        segment: Split detected speech regions into separate clip documents.
        checkpoint_path: Optional SQLite checkpoint path for resumable runs.
        transcribe: Transcribe kept clips with Whisper.
        whisper_model: Whisper model size to use when transcribing.

    Returns:
        Summary with kept, filtered, total_duration_seconds, and output_files.
    """
    if export_format is None:
        export_format = ["ljspeech", "f5tts"]
    if extensions is None:
        extensions = ["wav"]

    input_file = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = (
        [str(input_file / f"*.{extension.lstrip('.')}") for extension in extensions]
        if input_file.is_dir()
        else [str(input_file)]
    )
    reader = LocalAudioReader(patterns, min_duration_seconds=0.0, max_duration_seconds=None)
    pipeline = [
        *([VADSegmenter(device=device)] if segment else []),
        SileroVADFilter(min_speech_ratio=0.1, device=device),
        SilenceTrimmingTransformer(padding_ms=padding_ms, device=device),
        SNRFilter(min_snr_db=snr_min, device=device),
        DurationBucketFilter(min_duration, max_duration),
        # TODO: add an optional speaker consistency filter when a lightweight backend is available.
    ]
    if transcribe:
        from audiotrove.transformers.whisper_transcribe import WhisperTranscriber

        pipeline.insert(3, WhisperTranscriber(model_name=whisper_model, device=device))
    executor = LocalExecutor(
        pipeline=pipeline,
        checkpoint_path=checkpoint_path or str(output_dir / "checkpoint.db"),
        num_workers=workers,
        device=device,
    )

    exporter = TTSManifestExporter(str(output_dir), export_format=export_format)
    stats = executor.run(reader, exporter)

    return {
        "kept": stats["kept"],
        "filtered": stats["skipped"],
        "total_duration_seconds": round(exporter.total_duration_seconds, 4),
        "output_files": exporter.output_files,
    }
