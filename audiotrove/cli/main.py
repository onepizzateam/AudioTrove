import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import logging
import importlib.util
import os
import platform

from audiotrove import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging."
)
def cli(verbose):
    """AudioTrove: CPU-first audio curation and training pipeline."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


def _doctor_snapshot():
    """Collect lightweight local environment facts for the doctor command."""
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except (ImportError, AttributeError, OSError):
        available_ram = None
    optional_components = {
        "Rust extension": ("audiotrove_core",),
        "transcribe": ("whisper", "faster_whisper"),
        "train-piper": ("piper_train",),
        "infer": ("kokoro",),
        "enhance": ("deepfilternet",),
        "diarize": ("pyannote.audio",),
        "embed-dedup": ("faiss", "speechbrain"),
    }
    def module_available(module):
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    installed = {
        name: any(module_available(module) for module in modules)
        for name, modules in optional_components.items()
    }
    return {"python": platform.python_version(), "cpu_cores": os.cpu_count() or 1,
            "available_ram": available_ram, "components": installed}


def _doctor_recommendations(cpu_cores, available_ram):
    """Return conservative worker/RAM defaults from host resources."""
    if available_ram is None:
        return {"workers": max(1, min(cpu_cores, 4)), "ram_per_worker_gib": 1.0}
    ram_gib = available_ram / (1024 ** 3)
    workers = max(1, min(cpu_cores, int(ram_gib // 1.0), 8))
    return {"workers": workers, "ram_per_worker_gib": round(ram_gib / workers * 0.75, 2)}


@cli.command()
def doctor():
    """Report local CPU, memory, Rust, and optional-extra availability."""
    snapshot = _doctor_snapshot()
    table = Table(title="AudioTrove Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Python", snapshot["python"])
    table.add_row("CPU cores", str(snapshot["cpu_cores"]))
    ram = snapshot["available_ram"]
    table.add_row("Available RAM", f"{ram / (1024 ** 3):.2f} GiB" if ram else "unavailable")
    for name, present in snapshot["components"].items():
        table.add_row(name, "installed" if present else "not installed")
    recommendation = _doctor_recommendations(snapshot["cpu_cores"], snapshot["available_ram"])
    table.add_row("Recommended workers", str(recommendation["workers"]))
    table.add_row("Recommended RAM/worker", f"{recommendation['ram_per_worker_gib']:.2f} GiB")
    console.print(table)


@cli.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--vad-threshold",
    default=0.3,
    type=click.FloatRange(0.0, 1.0),
    show_default=True,
    help="Generic JSONL-path minimum speech ratio (0-1); ignored with --tts.",
)
@click.option(
    "--snr-min",
    default=15.0,
    type=float,
    show_default=True,
    help="Minimum SNR in dB to keep a clip.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["jsonl"]),
    default="jsonl",
    show_default=True,
    help="Output format.",
)
@click.option("--checkpoint", default=None, help="Path to checkpoint database for resumable runs.")
@click.option(
    "--workers",
    default=1,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of worker processes for parallel execution.",
)
@click.option(
    "--max-ram-per-worker",
    default=None,
    type=click.FloatRange(min=0.1),
    help="Soft RAM ceiling in GiB per worker (recorded for resource-aware runs).",
)
@click.option(
    "--extensions",
    default="wav",
    show_default=True,
    help='Comma-separated audio file extensions to process (e.g., "wav,mp3,flac").',
)
@click.option(
    "--segment",
    is_flag=True,
    default=False,
    help="Split audio into per-speech-segment documents (VAD fan-out); applies to JSONL and --tts modes.",
)
@click.option(
    "--enhance",
    is_flag=True,
    default=False,
    help="Generic JSONL-path DeepFilterNet2 enhancement; ignored with --tts (requires audiotrove[enhance]).",
)
@click.option(
    "--tts", is_flag=True, default=False, help="Run the TTS curation and manifest pipeline."
)
@click.option(
    "--tts-min-duration",
    default=2.0,
    type=float,
    show_default=True,
    help="Minimum TTS clip duration in seconds.",
)
@click.option(
    "--tts-max-duration",
    default=15.0,
    type=float,
    show_default=True,
    help="Maximum TTS clip duration in seconds.",
)
@click.option(
    "--tts-snr-min", default=20.0, type=float, show_default=True, help="Minimum TTS SNR in dB."
)
@click.option(
    "--tts-transcribe",
    is_flag=True,
    default=False,
    help="Transcribe kept clips with Whisper (requires pip install audiotrove[transcribe]).",
)
@click.option(
    "--tts-whisper-model",
    default="base",
    show_default=True,
    help="Whisper model size: tiny, base, small, medium, large.",
)
@click.option(
    "--tts-whisper-backend",
    type=click.Choice(["openai", "faster"]),
    default="faster",
    show_default=True,
    help="CPU transcription backend.",
)
@click.option(
    "--tts-diarize",
    is_flag=True,
    default=False,
    help="Run speaker diarization before VAD (requires pip install audiotrove[diarize]).",
)
@click.option(
    "--tts-hf-token",
    default=None,
    help="HuggingFace token for pyannote models (required when --tts-diarize is set).",
)
@click.option(
    "--tts-diarize-min-speakers",
    default=None,
    type=int,
    help="Minimum number of speakers for diarization.",
)
@click.option(
    "--tts-diarize-max-speakers",
    default=None,
    type=int,
    help="Maximum number of speakers for diarization.",
)
def curate(
    input_path,
    output_path,
    vad_threshold,
    snr_min,
    output_format,
    checkpoint,
    workers,
    max_ram_per_worker,
    extensions,
    segment,
    enhance,
    tts,
    tts_min_duration,
    tts_max_duration,
    tts_snr_min,
    tts_transcribe,
    tts_whisper_model,
    tts_whisper_backend,
    tts_diarize,
    tts_hf_token,
    tts_diarize_min_speakers,
    tts_diarize_max_speakers,
):
    """Curate audio files from INPUT_PATH into OUTPUT_PATH.

    Applies VAD and SNR filters, writes JSONL manifest.
    """
    from audiotrove.io.readers import LocalAudioReader
    from audiotrove.io.writers import JSONLWriter
    from audiotrove.filters.vad import SileroVADFilter
    from audiotrove.filters.snr import SNRFilter
    from audiotrove.executor.local import LocalExecutor

    if enhance:
        try:
            from audiotrove.filters.enhance import DeepFilterEnhancer
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc

    # Validate paths
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.exists():
        console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
        raise SystemExit(1)

    output_p.mkdir(parents=True, exist_ok=True)

    if tts:
        from audiotrove.pipelines.tts import tts_pipeline

        summary = tts_pipeline(
            input_path=input_path,
            output_path=output_path,
            min_duration=tts_min_duration,
            max_duration=tts_max_duration,
            snr_min=tts_snr_min,
            extensions=[extension.strip().lower() for extension in extensions.split(",")],
            workers=workers,
            segment=segment,
            checkpoint_path=checkpoint,
            transcribe=tts_transcribe,
            whisper_model=tts_whisper_model,
            whisper_backend=tts_whisper_backend,
            diarize=tts_diarize,
            hf_token=tts_hf_token,
            diarize_min_speakers=tts_diarize_min_speakers,
            diarize_max_speakers=tts_diarize_max_speakers,
        )
        console.print("[cyan]TTS curation pipeline complete[/cyan]")
        console.print(f"  Kept: {summary['kept']}")
        console.print(f"  Filtered: {summary['filtered']}")
        console.print(f"  Duration: {summary['total_duration_seconds']:.2f}s")
        for output_file in summary["output_files"]:
            console.print(f"  Output: {output_file}")
        console.print(f"  QC report: {summary['qc_report']}")
        return

    # Build pipeline
    pipeline = []
    if enhance:
        try:
            pipeline.append(DeepFilterEnhancer())
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc
    # Treat a vad_threshold of 0.0 as "no VAD filtering" (accept all audio).
    if vad_threshold > 0.0 and vad_threshold < 1.0:
        pipeline.append(SileroVADFilter(min_speech_ratio=vad_threshold))
    if snr_min > 0:
        pipeline.append(SNRFilter(min_snr_db=snr_min))

    if not pipeline:
        console.print(
            "[yellow]Warning: No filters in pipeline. All audio will pass through.[/yellow]"
        )

    # Parse extensions
    exts = [e.strip().lower() for e in extensions.split(",")]

    # Reader: build glob patterns for each extension
    if input_p.is_dir():
        patterns = [str(input_p / f"*.{ext}") for ext in exts]
    else:
        # If input is a file, just use that single file
        patterns = [str(input_p)]

    reader = LocalAudioReader(patterns)
    output_manifest = output_p / "manifest.jsonl"
    writer = JSONLWriter(str(output_manifest))

    # Executor
    checkpoint_db = checkpoint or str(output_p / "checkpoint.db")
    # Optionally insert segmentation
    if "segment" in locals() and segment:
        from audiotrove.filters.vad import VADSegmenter

        # Insert segmenter after VAD filter (if present), otherwise at start
        vad_idx = next(
            (i for i, b in enumerate(pipeline) if getattr(b, "name", "") == "silero_vad"), None
        )
        insert_at = (vad_idx + 1) if vad_idx is not None else 0
        pipeline.insert(insert_at, VADSegmenter(threshold=0.5))
        console.print("  [cyan]Segmentation: enabled (VADSegmenter)[/cyan]")

    executor = LocalExecutor(
        pipeline=pipeline,
        checkpoint_path=checkpoint_db,
        num_workers=workers,
        max_ram_per_worker=max_ram_per_worker,
    )

    # Run
    console.print("[cyan]Starting curation pipeline[/cyan]")
    console.print(f"  Input:  {input_path}")
    console.print(f"  Output: {output_manifest}")
    console.print(f"  Checkpoint: {checkpoint_db}")
    console.print(f"  Workers: {workers}")
    if max_ram_per_worker is not None:
        console.print(f"  RAM ceiling: {max_ram_per_worker:.2f} GiB/worker")
    console.print(f"  Extensions: {', '.join(exts)}")
    console.print(f"  Filters: {', '.join(b.name for b in pipeline) or 'none'}")
    if enhance:
        console.print("Enhancement enabled (DeepFilterNet2). First run downloads ~60MB model.")
    console.print(
        f"  Segmentation: {'enabled' if 'segment' in locals() and segment else 'disabled'}"
    )
    console.print()

    stats = executor.run(reader, writer)

    # Print results
    table = Table(title="Curation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Processed", str(stats["processed"]))
    table.add_row("Kept", str(stats["kept"]))
    table.add_row("Filtered", str(stats["skipped"]))
    console.print(table)

    console.print(f"[green]Manifest written to {output_manifest}[/green]")


@cli.command()
@click.argument("input_path")
@click.option(
    "--extensions",
    default="wav",
    show_default=True,
    help='Comma-separated audio extensions to inspect (e.g. "wav,mp3,flac").',
)
@click.option(
    "--limit",
    default=None,
    type=int,
    show_default=True,
    help="Show stats for first N files. Cap the number of files inspected (default: all).",
)
def inspect(input_path, extensions, limit):
    """Show statistics for an audio directory without filtering."""
    from audiotrove.io.readers import LocalAudioReader
    import numpy as np

    input_p = Path(input_path)
    if not input_p.exists():
        console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
        raise SystemExit(1)

    exts = [e.strip().lower() for e in extensions.split(",")]

    if input_p.is_dir():
        patterns = [str(input_p / f"**/*.{ext}") for ext in exts]
    else:
        patterns = [str(input_p)]

    reader = LocalAudioReader(patterns, min_duration_seconds=0.0, max_duration_seconds=None)

    durations = []
    format_counts = {}
    count = 0

    for doc in reader:
        if doc is None:
            continue
        durations.append(doc.duration_seconds)
        ext = Path(doc.source_path).suffix.lstrip(".").lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
        count += 1
        if limit and count >= limit:
            break

    if not durations:
        console.print("[yellow]No audio files found.[/yellow]")
        return

    durations = np.array(durations)
    total_hours = durations.sum() / 3600

    table = Table(title=f"Audio Directory Stats ({len(durations)} files)")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Files", str(len(durations)))
    table.add_row("Total duration", f"{total_hours:.2f}h ({durations.sum():.0f}s)")
    table.add_row("Mean duration", f"{durations.mean():.2f}s")
    table.add_row("Median duration", f"{np.median(durations):.2f}s")
    table.add_row("Min duration", f"{durations.min():.2f}s")
    table.add_row("Max duration", f"{durations.max():.2f}s")
    for fmt, cnt in sorted(format_counts.items()):
        table.add_row(f"Format: .{fmt}", str(cnt))
    console.print(table)


if __name__ == "__main__":
    cli()
