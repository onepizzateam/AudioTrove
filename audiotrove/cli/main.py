import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import logging

from audiotrove import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose (DEBUG) logging.")
def cli(verbose):
    """AudioTrove: Open-source audio data curation pipeline."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


@cli.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--vad-threshold",
    default=0.3,
    type=float,
    show_default=True,
    help="Minimum speech ratio (0-1) to keep a clip.",
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
    type=int,
    show_default=True,
    help="Number of worker processes for parallel execution.",
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
    help="Split audio into per-speech-segment sub-documents (VAD fan-out). Each speech segment becomes its own JSONL entry.",
)
@click.option(
    "--enhance",
    is_flag=True,
    default=False,
    help="Optional neural denoising via DeepFilterNet2 (requires pip install audiotrove[enhance]). Runs before VAD/SNR filtering.",
)
def curate(
    input_path,
    output_path,
    vad_threshold,
    snr_min,
    output_format,
    checkpoint,
    workers,
    extensions,
    segment,
    enhance,
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
        from audiotrove.filters.enhance import DeepFilterEnhancer

    # Validate paths
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.exists():
        console.print(f"[red]Error: Input path does not exist: {input_path}[/red]")
        raise SystemExit(1)

    output_p.mkdir(parents=True, exist_ok=True)

    # Build pipeline
    pipeline = []
    if enhance:
        pipeline.append(DeepFilterEnhancer())
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

    executor = LocalExecutor(pipeline=pipeline, checkpoint_path=checkpoint_db, num_workers=workers)

    # Run
    console.print("[cyan]Starting curation pipeline[/cyan]")
    console.print(f"  Input:  {input_path}")
    console.print(f"  Output: {output_manifest}")
    console.print(f"  Checkpoint: {checkpoint_db}")
    console.print(f"  Workers: {workers}")
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

    console.print(f"[green]✓ Manifest written to {output_manifest}[/green]")


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
