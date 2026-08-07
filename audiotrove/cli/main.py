import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import logging

from audiotrove import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging."
)
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
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="cpu",
    show_default=True,
    help="Compute device for GPU-aware filters/transformers.",
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
    extensions,
    segment,
    enhance,
    tts,
    tts_min_duration,
    tts_max_duration,
    tts_snr_min,
    tts_transcribe,
    tts_whisper_model,
    device,
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
            device=device,
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


@cli.command()
@click.argument("manifest_path")
@click.argument("output_path")
@click.option(
    "--framework",
    type=click.Choice(["f5tts", "styletts2", "piper", "matcha"]),
    default="f5tts",
    show_default=True,
    help="Training framework.",
)
@click.option("--epochs", default=100, type=int, show_default=True)
@click.option("--batch-size", default=16, type=int, show_default=True)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--num-gpus", default=1, type=int, show_default=True)
@click.option("--resume-from", default=None, help="Checkpoint path to resume training from.")
def train(manifest_path, output_path, framework, epochs, batch_size, device, num_gpus, resume_from):
    """Fine-tune a TTS model from MANIFEST_PATH, writing to OUTPUT_PATH."""
    from audiotrove.training import get_trainer
    from audiotrove.training.base import TrainingConfig

    config = TrainingConfig(
        manifest_path=manifest_path,
        output_dir=output_path,
        model_name=framework,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        num_gpus=num_gpus,
        resume_from=resume_from,
    )
    trainer = get_trainer(framework, config)
    trainer.validate_manifest()
    console.print(f"[cyan]Training {framework} ({epochs} epochs, device={device})[/cyan]")
    try:
        metrics = trainer.train()
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    final = trainer.export(str(Path(output_path) / "final.pt"))
    console.print(f"[green]Training complete. Model: {final}[/green]")
    console.print(f"  Metrics: {metrics}")


@cli.command()
@click.option(
    "--task",
    type=click.Choice(["tts", "asr", "vc"]),
    required=True,
    help="Inference task type.",
)
@click.option("--family", default=None, help="Model family (e.g. f5tts, faster_whisper, seed_vc).")
@click.option("--model", "model_path", default=None, help="Local model path.")
@click.option("--text", default=None, help="TTS input text.")
@click.option("--audio", "audio_path", default=None, help="ASR / VC source audio path.")
@click.option("--voice-ref", default=None, help="TTS / VC reference voice path.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--out", "out_path", default=None, help="Output WAV (TTS/VC).")
def infer(task, family, model_path, text, audio_path, voice_ref, device, out_path):
    """Run a single inference request (TTS, ASR, or VC)."""
    try:
        if task == "tts":
            from audiotrove.inference.tts import get_tts_session

            session = get_tts_session(
                family or "f5tts",
                model_path=model_path,
                device=device,
                voice_ref=voice_ref,
            )
            with session:
                result = session.run(text=text, voice_ref=voice_ref)
            _write_or_print_audio(result, out_path)
        elif task == "asr":
            from audiotrove.inference.asr import get_asr_session

            session = get_asr_session(
                family or "faster_whisper",
                model_path=model_path or "base",
                device=device,
            )
            with session:
                result = session.run(audio_path=audio_path)
            console.print(result.text or "")
        else:  # vc
            from audiotrove.inference.vc import get_vc_session

            session = get_vc_session(family or "seed_vc", model_path=model_path, device=device)
            with session:
                result = session.run(
                    source_audio_path=audio_path, target_voice_path=voice_ref
                )
            _write_or_print_audio(result, out_path)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


def _write_or_print_audio(result, out_path):
    """Write TTS/VC audio to ``out_path`` or report that no path was given."""
    if out_path and result.audio is not None:
        import soundfile as sf

        sf.write(out_path, result.audio, result.sample_rate)
        console.print(f"[green]Wrote {out_path}[/green]")
    else:
        console.print("[yellow]No --out path given; audio not written.[/yellow]")


@cli.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--min-duration", default=2.0, type=float, show_default=True)
@click.option("--max-duration", default=15.0, type=float, show_default=True)
@click.option("--snr-min", default=20.0, type=float, show_default=True)
@click.option("--extensions", default="wav", show_default=True)
@click.option("--workers", default=1, type=click.IntRange(min=1), show_default=True)
@click.option("--transcribe/--no-transcribe", default=True, show_default=True)
@click.option("--whisper-model", default="base", show_default=True)
@click.option("--train/--no-train", "do_train", default=True, show_default=True)
@click.option("--framework", default="f5tts", show_default=True)
@click.option("--epochs", default=100, type=int, show_default=True)
@click.option("--batch-size", default=16, type=int, show_default=True)
@click.option("--num-gpus", default=1, type=int, show_default=True)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--validate/--no-validate", "do_validate", default=False, show_default=True)
@click.option("--validate-text", default="Hello, this is a test of the trained voice.")
@click.option("--validate-voice-ref", default=None)
def run(
    input_path,
    output_path,
    min_duration,
    max_duration,
    snr_min,
    extensions,
    workers,
    transcribe,
    whisper_model,
    do_train,
    framework,
    epochs,
    batch_size,
    num_gpus,
    device,
    do_validate,
    validate_text,
    validate_voice_ref,
):
    """End-to-end: raw audio -> curated -> trained voice model."""
    from audiotrove.pipelines.e2e import E2EConfig, e2e_pipeline

    config = E2EConfig(
        input_path=input_path,
        output_path=output_path,
        min_duration=min_duration,
        max_duration=max_duration,
        snr_min=snr_min,
        extensions=[e.strip().lower() for e in extensions.split(",")],
        workers=workers,
        device=device,
        transcribe=transcribe,
        whisper_model=whisper_model,
        train=do_train,
        train_framework=framework,
        epochs=epochs,
        batch_size=batch_size,
        num_gpus=num_gpus,
        validate_inference=do_validate,
        validate_text=validate_text,
        validate_voice_ref=validate_voice_ref,
    )
    console.print("[bold cyan]AudioTrove end-to-end pipeline[/bold cyan]")
    try:
        result = e2e_pipeline(config)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc

    curate = result["curate_summary"]
    console.print(f"  [1/3] Curated: {curate['kept']} clips kept, {curate['filtered']} filtered")
    if result["train_summary"] is not None:
        console.print("  [2/3] Training complete")
    if result["validation_audio_path"]:
        console.print(f"  [3/3] Validation: {result['validation_audio_path']}")
    console.print("[green]Done.[/green]")


@cli.command()
@click.argument("config_path")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, type=int, show_default=True)
def serve(config_path, host, port):
    """Start the AudioTrove inference HTTP server from CONFIG_PATH.

    The server exposes inference endpoints without authentication. Run it only
    on a trusted network or behind an authenticating reverse proxy.
    """
    from audiotrove.inference.server import AudioTroveServer

    server = AudioTroveServer(config_path)
    # CLI flags override config values when provided.
    server.host = host
    server.port = port
    console.print(
        f"[yellow]Serving without authentication on {host}:{port}. "
        "Do not expose to untrusted networks.[/yellow]"
    )
    try:
        server.run()
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
