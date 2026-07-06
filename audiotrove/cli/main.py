import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli():
    """AudioTrove: Open-source audio data curation pipeline."""
    pass


@cli.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--vad-threshold', default=0.3, show_default=True,
              help='Minimum speech ratio (0-1) to keep a clip.')
@click.option('--snr-min', default=15.0, show_default=True,
              help='Minimum SNR in dB to keep a clip.')
@click.option('--workers', default=1, show_default=True)
@click.option('--format', 'output_format', 
              type=click.Choice(['jsonl']),
              default='jsonl', show_default=True)
@click.option('--checkpoint', default=None,
              help='Path to checkpoint database for resumable runs.')
def curate(input_path, output_path, vad_threshold, snr_min, workers, output_format, checkpoint):
    """Curate audio files from INPUT_PATH into OUTPUT_PATH."""
    console.print("Curate not yet fully implemented in this local build.")


@cli.command()
@click.argument('input_path')
def inspect(input_path):
    """Show statistics for an audio directory without filtering."""
    console.print(f"Inspect not implemented: {input_path}")


if __name__ == '__main__':
    cli()
