import json
import subprocess
from pathlib import Path


def run_help(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = p.stdout or p.stderr
    return out


def main():
    base = Path(__file__).resolve().parents[1]
    bench = base / "benchmarks" / "e2e_results.json"
    demo = base / "packaging" / "curate_demo_output_clean.txt"

    data = json.loads(bench.read_text(encoding="utf8"))

    seq = data["runs"]["sequential"]
    par4 = data["runs"]["parallel_4"]
    seg = data["runs"]["segmentation"]

    demo_text = demo.read_text(encoding="utf8")

    help_curate = run_help(f"{base}/.venv/Scripts/python.exe -m audiotrove.cli.main curate --help")
    help_inspect = run_help(f"{base}/.venv/Scripts/python.exe -m audiotrove.cli.main inspect --help")

    readme = f"""# AudioTrove

Composable audio dataset curation: VAD, SNR filtering, deterministic segmentation, resumable checkpointing, and optional process-parallel execution.

## Demo

Captured output (cleaned):

```
{demo_text.strip()}
```

## Performance

Benchmarked on LibriSpeech dev-clean WAV fixtures ({data['fixture_count']} files) on {data['system']['platform']} with Python {data['system']['python']}.

| Mode | Files | Wall time | Throughput | Real-time factor |
|------|------:|----------:|-----------:|-----------------:|
| Sequential (--workers 1) | {seq['files_processed']} | {seq['elapsed_s']:.2f}s | {seq['throughput_files_per_sec']:.2f} files/sec | {seq['real_time_factor']:.2f}x |
| Parallel (--workers 4) | {par4['files_processed']} | {par4['elapsed_s']:.2f}s | {par4['throughput_files_per_sec']:.2f} files/sec | {par4['real_time_factor']:.2f}x |
| With segmentation (--segment) | {seg['files_processed']} | {seg['elapsed_s']:.2f}s | {seg['throughput_files_per_sec']:.2f} files/sec | {seg['real_time_factor']:.2f}x |

Checkpoint resume: second run skipped {data['checkpoint_resume']['second_run_skipped']} files and completed in {data['checkpoint_resume']['second_run_elapsed_s']:.2f}s.

## CLI Help: `curate`

```
{help_curate.strip()}
```

## CLI Help: `inspect`

```
{help_inspect.strip()}
```

## Tests

- Run `pytest --cov=audiotrove` to view coverage. This repository's most recent run showed >85% coverage.

## Installation

```bash
pip install -e .
```

## License

MIT
"""

    (base / "README.md").write_text(readme, encoding="utf8")


if __name__ == "__main__":
    main()
