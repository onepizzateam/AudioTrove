# AudioTrove

Composable audio dataset curation: VAD, SNR filtering, deterministic segmentation, resumable checkpointing, and optional process-parallel execution.

## Demo

Captured output (cleaned):

```
Starting curation pipeline
  Input:  tests/fixtures
  Output: packaging/demo_out/manifest.jsonl
  Checkpoint: packaging/demo_out/checkpoint.db
  Workers: 1
  Extensions: wav
  Filters: silero_vad, snr_filter
  Segmentation: disabled

WARNING:audiotrove.io.readers:Failed to load C:/Users/a558087/AudioTrove/tests/fixtures/corrupt.wav: Error opening <_io.BytesIO object at 0x000002822F2CD760>: Error in WAV file. No 'data' chunk marker.
  Curation Results   
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Count ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ Processed │ 0     │
│ Kept      │ 0     │
│ Filtered  │ 7     │
└───────────┴───────┘
✓ Manifest written to packaging/demo_out/manifest.jsonl
```

## Performance

Benchmarked on LibriSpeech dev-clean WAV fixtures (2703 files) on Windows-11-10.0.26200-SP0 with Python 3.13.13.

| Mode | Files | Wall time | Throughput | Real-time factor |
|------|------:|----------:|-----------:|-----------------:|
| Sequential (--workers 1) | 2703 | 324.90s | 8.32 files/sec | 59.60x |
| Parallel (--workers 4) | 2703 | 493.78s | 5.47 files/sec | 39.21x |
| With segmentation (--segment) | 2703 | 2145.88s | 1.26 files/sec | 1.01x |

Checkpoint resume: second run skipped 2703 files and completed in 106.09s.

## CLI Help: `curate`

```
Usage: python -m audiotrove.cli.main curate [OPTIONS] INPUT_PATH OUTPUT_PATH

  Curate audio files from INPUT_PATH into OUTPUT_PATH.

  Applies VAD and SNR filters, writes JSONL manifest.

Options:
  --vad-threshold FLOAT  Minimum speech ratio (0-1) to keep a clip.  [default:
                         0.3]
  --snr-min FLOAT        Minimum SNR in dB to keep a clip.  [default: 15.0]
  --format [jsonl]       Output format.  [default: jsonl]
  --checkpoint TEXT      Path to checkpoint database for resumable runs.
  --workers INTEGER      Number of worker processes for parallel execution.
                         [default: 1]
  --extensions TEXT      Comma-separated audio file extensions to process
                         (e.g., "wav,mp3,flac").  [default: wav]
  --segment              Split audio into per-speech-segment sub-documents
                         (VAD fan-out). Each speech segment becomes its own
                         JSONL entry.
  --help                 Show this message and exit.
```

## CLI Help: `inspect`

```
Usage: python -m audiotrove.cli.main inspect [OPTIONS] INPUT_PATH

  Show statistics for an audio directory without filtering.

Options:
  --extensions TEXT  Comma-separated audio extensions to inspect (e.g.
                     "wav,mp3,flac").  [default: wav]
  --limit INTEGER    Show stats for first N files. Cap the number of files
                     inspected (default: all).
  --help             Show this message and exit.
```

## Tests

- Run `pytest --cov=audiotrove` to view coverage. This repository's most recent run showed >85% coverage.

## Installation

```bash
pip install -e .
```

## License

MIT
