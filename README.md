# AudioTrove 🎙️→📦

[![CI Status](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml/badge.svg)](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Composable audio dataset curation: VAD, SNR filtering, deterministic segmentation, resumable checkpointing, and optional process-parallel execution.

Quick demo:

```bash
audiotrove curate ./raw_audio ./output \
  --vad-threshold 0.3 \
  --snr-min 15.0 \
  --workers 4 \
  --checkpoint checkpoints/run.db
```

What it does
- VAD filtering (Silero when available, energy fallback recorded per-file)
- SNR scoring (uses VAD timestamps when available)
- Deterministic segmentation (fan-out into per-speech segments)
- SQLite checkpointing for resumable runs
- Checkpoint-safe process-parallel execution (`--workers`)

Performance (real run)

Benchmarked on LibriSpeech dev-clean WAV fixtures, Python 3.13.13, Windows-11-10.0.26200-SP0.

| Mode | Files | Wall time | Throughput | Real-time factor |
|------|-------:|----------:|-----------:|-----------------:|
| Sequential (--workers 1) | 2703 | 144.68s | 18.68 files/sec | 87.17x |
| Parallel (--workers 4) | 2703 | 93.81s | 28.81 files/sec | 134.44x |
| With segmentation (--segment) | 2703 | 633.88s | 4.26 files/sec | 2.62x |

Results saved to `benchmarks/e2e_results.json` when running `scripts/benchmark_e2e.py`.

Tests

- 81 tests passed on the benchmark run; coverage for `audiotrove` ≈ 88% (see coverage report).

## Real benchmark (LibriSpeech dev-clean)

We also ran an end-to-end benchmark on LibriSpeech dev-clean (2703 WAV files, pre-converted to 16 kHz WAVs). Results (full JSON saved to `benchmarks/e2e_results.json`):

| Mode | Files | Wall time | Throughput | Real-time factor |
|------|-------|-----------|------------|-----------------|
| Sequential (--workers 1) | 2703 | 144.68s | 18.68 files/sec | 87.17x |
| Parallel (--workers 4) | 2703 | 93.81s | 28.81 files/sec | 134.44x |
| With segmentation (--segment) | 2703 | 633.88s | 4.26 files/sec | 2.62x |

Checkpoint resume: first run processed 2703 files; second run skipped all 2703 files and completed in 9.94s.

Installation

```bash
git clone https://github.com/onepizzateam/AudioTrove
cd AudioTrove
pip install -e .
```

Usage

```text
audiotrove curate INPUT_PATH OUTPUT_PATH [OPTIONS]
audiotrove inspect INPUT_PATH [OPTIONS]
```

See `audiotrove --help` and `audiotrove curate --help` for detailed CLI options.

Limitations & roadmap

- `inspect` currently focuses on WAV files; `curate` supports `--extensions` for mixed formats.
- Parallel execution is local (process-based), not distributed across machines.
- More coverage and edge-case tests for readers and some parallel code paths are planned.

Contributing

See CONTRIBUTING.md for test and development instructions.

License

MIT
