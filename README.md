# AudioTrove 🎙️→📦

[![CI Status](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml/badge.svg)](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Composable audio dataset curation. VAD, SNR filtering, and resumable checkpointing as a pip-installable pipeline instead of a full ML-curation stack.

```bash
audiotrove curate ./raw_audio ./output \
  --vad-threshold 0.3 \
  --snr-min 15.0 \
  --checkpoint checkpoints/run.db
```

```
Curation Results
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Count ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ Processed │  1000 │
│ Kept      │   450 │
│ Filtered  │   550 │
└───────────┴───────┘
✓ Manifest written to output/manifest.jsonl
```

## what it does

You point it at a directory of audio. AudioTrove runs each file through:

- **VAD filtering** — Silero VAD detects speech vs. silence/music, falls back to an energy-based heuristic if Silero can't load (network, no torch.hub access). Which backend ran is recorded per-file, not hidden.
- **SNR filtering** — signal-to-noise ratio estimated from VAD speech/non-speech segments, with an energy-percentile fallback when no VAD timestamps exist.
- **Checkpointing** — every processed `doc_id` is recorded in SQLite. Kill the process mid-run, rerun the same command, it skips what's already done.
- **JSONL manifest output** — one line per kept clip, with whatever metadata each filter attached (`vad_speech_ratio`, `snr_db`, etc.)

```json
{"doc_id": "abc123", "source_path": "raw/clip1.wav", "sample_rate": 16000, "duration_seconds": 5.2, "metadata": {"vad_speech_ratio": 0.92, "vad_backend": "silero", "snr_db": 18.5}}
```

## the problem

Every lab or team curating speech data ends up writing the same glue: load audio, run VAD, run SNR, checkpoint so a crash doesn't cost 6 hours, write a manifest. NVIDIA NeMo Curator and Alibaba Data-Juicer do this and more — segment-level filtering, GPU-scale fan-out, quality-model chains — but that's real infrastructure weight if you just want the VAD+SNR+resumability part on a laptop or a single box.

AudioTrove is the small version of that: three filter/transformer interfaces, a sequential executor, SQLite checkpointing. Not a replacement for NeMo Curator at scale — a lighter thing for the common case.

## installation

Not on PyPI yet. Install from source:

```bash
git clone https://github.com/onepizzateam/AudioTrove
cd AudioTrove
pip install -e .
```

Optional extras:
```bash
pip install -e ".[s3]"          # s3:// audio paths
pip install -e ".[gcs]"         # gs:// audio paths
pip install -e ".[diarize]"     # speaker diarization (pyannote, requires a HuggingFace token)
pip install -e ".[embed-dedup]" # semantic deduplication (ECAPA-TDNN)
```

## usage

```
audiotrove curate INPUT_PATH OUTPUT_PATH [OPTIONS]

Options:
  --vad-threshold FLOAT   Minimum speech ratio (0-1) to keep a clip. [default: 0.3]
  --snr-min FLOAT         Minimum SNR in dB to keep a clip. [default: 15.0]
  --format [jsonl]        Output format. [default: jsonl]
  --checkpoint PATH       Path to checkpoint database for resumable runs.
```

```
audiotrove inspect INPUT_PATH [OPTIONS]
```
Shows duration distribution and file counts for a directory without filtering anything — useful before picking thresholds.

## architecture

Three locked contracts, documented in full in [ARCHITECTURE.md](ARCHITECTURE.md):

- `AudioDocument` — canonical object: mono float32 audio, normalized sample rate, deterministic `doc_id`, a freeform `metadata` dict that filters append to.
- `AudioFilter` / `AudioTransformer` — the only two block types. Filters return a bool, transformers return a modified doc. Stateless, so they're independently testable and reorderable.
- `LocalExecutor` — walks reader → pipeline → writer, checkpoints each `doc_id` in SQLite, logs (not swallows) any exception a filter raises.

Adding a custom filter:

```python
from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument

class MyCustomFilter(AudioFilter):
    name = "my_custom_filter"

    def filter(self, doc: AudioDocument) -> bool:
        doc.metadata['my_metric'] = compute_something(doc.audio)
        return doc.metadata['my_metric'] > threshold
```

## current limitations

- Sequential execution only — no multiprocessing yet. A `--workers` flag existed at one point and did nothing; it's been removed rather than left fake. Real parallel execution (with checkpoint-safe SQLite writes) is a Phase 1 item, see [ROADMAP.md](ROADMAP.md).
- `.wav` only via the CLI glob pattern — other formats work if you use `LocalAudioReader` directly with a different glob, but the CLI doesn't expose it yet.
- Whole-clip filtering, not segment-level. NeMo Curator's approach — fan out one file into per-segment tasks so a single bad segment doesn't sink the whole clip — is a better design and is on the roadmap, not yet implemented here.
- `readers.py` is effectively untested right now (0% coverage). Load failures are logged, but the read path itself needs more test coverage before I'd trust it on messy real-world corpora.

## what AudioTrove is not

- Not a training framework — it prepares data, it doesn't train models.
- Not ASR/TTS — no transcription, no generation.
- Not a distributed query engine — JSONL + SQLite, fine up to some scale, not built for 10k+ hour corpora (consider WebDataset/HDF5 for that).
- Not a signal-processing library — no MFCCs, no spectral analysis. High-level filtering and I/O only.

## testing

```bash
pytest -v
pytest --cov=audiotrove --cov-report=html
```

27 tests passing as of this writing. Coverage is uneven — core filters and executor are well-covered, the CLI and reader are the weak spots (see limitations above).

## v0.1 scope

- Real parallelism in `LocalExecutor` (checkpoint-safe under multiple workers)
- Segment-level VAD fan-out instead of whole-clip pass/fail
- Multi-format support exposed through the CLI, not just `.wav`
- Test coverage on `readers.py` and `cli/main.py`

PRs and issues welcome, especially against the limitations above — if VAD or SNR filtering behaves oddly on real audio you have, an issue with a sample file is the most useful thing you can send.

## license

MIT