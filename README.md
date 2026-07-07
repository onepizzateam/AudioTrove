# AudioTrove 🎙️→📦

[![CI Status](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml/badge.svg)](https://github.com/onepizzateam/AudioTrove/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Composable audio dataset curation. VAD, SNR filtering, segment-level fan-out, parallel execution, and resumable checkpointing as a pip-installable pipeline instead of a full ML-curation stack.

```bash
audiotrove curate ./raw_audio ./output \
  --vad-threshold 0.3 \
  --snr-min 15.0 \
  --workers 4 \
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
- **Segment-level VAD fan-out** — `VADSegmenter` splits a file into per-speech-segment sub-documents instead of judging a whole clip at once, so one bad segment doesn't sink an otherwise-good file. Segment doc_ids are deterministic (derived from the parent doc_id + segment timing), so re-running is idempotent.
- **Checkpointing** — every processed doc_id is recorded in SQLite. Kill the process mid-run, rerun the same command, it skips what's already done.
- **Parallel execution** — `--workers N` fans document processing out across a `ProcessPoolExecutor`. Filters and transformers run in worker processes; the main process is the only one that touches the checkpoint database and writes the manifest, so parallel runs stay checkpoint-safe without extra coordination.
- **JSONL manifest output** — one line per kept clip, with whatever metadata each filter attached (`vad_speech_ratio`, `snr_db`, etc.)

```json
{"doc_id": "abc123", "source_path": "raw/clip1.wav", "sample_rate": 16000, "duration_seconds": 5.2, "metadata": {"vad_speech_ratio": 0.92, "vad_backend": "silero", "snr_db": 18.5}}
```

## the problem

Every lab or team curating speech data ends up writing the same glue: load audio, run VAD, run SNR, checkpoint so a crash doesn't cost 6 hours, write a manifest. NVIDIA NeMo Curator and Alibaba Data-Juicer do this and more — GPU-scale fan-out, quality-model chains, distributed orchestration — but that's real infrastructure weight if you just want VAD+SNR+segmentation+resumability on a laptop or a single box.

AudioTrove is the small version of that: three filter/transformer interfaces (including fan-out), a executor that runs sequentially or across local worker processes, and SQLite checkpointing. Not a replacement for NeMo Curator at scale — a lighter thing for the common case.

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
  --workers INTEGER       Number of worker processes for parallel execution. [default: 1]
  --extensions TEXT       Comma-separated audio file extensions to process (e.g. "wav,mp3,flac"). [default: wav]
```

```
audiotrove inspect INPUT_PATH [OPTIONS]
```

Shows duration distribution and file counts for a directory without filtering anything — useful before picking thresholds. `inspect` currently only globs `.wav`; use `curate --extensions` if your corpus has mixed formats.

## architecture

Four locked contracts, documented in full in `ARCHITECTURE.md`:

- **`AudioDocument`** — canonical object: mono float32 audio, normalized sample rate, deterministic doc_id, a freeform metadata dict that filters append to.
- **`AudioFilter`** — returns a bool. Stateless and independently testable.
- **`AudioTransformer`** — returns exactly one modified doc.
- **`AudioFanOutTransformer`** — returns zero, one, or many docs from a single input. This is what segmentation (`VADSegmenter`) is built on; the emitted docs get deterministic child doc_ids so re-running the pipeline is idempotent and checkpoint-safe.
- **`LocalExecutor`** — walks reader → pipeline → writer. Runs sequentially by default (`num_workers=1`, byte-for-byte the original behavior) or fans document processing out to a `ProcessPoolExecutor` when `num_workers>1`. Checkpoints each doc_id in SQLite and logs (not swallows) any exception a filter or transformer raises.

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

- **`inspect` is `.wav`-only.** `curate` accepts `--extensions` for multi-format globbing; `inspect` doesn't expose that yet.
- **Parallel execution is process-based, not distributed.** `--workers` scales across cores on one machine via `ProcessPoolExecutor`; it doesn't fan out across machines. Fine for a laptop or a single box, not a cluster.
- **`readers.py` correctness on messy real-world audio is still the least-battle-tested path.** Coverage is solid on the happy path and common failure cases, but if VAD or SNR filtering behaves oddly on real audio, that's the most likely place — see "PRs and issues welcome" below.
- **No GPU batching.** VAD and SNR run per-document; there's no batched inference path for throughput on a GPU box yet.

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

62 tests passing as of this writing, ~73% overall statement coverage. Core document/filter/writer logic and the CLI are well-covered; `executor/local.py`'s parallel code paths and some `vad.py` edge branches are the areas with the most room to grow.

## roadmap

Segment-level VAD fan-out and process-parallel execution (originally slated for a later phase) have landed. What's left for the next phase:

- Multi-format support in `inspect`, not just `curate`.
- GPU-batched VAD/SNR inference for throughput.
- Deeper test coverage on `executor/local.py`'s parallel paths and `vad.py`.
- Distributed (multi-machine) execution — currently out of scope; see `ROADMAP.md` for the longer-term thinking here.

See `ROADMAP.md` for full reasoning and phase gates.

PRs and issues welcome, especially against the limitations above — if VAD or SNR filtering behaves oddly on real audio you have, an issue with a sample file is the most useful thing you can send.

## license

MIT
