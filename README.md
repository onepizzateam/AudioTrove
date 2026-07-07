# AudioTrove — Open-Source Audio Dataset Curation

**Status:** Phase 0 (Foundation complete)

## The Problem

Audio dataset curation is tedious and repetitive. While frameworks like **NVIDIA NeMo Curator** and **Alibaba Data-Juicer** offer comprehensive ML pipelines, they add significant complexity for teams that just need practical, composable filters (VAD, SNR, deduplication) without the heavy stack. AudioTrove is a **lightweight alternative** designed to be straightforward to understand, extend, and run on modest compute — so you can focus on your data, not infrastructure.

## Installation

```bash
pip install audiotrove
```

Optional extras:
```bash
pip install audiotrove[s3]          # For s3:// audio paths
pip install audiotrove[gcs]         # For gs:// audio paths
pip install audiotrove[diarize]     # For speaker diarization (pyannote)
pip install audiotrove[embed-dedup] # For semantic deduplication (ECAPA-TDNN)
```

## 30-Second Demo

Given a directory of audio files, filter by voice activity and signal-to-noise ratio in one command:

```bash
audiotrove curate ./raw_audio ./output \
  --vad-threshold 0.3 \
  --snr-min 15.0 \
  --checkpoint checkpoints/run.db
```

Outputs a JSONL manifest with metadata for every kept clip:

```json
{"doc_id": "abc123", "source_path": "raw/clip1.wav", "sample_rate": 16000, "duration_seconds": 5.2, "metadata": {"vad_speech_ratio": 0.92, "snr_db": 18.5}}
{"doc_id": "def456", "source_path": "raw/clip2.wav", "sample_rate": 16000, "duration_seconds": 3.8, "metadata": {"vad_speech_ratio": 0.88, "snr_db": 22.1}}
```

## Before and After

**Input:** 100 hours of raw podcast audio (mixed speech, music, noise, silence).

**After Phase 0 curation:**
- VAD filters out music and silence (~40% removed)
- SNR filters keep only clean speech (~15% removed)
- Checkpoint DB allows resumable runs (crash-safe)
- **Result:** ~45 hours of speech-rich, clean data ready for ASR/TTS training

## Core Architecture

AudioTrove is built on three locked contracts:

1. **AudioDocument**: canonical object carrying audio (mono, float32, 16kHz), metadata, and deterministic `doc_id`.
2. **Blocks**: `AudioFilter` (returns bool) and `AudioTransformer` (returns new doc). Stateless, reorderable, independently testable.
3. **Executor**: manages parallelism, checkpointing, and pipeline orchestration. Blocks do not know they are being parallelized.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed contracts and rationale.

## What AudioTrove IS NOT

- **Not a model training framework.** Use PyTorch, TensorFlow, or HuggingFace Transformers to train models; AudioTrove prepares your data.
- **Not audio synthesis or ASR/TTS.** No generation, no transcription. Pure data filtering and curation.
- **Not a database system.** JSONL + SQLite checkpoint, not a distributed query engine. For 10k+ hour corpora, consider WebDataset or HDF5 formats.
- **Not a replacement for librosa.** We don't do spectral analysis, MFCCs, or advanced signal processing. We do high-level filtering and I/O.

## Phase 0 Components

- ✅ `AudioDocument` dataclass
- ✅ `AudioFilter` and `AudioTransformer` abstract base classes
- ✅ `LocalExecutor` with SQLite checkpointing
- ✅ `LocalAudioReader` (fsspec + torchaudio; resample/downmix)
- ✅ `JSONLWriter` (append manifest entries)
- ✅ `SileroVADFilter` (voice activity detection, lazy-load)
- ✅ `SNRFilter` (signal-to-noise ratio estimation)
- ✅ CLI skeleton (`curate`, `inspect` commands)

## How to Add a Custom Filter Block

Create a class inheriting from `AudioFilter`:

```python
from audiotrove.base import AudioFilter
from audiotrove.document import AudioDocument

class MyCustomFilter(AudioFilter):
    @property
    def name(self) -> str:
        return "my_custom_filter"
    
    def filter(self, doc: AudioDocument) -> bool:
        # Examine doc.audio, doc.metadata, etc.
        # Append metadata as needed.
        doc.metadata['my_metric'] = some_value
        # Return True to keep, False to discard.
        return should_keep_this_document
```

Then add it to your pipeline:

```python
from audiotrove.executor.local import LocalExecutor
from audiotrove.io.readers import LocalAudioReader
from audiotrove.io.writers import JSONLWriter

reader = LocalAudioReader('input/*.wav')
writer = JSONLWriter('output/manifest.jsonl')
pipeline = [
    MyCustomFilter(),
    SileroVADFilter(min_speech_ratio=0.3),
    SNRFilter(min_snr_db=15.0),
]
executor = LocalExecutor(pipeline=pipeline, checkpoint_path='checkpoint.db')
stats = executor.run(reader, writer)
print(stats)
```

## Testing

Run the test suite:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest --cov=audiotrove --cov-report=html
```

## Contributing

Contributions welcome! Please:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Add tests for new filters or transformers in `tests/`.
4. Run `pytest` and `ruff check --fix` before submitting a PR.
5. Keep blocks **stateless**. If you need shared state, inject it via the executor or a custom block initializer, never via global variables.

## License

MIT (permissive, suitable for academic and commercial use).

## Acknowledgments

Architecture inspired by DataTrove and the open-source audio ML ecosystem. Built for low-resource language teams, indie TTS developers, and academic labs training speech models on real-world data.
