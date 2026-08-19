# AudioTrove architecture reference

## Overview

AudioTrove curates local audio into training-ready datasets and supports the
CPU-real Piper training path. It discovers audio,
normalizes it into `AudioDocument` objects, applies ordered filters and
transformers, and writes manifests or TTS WAV/manifest outputs. Piper is the
CPU-first trainer; F5-TTS, StyleTTS2, Matcha and other GPU-oriented/multi-trainer
modules are not on the CPU-first path and are kept for future GPU rollout.
Kokoro is inference-only. AudioTrove is not a transcription service or
distributed orchestration system.

## Repository layout

```text
audiotrove/
  base.py                 component interfaces
  document.py             AudioDocument data model
  cli/main.py             Click commands
  executor/local.py       sequential/parallel execution and checkpoints
  filters/                VAD, SNR, duration, enhancement
  transformers/           audio transformations
  exporters/              TTS manifest export
  io/readers.py           discovery, decoding, resampling
  io/writers.py           JSONL export
  pipelines/tts.py        TTS assembly entry point
  utils/hashing.py        stable document IDs
tests/                    component and integration contracts
benchmarks/               benchmark runners and result JSON
```

`AudioDocument` carries mono `numpy.ndarray` audio, `sample_rate`,
`source_path`, `duration_seconds`, a deterministic `doc_id`, and mutable
metadata. Components communicate through that object rather than global state.

## Pipeline architecture

The TTS entry point constructs a `LocalAudioReader`, a
`TTSManifestExporter`, the ordered component list, and a `LocalExecutor`.
The reader yields documents; the executor applies each component; accepted
documents are exported and checkpointed.

```text
glob/file input
  -> LocalAudioReader (decode, mono, 16 kHz)
  -> SileroVADFilter (speech timestamps and ratio)
  -> SilenceTrimmingTransformer (trim leading/trailing silence)
  -> SNRFilter (speech/noise power estimate)
  -> DurationBucketFilter (post-trim bounds)
  -> TTSManifestExporter (WAV + manifests)
  -> checkpoint.db
```

Filtering is short-circuiting: a failed filter prevents later stages and the
original document ID is checkpointed. Transformers receive the document left by
the preceding stage. Fan-out transformers can return several documents.

## Readers

### `LocalAudioReader`

`LocalAudioReader(path_pattern, target_sample_rate=16000,
max_duration_seconds=None, min_duration_seconds=0.5)` accepts one glob/path or
a list. It uses `fsspec` globbing, deduplicates paths, and yields either an
`AudioDocument` or `None` for a load failure. It loads with `torchaudio` when
available, falling back to `soundfile`; if necessary it falls back from a
failed torchaudio backend to in-memory soundfile decoding.

Audio is downmixed by averaging channels and resampled to 16 kHz. Documents
shorter than `min_duration_seconds` are omitted; an optional maximum truncates
audio. IDs are produced from the source path by `make_doc_id`.

Directory input to `tts_pipeline` expands immediate `*.extension` patterns.
Recursive corpus execution therefore uses explicit recursive file patterns or
the benchmark chunk wrapper, which passes individual file paths to the reader.

## Filters

### `SileroVADFilter`

`SileroVADFilter(min_speech_ratio=0.3, threshold=0.5,
window_size_samples=512)` is an `AudioFilter`. It lazily loads Silero from
`~/.cache/torch/hub/snakers4_silero-vad_master` with
`torch.hub.load(..., source="local", trust_repo=True)` when present. Otherwise
it uses the remote `snakers4/silero-vad` hub entry with `force_reload=False`
and `trust_repo=True`. `trust_repo=True` prevents an interactive trust prompt
in non-interactive workers.

Silero timestamps are stored as `vad_speech_timestamps`, speech proportion as
`vad_speech_ratio`, and the backend as `vad_backend`. The filter keeps a clip
when its speech-sample ratio is at least `min_speech_ratio`.

If torch, model loading, or inference is unavailable, it uses an energy VAD:
512-sample frame energies, `mean + std` threshold, and a ten-percent active
frame floor. This is a compatibility fallback, not a benchmark backend. It can
give materially different decisions, so a benchmark stderr line containing
`Falling back to energy-based VAD` invalidates that run.

Models are excluded from pickle state. A module-level inference lock protects
shared inference.

### `SNRFilter`

`SNRFilter(min_snr_db=15.0, use_mos_scoring=False)` uses VAD timestamps to
form speech and non-speech masks. It writes rounded `snr_db` metadata and keeps
the document when SNR is at least the configured threshold. With fewer than
0.1 seconds of noise it writes `snr_note=insufficient_noise_floor` and returns
40 dB; zero noise power also returns 40 dB. Without timestamps it estimates
signal and noise from 512-sample frame energies split at the 75th percentile.

### `DurationBucketFilter`

`DurationBucketFilter(min_duration_seconds=2.0,
max_duration_seconds=15.0)` applies inclusive post-transform bounds. It marks
rejections as `duration_filter_reason=too_short` or `too_long`.

## Transformers

### `SilenceTrimmingTransformer`

`SilenceTrimmingTransformer(padding_ms=150, min_duration_seconds=1.0)` obtains
existing VAD timestamps or runs a private `SileroVADFilter`. It keeps the span
from the first speech start minus padding to the last speech end plus padding,
bounded by the audio array. It updates the contiguous float32 audio array,
duration, timestamps relative to the new origin, and
`trimmed_duration_seconds`. If no timestamps exist or the result would be
shorter than its internal minimum, it leaves audio unchanged.

### `VADSegmenter`

The optional CLI segmenter is an `AudioFanOutTransformer`. It creates one
document per detected speech interval, preserves provenance metadata, and uses
a deterministic hash of parent ID/start/end for each segment ID.

## Executor

`LocalExecutor(pipeline, checkpoint_path=None, num_workers=1)` owns execution,
statistics, export calls, and SQLite writes. `run(reader, writer)` returns
`processed`, `kept`, `skipped`, `errors`, and `errors_by_filter`.

With one worker it processes each document inline in pipeline order. With more
workers it collects unprocessed documents first, then uses a thread pool on
Windows so VAD model state is shared safely; other platforms use
`ProcessPoolExecutor` and preload Silero in workers. The main process always
writes exports and checkpoints. Worker exceptions become error statistics.

## Checkpoint system

The checkpoint is SQLite at the configured path, normally
`<output>/checkpoint.db`. Its table is:

```sql
CREATE TABLE processed (
  doc_id TEXT PRIMARY KEY,
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Before processing, the executor queries the document ID. Completed documents
are skipped on resume. Accepted documents are marked after export; rejected
original documents are also marked. Existing output without a checkpoint is not
deduplicated: manifests are opened in append mode, so deleting a checkpoint
while retaining output can create duplicate rows.

## Exporters

### `TTSManifestExporter`

`TTSManifestExporter(output_dir, speaker_id="speaker_00", export_format=None)`
accepts `ljspeech` and/or `f5tts`. It touches requested manifests, writes each
curated document as `<source stem>.wav` under `output_dir` using soundfile, and
appends manifest records. `total_duration_seconds` is accumulated for that
exporter instance.

`metadata.csv` is LJSpeech-style, pipe-delimited:

```text
filename.wav|speaker_00|transcription
```

`filelist.txt` is F5-TTS-style, tab-delimited:

```text
output/path/filename.wav<TAB>trimmed_duration_seconds<TAB>transcription
```

The transcription defaults to an empty string. Names use the source stem, so
callers must avoid colliding source stems in a shared output directory.

## TTS entry point

```python
tts_pipeline(input_path: str, output_path: str,
             min_duration: float = 2.0, max_duration: float = 15.0,
             snr_min: float = 20.0, padding_ms: int = 150,
             export_format: list[str] = ["ljspeech", "f5tts"],
             extensions: list[str] = ["wav"], workers: int = 1) -> dict
```

`input_path` is one file or a directory; `output_path` holds WAVs, manifests,
and checkpoint. Duration and SNR settings configure their filters; `padding_ms`
configures trimming; `export_format` selects manifests; `extensions` controls
directory discovery; `workers` selects executor concurrency. The returned
summary includes kept, filtered, per-run exporter duration, and output paths.

## CLI

Use `audiotrove curate INPUT_PATH OUTPUT_PATH --tts`. TTS mode maps
`--tts-min-duration` (float, 2.0), `--tts-max-duration` (float, 15.0),
`--tts-snr-min` (float, 20.0), `--workers` (int, 1), and comma-separated
`--extensions` (string, `wav`) to the entry point. The generic flags
`--vad-threshold`, `--snr-min`, `--format`, `--checkpoint`, `--segment`, and
`--enhance` apply to non-TTS curation rather than the TTS pipeline.

## Benchmark results

LibriSpeech dev-clean contained 2,703 FLAC clips and 19,396.1 seconds (5.39 h)
of input. Both real-Silero runs kept 2,123 clips and filtered 580. One worker
took 3,083.50 seconds: 6.3 RTFx and 0.88 clips/s. Four workers took 1,968.61
seconds: 9.9 RTFx and 1.37 clips/s. RTFx is total input duration divided by
wall time. The four-worker speedup is limited by decoding, export I/O, the
shared Silero inference lock, and main-process checkpoint/export ownership.

## Extension guide

Add a filter by subclassing `AudioFilter`, providing `name` and `filter(doc)`;
write metadata needed by downstream stages and test pass/fail behavior. Add a
regular transformer with `AudioTransformer.transform(doc)`, or use
`AudioFanOutTransformer` for a list of documents. Add an exporter with `write`
and `output_files`, then wire it into an entry point and add format/content,
resume, and end-to-end tests.

## Known limitations

Silero fallback is intentionally available for degraded environments but must
not be accepted for benchmark measurements. Large low-memory corpora can OOM
with one long worker run; the benchmark wrappers use 200-file chunks and
`gc.collect()` between chunks. Consequently their aggregated result JSON keeps
counts and wall time but reports `total_duration_seconds` as 0.0; derive kept
duration from exported WAVs. A TODO in `tts_pipeline` defers speaker
consistency filtering until a lightweight backend is available.
