# AudioTrove

> Reproducible audio curation for teams building speech and TTS datasets from raw recordings.

[![CI](https://img.shields.io/github/actions/workflow/status/onepizzateam/AudioTrove/ci.yml?branch=main&label=CI)](https://github.com/onepizzateam/AudioTrove/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/onepizzateam/AudioTrove)](LICENSE)

## What AudioTrove is

Building a TTS corpus from raw audio is repetitive and easy to get wrong:
files need decoding and normalization, silence must be removed consistently,
speech and noise quality must be checked, rejected clips need accounting, and
an interrupted run must not make a team start from zero. AudioTrove packages
those steps into a local, resumable curation pipeline.

It discovers local audio, runs real Silero VAD, trims leading and trailing
silence, estimates SNR, applies post-trim duration bounds, and writes curated
WAVs plus LJSpeech- and F5-TTS-compatible manifests. Every processed document
is recorded in SQLite, giving repeated runs deterministic skip behavior rather
than an ad-hoc ffmpeg loop that must be manually reconciled.

AudioTrove is deliberately a curation tool, not a transcription service or a
model trainer. It produces inspectable training inputs and manifests that a
training framework can consume. The detailed component contracts live in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Features

- Uses Silero VAD with cached local loading and a non-interactive `trust_repo=True` remote fallback.
- Computes VAD-aware SNR and records `snr_db` on each processed document.
- Trims outer silence while retaining configurable padding on both sides of speech.
- Applies duration bounds after trimming, with explicit short/long rejection metadata.
- Writes curated WAV files and LJSpeech `metadata.csv` plus F5-TTS `filelist.txt`.
- Stores processed document IDs in SQLite so interrupted runs can resume.
- Supports sequential execution and multi-worker local execution.
- Discovers WAV, FLAC, and other requested extensions through `fsspec` globs.
- Processed LibriSpeech dev-clean (5.39 h, 2,703 clips) at 6.3 RTFx with one worker.
- Uses Ruff in the declared development tooling.

## Requirements

AudioTrove requires Python 3.10 or later. Core dependencies declare
`torch>=2.0.0`, `torchaudio>=2.0.0`, `numpy>=1.24.0`, `soundfile`, `fsspec`,
Click, Rich, and PyYAML. Torchaudio is the preferred decoder; the reader falls
back to soundfile when appropriate. ffmpeg is not a direct Python dependency,
but may be needed by the torchaudio backend for formats it cannot decode alone.

The test suite is designed for Windows, macOS, and Linux. On Windows,
multi-worker execution uses a thread pool so the loaded Silero model is shared
safely; on other platforms it uses process workers.

## Installation

Install the source checkout:

```bash
git clone https://github.com/onepizzateam/AudioTrove.git
cd AudioTrove
python -m pip install -e .
```

Install development tooling when running tests or lint:

```bash
python -m pip install -e ".[dev]"
```

Verify that the command is available:

```bash
audiotrove --version
```

## Quickstart

### Curate a WAV directory

For ordinary WAV recordings, use the TTS pipeline’s defaults: two to fifteen
seconds after trimming, 20 dB SNR, 150 ms retained silence, and one worker.

```bash
audiotrove curate ./recordings ./curated-tts --tts
```

The CLI reports a TTS completion summary with kept and filtered counts, total
duration for that invocation, and manifest paths. It creates the output
directory before processing.

### Curate a FLAC corpus with four workers

Use explicitly requested extensions and stricter SNR when source audio is a
FLAC corpus:

```bash
audiotrove curate ./speech-flac ./curated-flac --tts \
  --extensions flac --workers 4 --tts-snr-min 25 \
  --tts-min-duration 2 --tts-max-duration 15
```

After a successful run, the directory contains artifacts like:

```text
curated-flac/
├── checkpoint.db
├── metadata.csv
├── filelist.txt
├── speaker-0001.wav
└── speaker-0002.wav
```

## Pipeline explained

A discovered file is loaded by `LocalAudioReader`, downmixed to mono, and
resampled to 16 kHz. The executor presents that `AudioDocument` to
`SileroVADFilter`, which records speech timestamps and speech ratio. A clip
with too little speech stops there and is counted as skipped; it does not stop
the run.

For a surviving clip, `SilenceTrimmingTransformer` finds the outermost speech
timestamps, keeps the requested padding, updates the audio array and duration,
and shifts timestamps to the trimmed origin. `SNRFilter` then uses the VAD
speech/non-speech regions to calculate an SNR estimate. Finally,
`DurationBucketFilter` checks the post-trim duration. A failure at any filter
short-circuits that file, records its original document ID in the checkpoint,
and lets the executor continue with later files.

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

Accepted documents are exported as WAV and appended to requested manifests.
The executor, not a worker, owns export and SQLite writes; this makes manifest
and checkpoint mutation serial and resumable.

## CLI reference

Use `audiotrove curate INPUT_PATH OUTPUT_PATH --tts`. TTS-specific options are
listed first. The remaining flags belong to the generic JSONL path and are
accepted by the same command but do not configure `tts_pipeline()`.

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--tts` | flag | `False` | Run TTS curation and manifest export. |
| `--tts-min-duration` | float | `2.0` | Minimum TTS duration in seconds. |
| `--tts-max-duration` | float | `15.0` | Maximum TTS duration in seconds. |
| `--tts-snr-min` | float | `20.0` | Minimum TTS SNR in dB. |
| `--workers` | integer | `1` | Local worker count. |
| `--extensions` | string | `wav` | Comma-separated extensions, for example `wav,mp3,flac`. |
| `--vad-threshold` | float | `0.3` | Generic-path minimum speech ratio. |
| `--snr-min` | float | `15.0` | Generic-path minimum SNR in dB. |
| `--format` | choice | `jsonl` | Generic-path output format. |
| `--checkpoint` | string | none | Generic-path checkpoint database path. |
| `--segment` | flag | `False` | Generic-path VAD fan-out segmentation. |
| `--enhance` | flag | `False` | Generic-path DeepFilterNet2 enhancement. |
| `-v`, `--verbose` | flag | `False` | Enable DEBUG logging. |

The TTS pipeline itself fixes VAD minimum speech ratio at `0.1` and VAD model
threshold at `0.5`; `--vad-threshold` is not forwarded in TTS mode.

## Output format

The exporter writes WAV files using each source file’s stem. It then writes a
LJSpeech row when `ljspeech` is selected and an F5-TTS row when `f5tts` is
selected. The default requests both formats.

`metadata.csv` is pipe-delimited:

```text
1272-128104-0000.wav|speaker_00|
```

Columns are `filename`, `speaker_id`, and `transcription`. The transcription
is an empty string unless `doc.metadata["transcription"]` is set.

`filelist.txt` is tab-delimited:

```text
curated-flac/1272-128104-0000.wav	5.3200	
```

Columns are the curated WAV path, the post-trim duration in seconds, and
transcription. The path points to the exported WAV, not the untrimmed source.
WAV names derive from source stems, so source-stem collisions must be avoided
within one output directory.

`checkpoint.db` has a `processed` SQLite table keyed by document ID. It is an
operational artifact, not a training manifest.

## Resumability

The executor initializes `checkpoint.db` with a primary-keyed `processed`
table. Before a document enters the pipeline, it checks that table; after an
accepted document is exported, or after a rejected original document is
handled, it inserts the relevant ID. An interrupted run can therefore be
restarted with the same input and output paths: already recorded IDs are
skipped, while unrecorded IDs continue through the pipeline.

Manifests are opened in append mode. Keep the checkpoint with its output
directory. Deleting the database while retaining manifests disables skip
deduplication and can append duplicate rows.

## Benchmark

The committed benchmark used LibriSpeech dev-clean: 2,703 FLAC clips and
19,396.1 seconds (5.39 h) of input. It used real cached Silero VAD, not the
energy fallback. Both configurations retained 2,123 clips and filtered 580.

| Workers | Corpus | Kept | Filtered | Wall time | RTFx | Clips/sec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LibriSpeech dev-clean (2703 clips) | 2123 | 580 | 3083.50 s | 6.3× | 0.88 |
| 4 | LibriSpeech dev-clean (2703 clips) | 2123 | 580 | 1968.61 s | 9.9× | 1.37 |

RTFx is total input audio duration divided by wall time. Four workers improve
throughput by about 1.57× rather than 4× because decoding and WAV/manifests
require I/O, exports and checkpoints stay in the main process, and Silero
inference is protected by a shared lock. The result JSON contains aggregate
kept/filtered counts, not a per-filter rejection breakdown; inspection of
component metadata/checkpoint data is needed for per-reason analysis.

## Integrating with training frameworks

F5-TTS-style consumers can read the default tab-separated filelist directly:

```python
for line in open("curated-tts/filelist.txt", encoding="utf-8"):
    wav_path, duration, text = line.rstrip("\n").split("\t")
```

LJSpeech-style consumers can use `metadata.csv` with WAVs in the same output
directory:

```python
for line in open("curated-tts/metadata.csv", encoding="utf-8"):
    filename, speaker_id, text = line.rstrip("\n").split("|", 2)
```

AudioTrove intentionally leaves the transcription field empty when upstream
metadata has none; generate or attach transcripts before training a supervised
TTS model.

## Extending AudioTrove

Custom filters subclass `AudioFilter`, expose a name, and return a boolean
from `filter`. The executor will short-circuit a false result for that document.

```python
from audiotrove.base import AudioFilter

class PeakLimitFilter(AudioFilter):
    name = "peak_limit"

    def __init__(self, maximum: float = 0.99):
        self.maximum = maximum

    def filter(self, doc):
        return float(abs(doc.audio).max()) <= self.maximum
```

Add the filter to a pipeline list, add focused pass/fail tests, and add an
end-to-end test where its placement matters. New export formats should follow
the exporter’s `write(doc)` and `output_files` contract. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full extension guide.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and keep changes small and tested.
Ruff is included in the development extras; add tests for new components and
run them across the supported Windows, macOS, and Linux environments.

## License

AudioTrove is released under the [MIT License](LICENSE).
