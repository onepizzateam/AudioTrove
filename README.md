# AudioTrove

> Reproducible audio curation for teams building speech and TTS datasets from raw recordings.

[![CI](https://img.shields.io/github/actions/workflow/status/onepizzateam/AudioTrove/ci.yml?branch=main&label=CI)](https://github.com/onepizzateam/AudioTrove/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/onepizzateam/AudioTrove)](LICENSE)

## What AudioTrove is

AudioTrove makes two honest promises:

1. **Curate (proven):** deterministically turn raw audio into a resumable,
   RAM-bounded, training-ready corpus with VAD, trimming, SNR and manifests.
2. **Train (CPU-real):** Piper is the first-class CPU training path. F5-TTS,
   StyleTTS2, Matcha and other GPU-oriented trainers remain behind optional
   extras, but are not the CPU story.

Kokoro-82M is an optional CPU inference/preview and synthetic-augmentation
utility only; it is not a training framework.

Building a TTS corpus from raw audio is repetitive and easy to get wrong:
On a single CPU worker it processes LibriSpeech dev-clean (5.4 hours, 2,703 clips) at 6.3× real-time.
files need decoding and normalization, silence must be removed consistently,
speech and noise quality must be checked, rejected clips need accounting, and
an interrupted run must not make a team start from zero. AudioTrove packages
those steps into a local, resumable curation pipeline.

It discovers local audio, runs real Silero VAD, trims leading and trailing
silence, estimates SNR, applies post-trim duration bounds, and writes curated
WAVs plus LJSpeech- and F5-TTS-compatible manifests. Every processed document
is recorded in SQLite, giving repeated runs deterministic skip behavior rather
than an ad-hoc ffmpeg loop that must be manually reconciled.

AudioTrove is deliberately CPU-first. It produces inspectable training inputs
and manifests alongside the CPU-real Piper training path. The detailed
component contracts live in
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

```bash
pip install audiotrove
```

> **No GPU?** Install the CPU-only PyTorch build first to avoid a multi-GB CUDA download:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install audiotrove
> ```

For development:

```bash
git clone https://github.com/onepizzateam/AudioTrove.git
cd AudioTrove
pip install -e ".[dev]"
```

### Optional native/runtime setup

The optional Piper training extra builds native eSpeak components on Windows.
It requires Visual Studio C++ Build Tools; the training extra was installed
successfully under Python 3.11 after enabling Windows long paths.

Kokoro preview requires a Python environment supported by its native/runtime
dependencies. It was verified on 2026-08-28 in the dedicated Python 3.11
environment by downloading Kokoro-82M and producing a WAV preview. It remains
an inference-only capability; it is not a training path.

The Piper adapter disables Piper-side silence trimming (AudioTrove has already
curated the clips), uses a validation holdout for real manifests, and supplies
a CPU-safe checkpoint callback. A ten-utterance, 22.05 kHz smoke run completed
one epoch on 2026-08-28 and produced a loadable Lightning checkpoint. A fresh
one-epoch measurement on the same
corpus took 260.47 s and reached 2,646.6 MiB aggregate peak RSS, sampled across
the parent and child processes with psutil.

On the same LibriSpeech dev-clean clip (`1272-128104-0000.flac`), Python 3.11,
CPU-only, base models, and psutil peak-RSS sampling, faster-whisper measured
2.30 s / 490.6 MiB and openai-whisper 8.11 s / 697.5 MiB. A fresh full-corpus
one-worker curation comparison on 2,703 LibriSpeech
dev-clean clips measured 39.661 RTFx with Rust enabled (489.05 s) and 32.540
RTFx with it disabled (596.06 s), with identical 2,123 kept / 580 filtered
results. Peak RSS was not instrumented for that pair; Piper RAM remains
unmeasured.

Verify the CLI is available:

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

### Add transcriptions with Whisper

```bash
pip install audiotrove[transcribe]
audiotrove curate ./recordings ./curated-tts --tts --tts-transcribe
```

This runs `openai-whisper` (base model, CPU) on each kept clip and populates the transcription column in `metadata.csv` and `filelist.txt`. Use `--tts-whisper-model small` for better accuracy at the cost of speed.

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

Use `audiotrove curate INPUT_PATH OUTPUT_PATH --tts`. The TTS pipeline accepts
the TTS-specific options below as well as `--segment`, `--workers`, and
`--extensions`. The remaining generic JSONL flags are accepted by the same
command but do not configure `tts_pipeline()`.

**Global flag:** `--verbose` must come before the subcommand: `audiotrove --verbose curate ...`

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--tts` | flag | `False` | Run TTS curation and manifest export. |
| `--tts-min-duration` | float | `2.0` | Minimum TTS duration in seconds. |
| `--tts-max-duration` | float | `15.0` | Maximum TTS duration in seconds. |
| `--tts-snr-min` | float | `20.0` | Minimum TTS SNR in dB. |
| `--tts-transcribe` | flag | `False` | Transcribe kept clips with Whisper (requires `pip install audiotrove[transcribe]`). |
| `--tts-whisper-model` | str | `base` | Whisper model size: tiny, base, small, medium, large. |
| `--tts-diarize` | flag | `False` | Run speaker diarization before VAD (requires `pip install audiotrove[diarize]`). |
| `--tts-hf-token` | string | `None` | HuggingFace token for pyannote models (required when `--tts-diarize` is set). |
| `--tts-diarize-min-speakers` | integer | `None` | Optional minimum speaker count for diarization. |
| `--tts-diarize-max-speakers` | integer | `None` | Optional maximum speaker count for diarization. |
| `--workers` | integer | `1` | Local worker count. |
| `--extensions` | string | `wav` | Comma-separated extensions, for example `wav,mp3,flac`. |
| `--vad-threshold` | float | `0.3` | Generic-path minimum speech ratio; ignored with `--tts`. |
| `--snr-min` | float | `15.0` | Generic-path minimum SNR in dB. |
| `--format` | choice | `jsonl` | Generic-path output format. |
| `--checkpoint` | string | none | Generic-path checkpoint database path. |
| `--segment` | flag | `False` | VAD fan-out segmentation in generic and TTS modes. In TTS mode, each accepted speech segment is exported to the manifests. |
| `--enhance` | flag | `False` | Generic-path DeepFilterNet2 enhancement; ignored with `--tts`. |

The TTS pipeline itself fixes VAD minimum speech ratio at `0.1` and VAD model
threshold at `0.5`; `--vad-threshold` is not forwarded in TTS mode.

### Speaker Diarization

To perform multi-speaker segmentation using pyannote:

1. Install the `diarize` extra: `pip install audiotrove[diarize]`
2. Accept the model user agreement on HuggingFace for [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1).
3. Pass your HuggingFace user access token using `--tts-hf-token`:

```bash
audiotrove curate ./raw-recordings ./curated-tts --tts \
  --tts-diarize --tts-hf-token hf_xxx \
  --tts-diarize-max-speakers 2
```

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
