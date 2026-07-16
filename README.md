# AudioTrove 🎙️→📦

[![CI](badge)] [![PyPI](badge)] [![License: MIT](badge)]

**VAD filtering, SNR scoring, segmentation, and resumable checkpointing. One `pip install`. No GPU. No cluster.**

---

## the problem

Every team curating speech data reimplements the same glue: load audio, run VAD, estimate SNR, checkpoint so a crash doesn't cost hours, write a manifest. The tools that do all of this — NVIDIA NeMo Curator, Alibaba Data-Juicer — require GPU infrastructure, Ray clusters, or hours of setup. There was no lightweight version for the common case: a researcher on a laptop or a single box who needs clean audio by tomorrow.

AudioTrove is that version.

---

## install

```
pip install audiotrove
```

That's it. No CUDA. No Ray. No multi-gigabyte framework. Silero VAD downloads its model (~5MB) on first run and caches it locally — subsequent runs are fully offline.

---

## 30-second demo

```
Starting curation pipeline
  Input:  tests/fixtures/
  Output: demo_out/manifest.jsonl
  Checkpoint: demo_out/checkpoint.db
  Workers: 1
  Extensions: wav
  Filters: silero_vad, snr_filter
  Segmentation: disabled

WARNING:audiotrove.io.readers:Failed to load C:/Users/a558087/AudioTrove/tests/fixtures/corrupt.wav: Error opening <_io.BytesIO object at 0x000002342D0160C0>: Error in WAV file. No 'data' chunk marker.
  Curation Results   
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Count ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ Processed │ 5     │
│ Kept      │ 0     │
│ Filtered  │ 7     │
└───────────┴───────┘
✓ Manifest written to demo_out/manifest.jsonl
```

---

## how it compares

| | NeMo Curator | Data-Juicer | webrtcvad (DIY) | AudioTrove |
|---|---|---|---|---|
| Install | `pip install nemo_toolkit[all]` (multi-GB, CUDA required) | `pip install py-data-juicer` + Ray | `pip install webrtcvad` + write VAD + SNR + checkpoint yourself | `pip install audiotrove` |
| GPU required | Yes | Optional | No | **No** |
| VAD | ✓ GPU-accelerated | ✓ | ✓ (WebRTC, lower accuracy) | ✓ Silero (CPU) |
| SNR filtering | ✓ SIGMOS/UTMOS (GPU) | ✗ | ✗ | ✓ VAD-based (CPU) |
| Checkpointing | ✓ | ✓ Ray | ✗ | ✓ SQLite |
| Segmentation | ✓ | ✗ | ✗ | ✓ |
| JSONL manifest | ✓ | ✓ | ✗ | ✓ |
| Laptop-friendly | ✗ | ✗ | Partial | **✓** |
| Time to first run | Hours | 30+ min | Write the code | **< 10 min** |

**AudioTrove is not a replacement for NeMo Curator at GPU scale.** It's the tool for the common case: a corpus that fits on one machine, a researcher who needs clean audio without standing up infrastructure.

---

## what it does

- Apply Silero VAD (CPU) to detect speech regions.
- Score clips by SNR using VAD timestamps when available.
- Optionally segment files into per-speech segments (fan-out).
- Resumable runs via an on-disk SQLite checkpoint and JSONL manifest output.

Sample manifest entry:
```
{
  "doc_id": "d9efa01d2b6465be",
  "source_path": "C:/tmp/librispeech_dev_clean/clip_00000.flac",
  "sample_rate": 16000,
  "duration_seconds": 5.855,
  "metadata": {
    "vad_speech_ratio": 0.8574,
    "vad_speech_timestamps": [
      {
        "start": 8224,
        "end": 88544
      }
    ],
    "vad_backend": "silero",
    "snr_db": 39.24
  },
  "pipeline_version": "0.1.0",
  "processed_at": "2026-07-16T11:07:37.232819Z"
}
```

---

## real-world results

Benchmarked on LibriSpeech dev-clean (2703 clips, 5.4h, ~0.99h) — the standard evaluation corpus for speech processing tooling (used by Lhotse, SpeechBrain, and NVIDIA NeMo).

### before curation

| Metric | Value |
|--------|-------|
| Total clips | 500 |
| Total duration | 0.99h |
| Mean clip duration | 7.1s |
| Mean SNR (all clips) | 30.3 dB |
| Mean VAD speech ratio (all clips) | 0.804 |

### after curation (`--vad-threshold 0.3 --snr-min 15`)

| Metric | Value |
|--------|-------|
| Kept clips | 462 (92.4%) |
| Filtered clips | 38 (7.6%) |
| Total kept duration | 0.93h |
| Mean SNR (kept clips) | 31.8 dB |
| Mean VAD speech ratio (kept clips) | 0.805 |

### throughput

| Mode | RTFx | Wall time for 0.99h corpus |
|------|------|--------------------------|
| Sequential (`--workers 1`) | 40.8x | 87.4s |
| Parallel (`--workers 4`) | 36.0x | 99.0s |

> First run downloads the Silero VAD model (~5MB, one-time). Numbers above reflect warm-cache runs. Run `python scripts/benchmark_e2e.py --corpus-dir <your-audio-dir>` to reproduce on your hardware.

> Note: with 500 clips, parallel spawn overhead can exceed per-file speedup. Parallel scaling improves on corpora of 5000+ clips.

---

## usage

```
Usage:  [OPTIONS] INPUT_PATH OUTPUT_PATH

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

```
Usage:  [OPTIONS] INPUT_PATH

  Show statistics for an audio directory without filtering.

Options:
  --extensions TEXT  Comma-separated audio extensions to inspect (e.g.
                     "wav,mp3,flac").  [default: wav]
  --limit INTEGER    Show stats for first N files. Cap the number of files
                     inspected (default: all).
  --help             Show this message and exit.
```

---

## adding a custom filter

Create a class that implements `filter(self, AudioDocument) -> bool` (for keep/skip) or an `AudioTransformer` that returns a modified `AudioDocument`. See `audiotrove/filters` for examples.

---

## current limitations

- Silero is run on CPU by default; for extremely large corpora consider GPU-accelerated tooling.
- Hugging Face streaming datasets or network downloads may require an `HF_TOKEN` in restricted environments.
- This tool focuses on simplicity and reproducibility, not on out-of-the-box state-of-the-art denoising or perceptual quality metrics.

---

## roadmap

- Improve parallel worker startup and model warm-up to scale better on medium-sized corpora.
- Add optional prefetching for network-backed datasets.
- Provide an official Docker image for reproducible benchmarking.

---

## license

MIT
