# AudioTrove

AudioTrove curates local audio into training-ready datasets. It is for teams
preparing speech/TTS corpora that need repeatable VAD, silence trimming, SNR,
duration filtering, resumable processing, and manifests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for component contracts and extension
details.

## Install

Python 3.10+ and a working PyTorch/torchaudio installation are required.
`soundfile` provides the fallback decoder and WAV output; ffmpeg is useful when
your torchaudio backend needs it.

```bash
git clone https://github.com/onepizzateam/AudioTrove.git
cd AudioTrove
python -m pip install -e .
audiotrove --version
```

## Quickstart

Put WAV or FLAC files in a directory, then create curated WAVs and manifests:

```bash
audiotrove curate ./recordings ./curated-tts --tts \
  --extensions wav,flac --tts-min-duration 2 \
  --tts-max-duration 15 --tts-snr-min 20 --workers 4
```

The TTS path runs VAD, trims leading/trailing silence, estimates SNR, checks
post-trim duration, writes curated WAVs, and appends manifests. See the
architecture reference for the complete data flow.

## CLI reference

`audiotrove curate INPUT_PATH OUTPUT_PATH --tts` supports:

| Flag | Type/default | Description |
|---|---|---|
| `--tts` | flag | Select the TTS pipeline. |
| `--tts-min-duration` | float / `2.0` | Minimum post-trim duration. |
| `--tts-max-duration` | float / `15.0` | Maximum post-trim duration. |
| `--tts-snr-min` | float / `20.0` | Minimum SNR in dB. |
| `--workers` | integer / `1` | Local executor worker count. |
| `--extensions` | string / `wav` | Comma-separated directory extensions. |

The generic `curate` path also exposes `--vad-threshold`, `--snr-min`,
`--format`, `--checkpoint`, `--segment`, and `--enhance`; those apply to its
JSONL workflow, not TTS mode.

## Output format

The output directory contains `checkpoint.db`, curated WAVs, `metadata.csv`,
and `filelist.txt`.

```text
# metadata.csv (LJSpeech)
utterance.wav|speaker_00|transcription

# filelist.txt (F5-TTS)
output/utterance.wav<TAB>4.2500<TAB>transcription
```

The WAV path is the curated, post-trim output. Transcription is empty unless
provided in document metadata.

## Benchmark

Real Silero VAD, LibriSpeech dev-clean (2,703 FLAC clips; 5.39 h input):

| Workers | Corpus | Kept | Filtered | Wall time | RTFx | Clips/sec |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LibriSpeech dev-clean (2703 clips) | 2123 | 580 | 3083.50s | 6.3× | 0.88 |
| 4 | LibriSpeech dev-clean (2703 clips) | 2123 | 580 | 1968.61s | 9.9× | 1.37 |

RTFx is total input audio duration divided by wall time. The curated output
contained 13,113.5 seconds (3.64 h) of WAV audio.

## Extending AudioTrove

Add filters, transformers, or exporters through the base interfaces and test
their behavior in isolation plus an end-to-end pipeline test. The
[extension guide](ARCHITECTURE.md#extension-guide) documents the contracts.

## License

[MIT](LICENSE)
