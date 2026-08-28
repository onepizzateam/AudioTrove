# Changelog

## 0.1.2 — 2026-08-27

AudioTrove is now explicitly CPU-first: deterministic curation remains the
proven path, Piper is the CPU training path, and Kokoro is inference-only.
This release adds the doctor command, optional Rust resampling with a Python
fallback, checkpoint versioning/WAL journaling, faster-whisper selection,
lightweight fingerprint QC, and the pipeline CLI.

### Benchmark status

Kokoro preview was verified separately in a Python 3.11 environment. Piper CPU
training was verified with a ten-utterance, 22.05 kHz smoke corpus and produced
a loadable checkpoint. The same Piper run took 260.47 s and reached 2,646.6 MiB
aggregate peak RSS (parent plus child processes, sampled every 100 ms). On the same
LibriSpeech dev-clean clip (`1272-128104-0000.flac`), Python 3.11, CPU-only,
base models, and peak RSS sampled with psutil, faster-whisper took 2.30 s and
490.6 MiB while openai-whisper took 8.11 s and 697.5 MiB. The existing
full-corpus curation claim remains 6.3 RTFx on LibriSpeech dev-clean (5.39
hours, 2,703 clips, one worker). A fresh 2026-08-28 one-worker run on that
corpus measured 39.661 RTFx with Rust enabled (489.05 s) and 32.540 RTFx with
it disabled (596.06 s), with identical 2,123 kept / 580 filtered results. Peak
RSS was not instrumented for the curation pair.
The >500 MB peak-RSS experiment was not measured — no representative file available in the execution environment.

The Day 1–9 implementation commits are tagged `day-1` through `day-9`.

### Final implementation report

- Day 1 repositioned the product around proven curation and CPU-real Piper
  training and added the doctor skeleton.
- Day 2 added optional PyO3 Rust glob/decode/rubato-resample acceleration with
  Python fallbacks and a reproducible maturin build command.
- Day 3 added checkpoint schema versioning, crash-safe persistence, and RAM
  budgeting controls.
- Day 4 added selectable faster-whisper CPU transcription while preserving the
  openai-whisper backend and manifest schema.
- Day 5 added lightweight fingerprint deduplication and QC reporting; the
  fingerprint now uses Rust FFT when available and Python otherwise.
- Day 6 completed doctor recommendations, documented Homebrew/pip setup, and
  added the CPU-only install assertion to CI.
- Day 7 promoted Piper, added resumable CPU-safe training and Kokoro
  preview/augmentation, and verified both with real local runtimes.
- Day 8 added resumable curate-to-train-to-preview pipeline execution and an
  interrupted-training resume test.
- Day 9 refreshed the documentation and recorded fresh benchmark artifacts.

Fresh measurements were made on Windows CPU-only hardware: curation on 2,703
LibriSpeech dev-clean clips measured 39.661 RTFx Rust-on versus 32.540 RTFx
Rust-off at one worker; Whisper base on the same clip measured 2.30 s / 490.6
MiB faster-whisper versus 8.11 s / 697.5 MiB openai-whisper; and the Piper
ten-utterance, 22.05 kHz smoke run took 260.47 s / 2,646.6 MiB aggregate peak
RSS and produced a loadable checkpoint. Day 3's requested >500 MB-file
before/after RSS experiment was narrowed because no suitable representative
corpus file was available; no number is claimed for it.

The nine dated tags are `day-1` (2026-08-19), `day-2` (2026-08-20), `day-3`
(2026-08-21), `day-4` (2026-08-22), `day-5` (2026-08-23), `day-6` (2026-08-24),
`day-7` (2026-08-25), `day-8` (2026-08-26), and `day-9` (2026-08-27).
