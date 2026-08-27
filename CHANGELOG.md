# Changelog

## 0.1.2 — 2026-08-27

AudioTrove is now explicitly CPU-first: deterministic curation remains the
proven path, Piper is the CPU training path, and Kokoro is inference-only.
This release adds the doctor command, optional Rust resampling with a Python
fallback, checkpoint versioning/WAL journaling, faster-whisper selection,
lightweight fingerprint QC, and the pipeline CLI.

### Benchmark status

No fresh representative LibriSpeech, Whisper, or Piper benchmark was possible
in this Windows environment because the required corpora and optional model
packages were unavailable. Consequently this release makes no new throughput,
speedup, or RAM claims; the existing measured curation claim remains 6.3 RTFx
on LibriSpeech dev-clean (5.39 hours, 2,703 clips, one worker).

The Day 1–9 implementation commits are tagged `day-1` through `day-9`.
