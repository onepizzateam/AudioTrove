# From Raw Audiobook to TTS-Ready Dataset in One Command

## The scenario

You found a single-speaker audiobook recording online. It is free, public-domain
material with a home-recording character: variable microphone quality, room
acoustics, and long chapter files with leading and trailing silence. You want to
prepare a small, reproducible slice for a voice-cloning or TTS experiment.

The old way is a chain of ffmpeg commands to split chapters, a denoiser, manual
clip-boundary edits, a hand-written manifest, and a run that must be restarted
after a crash.

The AudioTrove way is one resumable command:

```bash
pip install audiotrove
audiotrove --verbose curate ./sherlock ./curated --tts --segment --extensions mp3 --workers 4
```

## Source material

- **Recording:** *The Adventures of Sherlock Holmes*, read by Mark F. Smith (LibriVox).
- **Archive item:** `adventures_sherlockholmes_1007_librivox`.
- **Input used for this bounded demo:** one 10-second speech excerpt taken from
  the downloaded chapter MP3 at the 43:30 mark. The original chapter is retained
  locally as `adventuresherlockholmes_01_doyle.mp3.source`.
- **Why this:** it is a real public-domain home recording rather than synthetic
  test audio. The excerpt is short enough to run predictably while exercising
  discovery, real Silero VAD, silence trimming, SNR scoring, duration filtering,
  WAV export, and checkpointing.

The excerpt is passed through AudioTrove's `mp3` extension path. The local
download environment could not complete a second chapter within the benchmark
window, so the committed numbers below describe this one-file demonstration,
not the full eleven-hour audiobook.

## What AudioTrove did

| Metric | Value |
|---|---|
| Input | 1 MP3-path excerpt (10 seconds) |
| Output clips | 1 WAV segment |
| Curated duration | 0.00h (9.508s) |
| Avg clip duration | 9.5s |
| Wall time (4 workers) | 48.57s |
| Resume time (checkpoint intact) | 45.29s |
| First-run filter result | 1 kept, 0 filtered |
| Output formats | LJSpeech `metadata.csv` + F5-TTS `filelist.txt` |

Wall time is measured by the driver around the complete CLI invocation. The
resume timing includes rediscovering the input and consulting SQLite; the
already-processed document is not exported again. The CLI summary reports the
checkpointed document as filtered on the resume pass (`Kept: 0`, `Filtered: 1`),
while the original pass reports the actual curation result.

## What AudioTrove did to the clip

With `--segment`, the TTS pipeline uses real Silero VAD to fan out speech
regions into candidate clips. It then applies per-segment VAD and silence
trimming, scores the remaining audio with the SNR filter, checks the final
duration bucket, and exports accepted documents. The excerpt had speech
detected, passed the 20 dB SNR threshold, and finished at 9.508 seconds, inside
the requested 2–15 second duration range.

The input recording does not include a transcript. Consequently the generated
manifest has an empty transcription field; AudioTrove does not invent text or
run speech recognition as a side effect of curation.

## Crash recovery

The exact command was run a second time with the same output directory and its
SQLite checkpoint intact:

```
First run:  48.57s for 1 kept clip
Resume run: 45.29s; the processed document was skipped by checkpoint state
```

The checkpoint lives at `sherlock-curated/checkpoint.db`. It makes the run
restartable without duplicating the existing WAV or manifest row.

## Output files

```text
sherlock-curated/
├── chapter01_excerpt10.wav
├── checkpoint.db
├── filelist.txt
└── metadata.csv
```

The measured F5-TTS row is:

```text
sherlock-curated\chapter01_excerpt10.wav	9.5080	
```

The measured LJSpeech row is:

```text
chapter01_excerpt10.wav|speaker_00|
```

The final field is empty because the source provides no transcript. The WAV is
trimmed audio written by the exporter; the source file is never overwritten.

## Plugging into F5-TTS

The output `filelist.txt` is already tab-separated with path, duration, and text:

```python
for line in open("sherlock-curated/filelist.txt", encoding="utf-8"):
    wav_path, duration, text = line.rstrip("\n").split("\t")
    # Add the transcript produced by your ASR or annotation step before training.
```

LJSpeech-compatible trainers can consume `metadata.csv` after the same transcript
annotation step:

```python
for line in open("sherlock-curated/metadata.csv", encoding="utf-8"):
    wav_name, speaker, text = line.rstrip("\n").split("|", 2)
```

## Reproduce it

Download the public-domain recording and select a bounded speech excerpt if you
want a quick local demonstration:

```bash
pip install internetarchive
ia download adventures_sherlockholmes_1007_librivox --glob="*.mp3" --destdir ./sherlock --no-directories

cd audiotrove
audiotrove --verbose curate ./sherlock ./curated \
  --tts --segment --extensions mp3 --workers 4 \
  --tts-min-duration 2 --tts-max-duration 15 --tts-snr-min 20
```

For a full chapter, expect substantially more wall time than this bounded
excerpt because the VAD stage analyzes the complete decoded waveform. No GPU is
required; the benchmark used the local CPU installation of AudioTrove.

## Files captured for this benchmark

The complete first-pass CLI output, including its measured wall time, is in
[`sherlock_run.log`](../sherlock_run.log). The checkpointed second-pass output
and timing are in [`sherlock_resume.log`](../sherlock_resume.log).
