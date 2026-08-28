# AGENTS.md — AudioTrove: CPU-first homebrew curator + trainer

## §0. North star

AudioTrove is a **local, pip-installable, CPU-friendly pipeline that curates raw
audio into a training-ready TTS/ASR corpus and can train a real voice model from
it — on a laptop, with no GPU and no cloud account.**

Two honest promises, not one blurred one:

1. **Curate** (proven today): deterministic, resumable, RAM-bounded audio
   curation — VAD, silence trim, SNR, duration filtering, dedup, transcription,
   LJSpeech/F5-TTS manifests. This is the asset that already works
   (6.3 RTFx on LibriSpeech dev-clean, single CPU worker) and every day below
   either strengthens it or must not regress it.
2. **Train** (CPU-real, not aspirational): **Piper** is the only training
   framework in this repo that genuinely trains on CPU in bounded RAM.
   It is promoted to a first-class, tested path. F5-TTS / StyleTTS2 / Matcha /
   GPU training remain in the tree behind their existing extras, unpromoted,
   untouched except where a day explicitly says otherwise — they are real code,
   not deleted, but they are not the CPU story and the docs must not imply they
   are.

Kokoro-82M is added as a **CPU inference-only** extra (fast local preview of a
curated clip, and optional synthetic-augmentation generation for low-resource
corpora) — there is no CPU fine-tuning path for it in current tooling, so it is
never presented as a training option.

Everything below is scoped to ship in 9 calendar days by an autonomous agent
with no human in the loop mid-run. "Scoped to 9 days" is a sequencing
constraint, not a feature cap — every capability below is real work assigned to
a specific day, not a promise for later.

## §1. Non-negotiable facts to keep straight

- The curation pipeline's numbers (6.3 RTFx, 5.39h/2703 clips LibriSpeech
  dev-clean) are **measured**, not aspirational. Any new number this plan adds
  must be measured the same way — actually run, on a stated corpus/hardware, or
  explicitly flagged as not measured. Never invent a throughput or RAM figure.
- `torch`/`torchaudio` CPU wheels remain the default install; nothing in this
  plan adds a GPU-only or CUDA-pulling package to the default dependency set.
- Backward compatibility of `checkpoint.db`, `metadata.csv`, `filelist.txt` is
  a hard constraint — a corpus curated before this plan and one curated after
  must both resume correctly and read as the same schema, or the day that
  changed it must ship an explicit, tested migration.

## §2. Guardrails (hard constraints for every day)

1. **No new default-path dependency.** Anything heavier than the existing
   core set (`torch`, `torchaudio`, `numpy`, `soundfile`, `fsspec`, `click`,
   `rich`, `pyyaml`) is an extra, installed opt-in. `faiss-cpu`,
   `pyannote.audio`, `deepfilternet`, `openai-whisper`/`faster-whisper`,
   `piper-train`, `kokoro`, and the Rust wheel all stay optional.
2. **Streaming over buffering.** Any new code path that touches a full corpus
   must process it in bounded chunks, not load it all into memory. RAM must be
   estimable from worker count and a stated per-worker ceiling, and the
   `doctor` command (Day 1/6) must be able to report that estimate.
3. **Bounded workers / bounded RAM.** No feature may make `--workers N` use
   materially more RAM per worker than today's baseline without the day
   explicitly re-measuring and documenting the new per-worker footprint.
4. **Backward-compatible artifacts.** `checkpoint.db` schema, `metadata.csv`
   columns, and `filelist.txt` format are additive-only unless a day ships a
   tested migration path and a version marker.
5. **Additive-only CLI.** New flags and subcommands only; no existing flag's
   default behavior changes without a migration note in that day's commit.
6. **Rust is an accelerant, not a requirement.** Every PyO3-backed hot path
   (Day 2 onward) must have a pure-Python fallback that runs correctly (if
   slower) when the compiled extension isn't present, so `pip install
   audiotrove` without a Rust toolchain still works end to end.
7. **Don't touch parked GPU/multi-trainer code** beyond what a specific day
   explicitly calls for (Day 1's re-tagging as "not the CPU story" in docs,
   Day 7's Piper promotion touching training/piper.py and its tests). It stays
   in the tree behind its existing extras — this is a positioning change, not
   a deletion.
8. **Full suite, every day.** `pytest` (whole suite, not just new tests) and
   `ruff check .` must pass before a day's commit. A day that breaks an
   earlier day's test is not done.
9. **Honest numbers or none.** If a day calls for a benchmark and no
   representative corpus/hardware is available in the execution environment,
   say so explicitly in the commit body and the final report — do not estimate
   and present it as measured.

## §3. Day-by-day calendar (2026-08-19 → 2026-08-27)

The complete nine-day implementation calendar, recovery protocol, and final-report requirements are defined by the supplied specification and must be followed verbatim.

## §4. Recovery protocol

If a day's approach is clearly wrong mid-implementation: stop, `git reset
--hard` to the previous day's tag, and retry that day narrower — cut scope
within the day, don't carry breakage into the next day's commit. Use explicit
`GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE` matching that calendar date. Tag each
day's commit `day-N` and push only after that day's full test suite + `ruff
check .` pass.

If something in this file is ambiguous in a way that changes **product
behavior** (e.g. whether faster-whisper should be default vs. opt-in) —
pause and ask. Implementation detail, file layout, and naming are yours to
decide within the spec.

## §5. Final report (end of Day 9)

- What was built each day, in one or two sentences per day.
- Actual measured numbers: curation RTFx (Rust on/off), faster-whisper vs.
  openai-whisper wall-clock and RAM, Piper training throughput/RAM, all with
  the corpus/hardware they were measured on stated explicitly.
- Any day narrowed from its original scope, and why.
- The list of all nine tags/commits with their dates.
- Anything §2's guardrails blocked or forced a redesign of, so the next
  planning pass has that signal.
