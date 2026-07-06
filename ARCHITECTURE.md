# AudioTrove Architecture (summary)

This document records the locked, foundational decisions for Phase 0 of AudioTrove and the short-term rationale so future contributors do not accidentally change these cheaply-expensive decisions.

## Core contracts

- `AudioDocument` (canonical object):
	- Fields: `audio: np.ndarray` (mono, float32, range roughly [-1, 1]), `sample_rate: int` (normalized at ingestion, default 16000), `source_path: str`, `duration_seconds: float`, `doc_id: str` (deterministic hash), `metadata: dict` (freeform provenance).

- Block model: two explicit abstract base classes:
	- `AudioFilter`: `filter(doc: AudioDocument) -> bool` — returns `True` to keep the document, `False` to discard. Side-effects: append to `doc.metadata`.
	- `AudioTransformer`: `transform(doc: AudioDocument) -> AudioDocument` — returns a new or modified `AudioDocument`. Transformers do not discard documents.

Rationale: explicit filter vs. transformer removes ambiguity of `None` return values and makes unit tests simpler and safer under multiprocessing.

## Executor model

- `LocalExecutor` (Phase 0): sequential worker loop (future: multiprocessing). Responsibility: iterate reader → apply pipeline blocks → write output → checkpoint processed `doc_id`.
- Checkpointing: SQLite database (`checkpoint_path`) with a `processed(doc_id)` table and WAL-mode recommendation. Checkpointing is a Phase 0 requirement to allow resumable long runs.

Rationale: keep parallelism and checkpointing out of block code; executor manages it so blocks remain stateless and reorderable.

## I/O and dependencies

- Audio I/O: `torchaudio` used for reading and resampling.
- Filesystem abstraction: `fsspec` for `file://`, `s3://`, etc.
- CLI: `click` for command-line entrypoints; `rich` for console rendering.

Rationale: these libraries are standard in ML/ASR environments and keep installs predictable for users who run PyTorch-based stacks.

## Phase 0 implemented components (status)

- `audiotrove/document.py`: `AudioDocument` dataclass. [done]
- `audiotrove/base.py`: `AudioFilter`, `AudioTransformer` ABCs. [done]
- `audiotrove/executor/local.py`: sequential `LocalExecutor` with SQLite checkpointing. [done]
- `audiotrove/io/readers.py`: `LocalAudioReader` (fsspec + torchaudio, resample/downmix). [done]
- `audiotrove/io/writers.py`: `JSONLWriter` (append manifest entries). [done]
- `audiotrove/filters/vad.py`: `SileroVADFilter` (lazy-load via `torch.hub`; energy fallback). [done]
- `audiotrove/filters/snr.py`: `SNRFilter` (VAD-based SNR + fallback). [done]
- `audiotrove/cli/main.py`: CLI skeleton (`curate`, `inspect`). [done (skeleton)]
- Tests updated to current dataclass signatures. [done]

## Phase 0 items intentionally deferred or incomplete (per user scope)

- Committed real-world audio fixtures in `tests/fixtures/` — NOT added (you preferred to decide); tests currently use synthetic audio. [pending: user decision]
- CI / GitHub Actions and publishing steps — explicitly excluded per your instruction. [deferred]
- Full end-to-end benchmarks and heavy runs (Silero model inference + large-corpus runs) — logged in `tasks.txt` for manual execution. [deferred/manual]
- CLI: `curate` is a skeleton (prints placeholder). Integrating a production-ready CLI runner and argument parsing is remaining work. [partial]
- Test run in this environment — not executed due to missing `pytest`/dev deps here. [manual]

## Notes for maintainers

- `doc_id` generation: deterministic hash of `source_path` via SHA256 truncated to 16 hex chars. If you change this, update checkpoint semantics and any persisted DBs.
- Filters may append arbitrary keys to `doc.metadata`; the writer serialises what is present. Consider adding a metadata schema if downstream consumers need strong guarantees.
- Keep blocks stateless: do not add global mutable state to filters or transformers. If a block needs shared data (e.g., a dedup DB), keep it in the executor or make it explicitly injected.

## Phase 0 gate (what to finish before Phase 1)

The roadmap gate is unchanged: Phase 1 should not begin until Phase 0 passes these items. Given the local scope you requested, the actionable remaining items are:

- Add or confirm real audio fixtures and verify numeric thresholds in tests.
- Run `pytest` in a dev environment with `torch/torchaudio` available and fix any failing tests.
- Optionally flesh out the CLI `curate` runner to wire `LocalExecutor`, `LocalAudioReader`, pipeline construction, and `JSONLWriter` end-to-end.

If you want, I can proceed to implement any of the remaining items you approve (for local work). Otherwise follow the play-by-play in `tasks.txt` to run tests and report failures.

---
Last updated: Phase 0 work in this repo. See `tasks.txt` for developer instructions and manual heavy-run notes.
