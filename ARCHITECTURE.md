# AudioTrove Architecture (summary)

This document records the locked, foundational decisions for Phase 0 of AudioTrove and the short-term rationale so future contributors do not accidentally change these cheaply-expensive decisions.

## Core contracts

- `AudioDocument` (canonical object):
	- Fields: `audio: np.ndarray` (mono, float32, range roughly [-1, 1]), `sample_rate: int` (normalized at ingestion, default 16000), `source_path: str`, `duration_seconds: float`, `doc_id: str` (deterministic hash), `metadata: dict` (freeform provenance).

- Block model: two explicit abstract base classes:
	- `AudioFilter`: `filter(doc: AudioDocument) -> bool` — returns `True` to keep the document, `False` to discard. Side-effects: append to `doc.metadata`.
	- `AudioTransformer`: `transform(doc: AudioDocument) -> AudioDocument` — returns a new or modified `AudioDocument`. Transformers do not discard documents.

Rationale: explicit filter vs. transformer removes ambiguity of `None` return values and makes unit tests simpler and safer under multiprocessing.

- Block model (Phase 0.5, added for segmentation): one additional block type for fan-out operations:
	- `AudioFanOutTransformer`: `transform(doc: AudioDocument) -> list[AudioDocument]` — returns zero, one, or many documents. Enables segmentation and expansion operations where one input doc produces multiple outputs.

**Why fan-out transformers (not in original Phase 0):** The original block model (1:1 transform/filter) works for simple pipeline chains but doesn't support "right-splitting" operations like segmentation, where a long file should become multiple shorter segments for independent processing. Compared to alternatives:
  - Option A (keep 1:1, segment externally): requires special reader-side handling, breaks modularity of the pipeline.
  - Option B (extend executor to detect fan-out): requires hard-coding list expansion logic in the executor instead of keeping it in the block itself.
  - Option C (add AudioFanOutTransformer): keeps the block interface extensible and executor logic general. Blocks declare their cardinality (1:1 vs. 1:many) in their type.

Segments from a fan-out transformer must have deterministic `doc_id` derived from parent doc_id + segment metadata (start/end times) so re-running is idempotent and checkpoint-safe. The executor still processes each segment independently through remaining pipeline blocks.

## Executor model

- `LocalExecutor` (Phase 0): sequential worker loop, parallelizable via `ProcessPoolExecutor` (num_workers param). Responsibility: iterate reader → apply pipeline blocks (handling 1:1 and 1:many cardinality) → write output → checkpoint processed `doc_id`.
- Checkpointing: SQLite database (`checkpoint_path`) with a `processed(doc_id)` table and WAL-mode recommendation. Checkpointing is a Phase 0 requirement to allow resumable long runs.

Rationale: keep parallelism and checkpointing out of block code; executor manages it so blocks remain stateless and reorderable. Fan-out transformers emit results to the executor, which handles downstream pipeline steps on all expanded docs.

## I/O and dependencies

- Audio I/O: `torchaudio` used for reading and resampling.
- Filesystem abstraction: `fsspec` for `file://`, `s3://`, etc.
- CLI: `click` for command-line entrypoints; `rich` for console rendering.

Benchmarks: heavy end-to-end benchmarks are stored under `benchmarks/e2e_results.json`. For large parallel runs, pre-cache the Silero model in the local torch hub cache (e.g. run the included `scripts/precache_silero.py` or otherwise populate `~/.cache/torch/hub/snakers4_silero-vad_master`) before launching many worker processes. This avoids GitHub rate-limiting and repeated network load when worker processes attempt to call `torch.hub.load` concurrently.

### VAD model loading

- The Silero VAD model is lazy-loaded via a helper that prefers a local torch hub repository at `~/.cache/torch/hub/snakers4_silero-vad_master`.
- When the local cache exists, the loader invokes:

```
torch.hub.load(str(local_path), "silero_vad", source="local", trust_repo=True)
```

to avoid network requests from worker processes. The hub loader may return a `(model, utils)` tuple where `utils` can itself be a tuple of callables rather than an attribute namespace. The helper therefore wraps `utils` into a `types.SimpleNamespace` mapping function objects by their `__name__` so calling code can access `utils.get_speech_timestamps` as an attribute.

- Filters and segmenters exclude the `_model` and `_utils` fields from `__getstate__` so that pickling for `ProcessPoolExecutor` does not serialize large torch artifacts. Worker processes will lazily reload the model on first access.
- If Silero cannot be loaded, or if inference returns an empty timestamp list on synthetic/test signals, the code falls back to an energy-based VAD. The chosen backend is recorded per-file in `doc.metadata['vad_backend']`.

### Parallel execution & checkpointing

- `LocalExecutor` uses `concurrent.futures.ProcessPoolExecutor` when `num_workers > 1` to fan document processing out across worker processes.
- The main process is the sole writer of the JSONL manifest and the SQLite checkpoint database; worker processes run filters and transformers and return processed documents (or lists of documents for fan-out transformers) back to the main process for final writing.
- Checkpointing is implemented with a simple SQLite table recording processed `doc_id` values. The executor writes to the checkpoint DB in the main process to avoid concurrent-writer issues. Use WAL mode for robust concurrent reads during long runs.
- Filters and transformers must be picklable; heavy runtime assets (e.g., torch models) should be lazy-loaded in the worker process and excluded from pickling via `__getstate__`/`__setstate__` semantics.

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
