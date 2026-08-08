"""
Local executor with optional multi-worker parallelism.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _worker_process_doc(doc, pipeline):
    """Worker function that processes a single document through the pipeline.

    This runs in a separate process. It applies all filter/transformer blocks
    and returns results back to the main process.

    Supports AudioFanOutTransformer blocks which can emit multiple docs.

    Args:
        doc: AudioDocument to process
        pipeline: list of AudioFilter/AudioTransformer/AudioFanOutTransformer blocks

    Returns:
        Tuple of (docs: list, keep: bool, error: str or None, error_block: str or None)
    """
    from audiotrove.base import AudioFanOutTransformer

    docs = [doc]
    keep = True
    error = None
    error_block = None
    rejected = 0

    try:
        # Windows workers receive a freshly unpickled pipeline. Load the model
        # on that actual instance before processing the first document.
        for block in pipeline:
            if block.__class__.__name__ in ("SileroVADFilter", "VADSegmenter"):
                _ = block.model

        for block in pipeline:
            new_docs = []

            # Filters return bool
            if hasattr(block, "filter") and not isinstance(block, AudioFanOutTransformer):
                filter_error = False
                for d in docs:
                    try:
                        if block.filter(d):
                            new_docs.append(d)
                    except Exception as e:  # noqa: BLE001
                        block_name = getattr(block, "name", block.__class__.__name__)
                        error = f"{block_name}: {e}"
                        error_block = block_name
                        filter_error = True
                        keep = False
                        break
                rejected += len(docs) - len(new_docs)
                docs = new_docs
                if filter_error or not docs:
                    keep = False
                    break

            # Fan-out transformers can emit multiple docs
            elif isinstance(block, AudioFanOutTransformer):
                for d in docs:
                    try:
                        expanded = block.transform(d)
                        new_docs.extend(expanded if isinstance(expanded, list) else [expanded])
                    except Exception as e:  # noqa: BLE001
                        block_name = getattr(block, "name", block.__class__.__name__)
                        error = f"{block_name}: {e}"
                        error_block = block_name
                        keep = False
                        break
                if error:
                    break
                docs = new_docs

            # Regular transformers return one doc
            elif hasattr(block, "transform"):
                new_docs = []
                for d in docs:
                    try:
                        d = block.transform(d)
                        new_docs.append(d)
                    except Exception as e:  # noqa: BLE001
                        block_name = getattr(block, "name", block.__class__.__name__)
                        error = f"{block_name}: {e}"
                        error_block = block_name
                        keep = False
                        break
                if not keep:
                    break
                docs = new_docs
    except Exception as e:  # noqa: BLE001
        error = str(e)
        keep = False

    return (docs, keep, error, error_block, rejected)


def _preload_silero_model() -> None:
    """Preload Silero VAD model in worker process to avoid repeated hub downloads."""
    try:
        # Import locally to avoid importing torch in main process unnecessarily
        from audiotrove.filters.vad import SileroVADFilter

        v = SileroVADFilter()
        _ = v.model  # access to force lazy load
    except Exception:  # noqa: BLE001, S110
        # If loading fails, proceed without raising — worker will fallback to energy VAD
        logger.debug("Silero VAD preload failed; workers will fallback to energy VAD")


def _worker_init(preload_silero: bool) -> None:
    """Initializer run in each worker process."""
    if preload_silero:
        _preload_silero_model()


class LocalExecutor:
    """Local executor with optional multi-worker parallelism.

    When num_workers=1 (default), runs sequentially (backward compatible).
    When num_workers>1, uses ProcessPoolExecutor for parallel document processing.

    The main process always owns the SQLite checkpoint and performs all writes
    to avoid concurrent write issues. Workers only process documents through
    the pipeline and return results.

    When the optional ``audiotrove_core`` Rust extension is installed the
    checkpoint is backed by a lock-free store that supports batched commits;
    otherwise the built-in sqlite3 path is used transparently.
    """

    def __init__(
        self,
        pipeline: list,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 1,
        device: str = "cpu",
    ):
        self.pipeline = pipeline
        self.checkpoint_path = checkpoint_path
        self.num_workers = num_workers
        self.device = device
        self._conn = None
        # Optional Rust-backed checkpoint store (audiotrove_core). When present
        # it provides lock-free, batched SQLite writes that lift the parallel
        # scaling ceiling. When absent we transparently use the sqlite3 path.
        self._store = None
        self._use_rust_checkpoint = False
        if checkpoint_path:
            try:
                from audiotrove_core import CheckpointStore

                self._store = CheckpointStore(checkpoint_path)
                self._use_rust_checkpoint = True
            except ImportError:
                self._use_rust_checkpoint = False

    def _init_db(self):
        if self._use_rust_checkpoint:
            # Rust CheckpointStore initialises its own schema on construction.
            return
        if not self.checkpoint_path:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            doc_id TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self._conn.commit()

    def _is_processed(self, doc_id: str) -> bool:
        if self._use_rust_checkpoint:
            return self._store.is_processed(doc_id) if self._store else False
        if not self._conn:
            return False
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM processed WHERE doc_id = ?", (doc_id,))
        return cur.fetchone() is not None

    def _mark_processed(self, doc_id: str) -> None:
        if self._use_rust_checkpoint:
            if self._store:
                self._store.mark_processed(doc_id)
            return
        if not self._conn:
            return
        cur = self._conn.cursor()
        try:
            cur.execute("INSERT INTO processed (doc_id) VALUES (?)", (doc_id,))
            self._conn.commit()
        except sqlite3.IntegrityError:
            # already recorded
            pass

    def _mark_batch(self, doc_ids: list) -> None:
        """Persist a batch of completed doc_ids in a single transaction.

        This is the key throughput fix for the parallel executor: rather than
        committing once per document, all completed ids in a chunk of futures
        are committed together. Uses the Rust ``mark_batch`` when available and
        an executemany transaction otherwise.
        """
        if not doc_ids:
            return
        if self._use_rust_checkpoint:
            if self._store:
                self._store.mark_batch(list(doc_ids))
            return
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO processed (doc_id) VALUES (?)",
            [(doc_id,) for doc_id in doc_ids],
        )
        self._conn.commit()

    def _move_pipeline_to_device(self, device) -> None:
        """Move every GPU-aware pipeline block onto ``device``."""
        from audiotrove.base import GPUFilter, GPUTransformer

        for block in self.pipeline:
            if isinstance(block, (GPUFilter, GPUTransformer)):
                try:
                    block.to(device)
                except Exception:  # noqa: BLE001 - never let device move crash the run
                    logger.debug("Failed to move %s to %s", block, device, exc_info=True)

    def _maybe_move_to_device(self) -> None:
        """Resolve and apply the configured device to GPU-aware blocks."""
        if not self.device or self.device == "cpu":
            return
        try:
            from audiotrove.gpu.device import get_device

            resolved = get_device(self.device)
        except Exception:  # noqa: BLE001
            logger.debug("Device resolution failed; staying on CPU", exc_info=True)
            return
        self._move_pipeline_to_device(resolved)

    def run(self, reader, writer) -> dict:
        """Run the pipeline over documents produced by `reader`.

        When num_workers=1, runs sequentially (backward compatible).
        When num_workers>1, uses ProcessPoolExecutor for parallel processing.

        The main process always owns SQLite writes to avoid concurrent access issues.

        Returns a stats dict with keys: processed, kept, skipped, errors, errors_by_filter.
        """
        try:
            if self.num_workers == 1:
                return self._run_sequential(reader, writer)
            else:
                return self._run_parallel(reader, writer)
        finally:
            # Always close the database connection when done
            if self._conn:
                self._conn.close()
                self._conn = None

    def _run_sequential(self, reader, writer) -> dict:
        """Sequential processing (num_workers=1). Preserves original behavior exactly for non-fanout cases."""
        from audiotrove.base import AudioFanOutTransformer

        self._init_db()
        self._maybe_move_to_device()
        stats = {"processed": 0, "kept": 0, "skipped": 0, "errors": 0, "errors_by_filter": {}}

        for doc in reader:
            if doc is None:
                stats["skipped"] += 1
                continue

            if self._is_processed(doc.doc_id):
                stats["skipped"] += 1
                continue

            # Process doc(s) through the pipeline
            docs = [doc]
            keep = True
            error = None
            rejected = 0

            # Apply filters/transformers in pipeline order
            for block in self.pipeline:
                new_docs = []

                # Filters return bool
                if hasattr(block, "filter") and not isinstance(block, AudioFanOutTransformer):
                    filter_error = False
                    for d in docs:
                        try:
                            if block.filter(d):
                                new_docs.append(d)
                        except Exception:  # noqa: BLE001
                            block_name = getattr(block, "name", block.__class__.__name__)
                            logger.exception(
                                f"Filter {block_name} raised exception on {d.source_path}"
                            )
                            stats["errors"] += 1
                            if block_name not in stats["errors_by_filter"]:
                                stats["errors_by_filter"][block_name] = 0
                            stats["errors_by_filter"][block_name] += 1
                            filter_error = True
                            keep = False
                            break
                    rejected += len(docs) - len(new_docs)
                    docs = new_docs
                    if filter_error or not docs:
                        keep = False
                        break

                # Fan-out transformers can emit multiple docs
                elif isinstance(block, AudioFanOutTransformer):
                    for d in docs:
                        try:
                            expanded = block.transform(d)
                            new_docs.extend(expanded if isinstance(expanded, list) else [expanded])
                        except Exception:  # noqa: BLE001
                            block_name = getattr(block, "name", block.__class__.__name__)
                            logger.exception(
                                f"Fan-out Transformer {block_name} raised exception on {d.source_path}"
                            )
                            stats["errors"] += 1
                            if block_name not in stats["errors_by_filter"]:
                                stats["errors_by_filter"][block_name] = 0
                            stats["errors_by_filter"][block_name] += 1
                            error = True
                            break
                    if error:
                        break
                    docs = new_docs

                # Regular transformers return one doc
                elif hasattr(block, "transform"):
                    for d in docs:
                        try:
                            d = block.transform(d)
                            new_docs.append(d)
                        except Exception:  # noqa: BLE001
                            block_name = getattr(block, "name", block.__class__.__name__)
                            logger.exception(
                                f"Transformer {block_name} raised exception on {d.source_path}"
                            )
                            stats["errors"] += 1
                            if block_name not in stats["errors_by_filter"]:
                                stats["errors_by_filter"][block_name] = 0
                            stats["errors_by_filter"][block_name] += 1
                            keep = False
                            break
                    if not keep:
                        break
                    docs = new_docs

            stats["processed"] += 1
            stats["skipped"] += rejected
            if keep and docs and not error:
                for d in docs:
                    writer.write(d)
                    self._mark_processed(d.doc_id)
                stats["kept"] += len(docs)
            elif not rejected:
                stats["skipped"] += len(docs) or 1
            if not error:
                # The reader only sees source documents; checkpoint the source
                # as well as accepted fan-out children so a resumed run skips it.
                self._mark_processed(doc.doc_id)

        return stats

    def _run_parallel(self, reader, writer) -> dict:
        """Parallel processing using ProcessPoolExecutor.

        Main process owns the checkpoint. Workers process documents through the
        pipeline and return results. Completed doc_ids are committed in batches
        (one transaction per BATCH futures) which is the key throughput fix for
        parallel scaling.
        """
        self._init_db()
        self._maybe_move_to_device()
        stats = {"processed": 0, "kept": 0, "skipped": 0, "errors": 0, "errors_by_filter": {}}

        # Windows process workers cannot share the loaded torch model and four
        # independent Silero instances can terminate the pool under memory
        # pressure. Threads share the model safely for this CPU-bound pipeline;
        # retain processes on platforms where fork/spawn is stable.
        if os.name == "nt":
            for block in self.pipeline:
                if block.__class__.__name__ in ("SileroVADFilter", "VADSegmenter"):
                    _ = block.model
            executor_type = ThreadPoolExecutor
        else:
            executor_type = ProcessPoolExecutor

        # Collect all unprocessed documents first
        pending_docs = []
        for doc in reader:
            if doc is None:
                stats["skipped"] += 1
                continue

            if self._is_processed(doc.doc_id):
                stats["skipped"] += 1
                continue

            pending_docs.append(doc)

        if not pending_docs:
            return stats

        # Determine whether to preload Silero in workers (if pipeline contains VAD blocks)
        preload_silero = any(
            getattr(block, "__class__", None).__name__ in ("SileroVADFilter", "VADSegmenter")
            or hasattr(block, "model")
            for block in self.pipeline
        )

        # Process documents in parallel
        executor_kwargs = {"max_workers": self.num_workers}
        if executor_type is ProcessPoolExecutor:
            executor_kwargs.update(initializer=_worker_init, initargs=(preload_silero,))

        BATCH = 64
        with executor_type(**executor_kwargs) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(_worker_process_doc, doc, self.pipeline): doc
                for doc in pending_docs
            }

            # Drain futures in chunks and commit the checkpoint once per chunk
            # rather than once per document (single SQLite transaction per BATCH).
            futures = list(future_to_doc.keys())
            for i in range(0, len(futures), BATCH):
                chunk = futures[i : i + BATCH]
                completed_ids = []
                for future in as_completed(chunk):
                    try:
                        docs, keep, error, error_block, rejected = future.result()

                        stats["processed"] += 1
                        stats["skipped"] += rejected

                        if error:
                            for doc in docs:
                                logger.exception(f"Error processing {doc.source_path}: {error}")
                            stats["errors"] += 1
                            if error_block and error_block not in stats["errors_by_filter"]:
                                stats["errors_by_filter"][error_block] = 0
                            if error_block:
                                stats["errors_by_filter"][error_block] += 1

                        if keep and docs and not error:
                            for doc in docs:
                                writer.write(doc)
                                completed_ids.append(doc.doc_id)
                            stats["kept"] += len(docs)
                        elif not rejected:
                            stats["skipped"] += len(docs) or 1

                        if not error:
                            # Fan-out children are not reader inputs. Persist the
                            # original source ID so a resume skips the whole input.
                            completed_ids.append(future_to_doc[future].doc_id)

                    except Exception:  # noqa: BLE001
                        logger.exception("Worker task failed")
                        stats["errors"] += 1

                # One checkpoint transaction for the whole chunk.
                self._mark_batch(completed_ids)

        return stats
