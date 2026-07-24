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
                for d in docs:
                    try:
                        keep = block.filter(d)
                        if keep:
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

    return (docs, keep, error, error_block)


def _preload_silero_model() -> None:
    """Preload Silero VAD model in worker process to avoid repeated hub downloads."""
    try:
        # Import locally to avoid importing torch in main process unnecessarily
        from audiotrove.filters.vad import SileroVADFilter

        v = SileroVADFilter()
        _ = v.model  # access to force lazy load
    except Exception:  # noqa: BLE001
        # If loading fails, proceed without raising — worker will fallback to energy VAD
        pass


def _worker_init(preload_silero: bool) -> None:
    """Initializer run in each worker process."""
    if preload_silero:
        _preload_silero_model()


class LocalExecutor:
    """Local executor with optional multi-worker parallelism.

    When num_workers=1 (default), runs sequentially (backward compatible).
    When num_workers>1, uses ProcessPoolExecutor for parallel document processing.

    The main process always owns the SQLite connection and performs all writes
    to avoid concurrent write issues. Workers only process documents through
    the pipeline and return results.
    """

    def __init__(self, pipeline: list, checkpoint_path: Optional[str] = None, num_workers: int = 1):
        self.pipeline = pipeline
        self.checkpoint_path = checkpoint_path
        self.num_workers = num_workers
        self._conn = None

    def _init_db(self):
        if not self.checkpoint_path:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            doc_id TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self._conn.commit()

    def _is_processed(self, doc_id: str) -> bool:
        if not self._conn:
            return False
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM processed WHERE doc_id = ?", (doc_id,))
        return cur.fetchone() is not None

    def _mark_processed(self, doc_id: str) -> None:
        if not self._conn:
            return
        cur = self._conn.cursor()
        try:
            cur.execute("INSERT INTO processed (doc_id) VALUES (?)", (doc_id,))
            self._conn.commit()
        except sqlite3.IntegrityError:
            # already recorded
            pass

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

            # Apply filters/transformers in pipeline order
            for block in self.pipeline:
                new_docs = []

                # Filters return bool
                if hasattr(block, "filter") and not isinstance(block, AudioFanOutTransformer):
                    for d in docs:
                        try:
                            keep = block.filter(d)
                            if keep:
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
                            keep = False
                            break
                    if not keep:
                        break
                    docs = new_docs

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
            if keep and docs and not error:
                for d in docs:
                    writer.write(d)
                    self._mark_processed(d.doc_id)
                stats["kept"] += len(docs)
            else:
                stats["skipped"] += len(docs)
                # Mark original doc as processed even if expanded/filtered to nothing
                self._mark_processed(doc.doc_id)

        return stats

    def _run_parallel(self, reader, writer) -> dict:
        """Parallel processing using ProcessPoolExecutor.

        Main process owns SQLite connection. Workers process documents through
        the pipeline and return results. Main process writes and checkpoints.
        """
        self._init_db()
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
        with executor_type(**executor_kwargs) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(_worker_process_doc, doc, self.pipeline): doc
                for doc in pending_docs
            }

            # Process results as they complete (main process handles writes)
            for future in as_completed(future_to_doc):
                try:
                    docs, keep, error, error_block = future.result()

                    stats["processed"] += 1

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
                            self._mark_processed(doc.doc_id)
                        stats["kept"] += len(docs)
                    else:
                        stats["skipped"] += len(docs)
                        # Mark original doc as processed
                        for doc in docs:
                            self._mark_processed(doc.doc_id)

                except Exception:  # noqa: BLE001
                    logger.exception("Worker task failed")
                    stats["errors"] += 1

        return stats
