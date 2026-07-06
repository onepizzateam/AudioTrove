"""
Local executor.
"""
import sqlite3
from pathlib import Path
from typing import Optional



class LocalExecutor:
    """Sequential local executor with simple SQLite checkpointing.

    This intentionally runs sequentially for Phase 0. It records processed
    `doc_id` values in a SQLite DB so runs can be resumed.
    """

    def __init__(self, pipeline: list, num_workers: int = 1, checkpoint_path: Optional[str] = None):
        self.pipeline = pipeline
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path
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

        Returns a small stats dict.
        """
        self._init_db()
        stats = {"processed": 0, "kept": 0, "skipped": 0}

        for doc in reader:
            if doc is None:
                stats["skipped"] += 1
                continue

            if self._is_processed(doc.doc_id):
                stats["skipped"] += 1
                continue

            keep = True
            # Apply filters/transformers in pipeline order
            for block in self.pipeline:
                # Filters return bool
                if hasattr(block, "filter"):
                    try:
                        keep = block.filter(doc)
                    except Exception:
                        keep = False
                    if not keep:
                        break
                elif hasattr(block, "transform"):
                    doc = block.transform(doc)

            stats["processed"] += 1
            if keep:
                writer.write(doc)
                stats["kept"] += 1
            else:
                stats["skipped"] += 1

            self._mark_processed(doc.doc_id)

        return stats
