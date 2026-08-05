use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::Connection;
use std::sync::Mutex;

/// Thread-safe SQLite-backed checkpoint store exposed to Python.
///
/// Wraps a single connection behind a `Mutex` so concurrent writes from
/// multiple threads are serialised safely. The `mark_batch` method commits an
/// entire batch of completed doc ids in one transaction, which is the key
/// throughput win over per-document commits.
#[pyclass]
pub struct CheckpointStore {
    conn: Mutex<Connection>,
}

#[pymethods]
impl CheckpointStore {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let conn = Connection::open(path).map_err(to_py_err)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS processed (
                doc_id TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )
        .map_err(to_py_err)?;
        Ok(CheckpointStore {
            conn: Mutex::new(conn),
        })
    }

    /// Return true when `doc_id` has already been recorded.
    pub fn is_processed(&self, doc_id: &str) -> PyResult<bool> {
        let conn = self.conn.lock().map_err(lock_err)?;
        let mut stmt = conn
            .prepare("SELECT 1 FROM processed WHERE doc_id = ?1")
            .map_err(to_py_err)?;
        let exists = stmt.exists([doc_id]).map_err(to_py_err)?;
        Ok(exists)
    }

    /// Record a single `doc_id` (idempotent).
    pub fn mark_processed(&self, doc_id: &str) -> PyResult<()> {
        let conn = self.conn.lock().map_err(lock_err)?;
        conn.execute(
            "INSERT OR IGNORE INTO processed (doc_id) VALUES (?1)",
            [doc_id],
        )
        .map_err(to_py_err)?;
        Ok(())
    }

    /// Record every id in `doc_ids` in a single transaction.
    pub fn mark_batch(&self, doc_ids: Vec<String>) -> PyResult<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().map_err(lock_err)?;
        let tx = conn.transaction().map_err(to_py_err)?;
        {
            let mut stmt = tx
                .prepare("INSERT OR IGNORE INTO processed (doc_id) VALUES (?1)")
                .map_err(to_py_err)?;
            for doc_id in &doc_ids {
                stmt.execute([doc_id]).map_err(to_py_err)?;
            }
        }
        tx.commit().map_err(to_py_err)?;
        Ok(())
    }
}

fn to_py_err(e: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(format!("sqlite error: {e}"))
}

fn lock_err<T>(_e: T) -> PyErr {
    PyRuntimeError::new_err("checkpoint store lock poisoned")
}
