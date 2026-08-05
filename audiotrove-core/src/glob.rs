use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Fast recursive file discovery.
///
/// `patterns` is the same list of directory roots / glob-style patterns that
/// `LocalAudioReader` builds. For each pattern we derive a walk root (the
/// portion before the first glob metacharacter) and recursively walk it,
/// keeping files whose extension is in `extensions`. Results are returned as
/// sorted absolute paths for determinism.
#[pyfunction]
pub fn discover_files(patterns: Vec<String>, extensions: Vec<String>) -> PyResult<Vec<String>> {
    let ext_set: HashSet<String> = extensions
        .into_iter()
        .map(|e| e.trim_start_matches('.').to_lowercase())
        .collect();

    // Derive unique walk roots from the incoming patterns.
    let mut roots: Vec<PathBuf> = patterns.iter().map(|p| walk_root(p)).collect();
    roots.sort();
    roots.dedup();

    // Walk each root in parallel, then flatten and filter by extension.
    let mut found: Vec<String> = roots
        .par_iter()
        .flat_map_iter(|root| {
            WalkDir::new(root)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_file())
                .filter_map(|entry| {
                    let path = entry.path();
                    if matches_ext(path, &ext_set) {
                        Some(absolute(path))
                    } else {
                        None
                    }
                })
                .collect::<Vec<String>>()
        })
        .collect();

    found.sort();
    found.dedup();
    Ok(found)
}

/// Return the directory portion of `pattern` up to the first glob metacharacter.
fn walk_root(pattern: &str) -> PathBuf {
    let cut = pattern
        .find(|c| c == '*' || c == '?' || c == '[')
        .unwrap_or(pattern.len());
    let head = &pattern[..cut];
    let candidate = Path::new(head);
    // If the head ends mid-segment (e.g. "data/aud*"), back off to its parent.
    if head.ends_with('/') || head.ends_with('\\') || candidate.is_dir() || cut == pattern.len() {
        if candidate.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            candidate.to_path_buf()
        }
    } else {
        candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

/// True when `path`'s extension (lowercased) is in `ext_set`, or the set is
/// empty (match everything).
fn matches_ext(path: &Path, ext_set: &HashSet<String>) -> bool {
    if ext_set.is_empty() {
        return true;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| ext_set.contains(&e.to_lowercase()))
        .unwrap_or(false)
}

/// Best-effort absolute path as a String; falls back to the lossy display form.
fn absolute(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}
