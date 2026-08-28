use pyo3::prelude::*;

mod checkpoint;
mod decode;
mod glob;

/// PyO3 module root. Registers the checkpoint store and the glob/decode helpers.
#[pymodule]
fn audiotrove_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<checkpoint::CheckpointStore>()?;
    m.add_function(wrap_pyfunction!(glob::discover_files, m)?)?;
    m.add_function(wrap_pyfunction!(decode::decode_audio_batch, m)?)?;
    Ok(())
}
