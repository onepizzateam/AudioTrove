use pyo3::prelude::*;

#[pyfunction]
fn resample(audio: Vec<f32>, source_rate: usize, target_rate: usize) -> Vec<f32> {
    if source_rate == target_rate { return audio; }
    let count = audio.len() * target_rate / source_rate;
    (0..count).map(|i| {
        let position = i as f32 * source_rate as f32 / target_rate as f32;
        let left = position.floor() as usize;
        let right = (left + 1).min(audio.len() - 1);
        audio[left] + (audio[right] - audio[left]) * (position - left as f32)
    }).collect()
}

#[pymodule]
fn audiotrove_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    Ok(())
}
