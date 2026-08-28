use pyo3::prelude::*;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
             WindowFunction};
use rustfft::{num_complex::Complex, FftPlanner};

#[pyfunction]
fn glob_paths(pattern: String) -> Vec<String> {
    glob::glob(&pattern)
        .into_iter()
        .flatten()
        .filter_map(|path| path.ok().map(|value| value.to_string_lossy().into_owned()))
        .collect()
}

#[pyfunction]
fn decode_wav(path: String) -> PyResult<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| {
                let scale = ((1i64 << spec.bits_per_sample.saturating_sub(1)) - 1) as f32;
                value as f32 / scale
            }))
            .collect::<Result<Vec<_>, _>>(),
    }
    .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
    let channels = spec.channels as usize;
    let mono = if channels <= 1 {
        samples
    } else {
        samples.chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate as usize))
}

#[pyfunction]
fn resample(audio: Vec<f32>, source_rate: usize, target_rate: usize) -> Vec<f32> {
    if source_rate == target_rate || audio.is_empty() { return audio; }
    let linear_fallback = || {
        let count = (audio.len() * target_rate / source_rate).max(1);
        (0..count).map(|i| {
            let position = i as f32 * source_rate as f32 / target_rate as f32;
            let left = position.floor() as usize;
            let right = (left + 1).min(audio.len() - 1);
            audio[left] + (audio[right] - audio[left]) * (position - left as f32)
        }).collect::<Vec<_>>()
    };
    if audio.len() < 64 { return linear_fallback(); }
    let params = SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = target_rate as f64 / source_rate as f64;
    let mut resampler = match SincFixedIn::<f32>::new(ratio, 2.0, params, audio.len(), 1) {
        Ok(value) => value,
        Err(_) => return linear_fallback(),
    };
    resampler.process(&[audio.clone()], None)
        .map(|channels| channels.into_iter().next().unwrap_or_default())
        .map(|output| if output.is_empty() { linear_fallback() } else { output })
        .unwrap_or_else(|_| linear_fallback())
}

#[pyfunction]
fn fingerprint(audio: Vec<f32>) -> u64 {
    let mut input = vec![Complex::new(0.0f32, 0.0); 2048];
    for (slot, sample) in input.iter_mut().zip(audio.iter().take(2048)) {
        slot.re = *sample;
    }
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_forward(2048).process(&mut input);
    let spectrum: Vec<f32> = input[1..1024].iter().map(|value| value.norm()).collect();
    let mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
    let mut result = 0u64;
    for band in 0..64 {
        let start = band * spectrum.len() / 64;
        let end = (band + 1) * spectrum.len() / 64;
        let band_mean = spectrum[start..end].iter().sum::<f32>() / (end - start) as f32;
        if band_mean > mean { result |= 1u64 << band; }
    }
    result
}

#[pymodule]
fn audiotrove_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(glob_paths, m)?)?;
    m.add_function(wrap_pyfunction!(decode_wav, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint, m)?)?;
    Ok(())
}
