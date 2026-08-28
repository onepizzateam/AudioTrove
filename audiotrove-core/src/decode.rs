use pyo3::prelude::*;
use rayon::prelude::*;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::path::Path;

/// Decode a batch of audio files in parallel.
///
/// Returns one tuple per input path: `(samples, actual_sample_rate, path)`.
/// Samples are mono f32 in the range [-1, 1]. When `target_sr` is non-zero and
/// differs from a file's native rate, the samples are resampled with simple
/// linear interpolation. Files that fail to decode yield an empty sample vector
/// with a sample rate of 0 so the caller can detect and skip them.
#[pyfunction]
pub fn decode_audio_batch(
    paths: Vec<String>,
    target_sr: u32,
) -> PyResult<Vec<(Vec<f32>, u32, String)>> {
    let results: Vec<(Vec<f32>, u32, String)> = paths
        .par_iter()
        .map(|p| match decode_one(p, target_sr) {
            Ok((samples, sr)) => (samples, sr, p.clone()),
            Err(_) => (Vec::new(), 0u32, p.clone()),
        })
        .collect();
    Ok(results)
}

/// Decode a single file to mono f32, optionally resampling to `target_sr`.
fn decode_one(path: &str, target_sr: u32) -> Result<(Vec<f32>, u32), String> {
    let file = File::open(Path::new(path)).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let track_id = track.id;
    let native_sr = track.codec_params.sample_rate.unwrap_or(0);
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut mono: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // end of stream
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Convert whatever the sample format is into an f32 sample buffer.
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sample_buf =
            symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples();

        // Downmix interleaved frames to mono by averaging channels.
        for frame in interleaved.chunks(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
    }

    if target_sr != 0 && native_sr != 0 && target_sr != native_sr {
        mono = resample_linear(&mono, native_sr, target_sr);
        Ok((mono, target_sr))
    } else {
        Ok((mono, native_sr))
    }
}

/// Simple linear-interpolation resampler.
fn resample_linear(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if input.is_empty() || from_sr == 0 || to_sr == 0 || from_sr == to_sr {
        return input.to_vec();
    }
    let ratio = to_sr as f64 / from_sr as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}
