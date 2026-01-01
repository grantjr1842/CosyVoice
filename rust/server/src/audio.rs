//! Native audio processing module for CosyVoice.
//!
//! Provides mel spectrogram computation and audio I/O without Python dependencies.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use hound::WavReader;
use ndarray::{Array1, Array2};
use realfft::RealFftPlanner;
use std::f64::consts::PI;
use std::path::Path;

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub n_fft: usize,
    pub num_mels: usize,
    pub sampling_rate: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub fmin: f64,
    pub fmax: Option<f64>,
    pub center: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        // CosyVoice3 defaults from cosyvoice3.yaml
        Self {
            n_fft: 1920,
            num_mels: 80,
            sampling_rate: 24000,
            hop_size: 480,
            win_size: 1920,
            fmin: 0.0,
            fmax: None, // Use sampling_rate / 2
            center: false,
        }
    }
}

impl MelConfig {
    /// Create config for CosyVoice3 (24kHz, 80 mels)
    pub fn cosyvoice3() -> Self {
        Self::default()
    }

    /// Create config for inference (1024 n_fft variant)
    pub fn inference() -> Self {
        Self {
            n_fft: 1024,
            num_mels: 80,
            sampling_rate: 24000,
            hop_size: 256,
            win_size: 1024,
            fmin: 0.0,
            fmax: Some(8000.0),
            center: false,
        }
    }
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(freq: f64) -> f64 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix.
///
/// Returns a (num_mels, n_fft/2 + 1) matrix.
pub fn create_mel_filterbank(config: &MelConfig) -> Array2<f32> {
    let fmax = config.fmax.unwrap_or(config.sampling_rate as f64 / 2.0);
    let n_fft_bins = config.n_fft / 2 + 1;

    let mel_min = hz_to_mel(config.fmin);
    let mel_max = hz_to_mel(fmax);

    // Create mel points
    let mel_points: Vec<f64> = (0..config.num_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (config.num_mels + 1) as f64)
        .collect();

    // Convert mel points to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| (config.n_fft as f64 + 1.0) * hz / config.sampling_rate as f64)
        .collect();

    // Create filterbank
    let mut filterbank = Array2::<f32>::zeros((config.num_mels, n_fft_bins));

    for m in 0..config.num_mels {
        let f_m_minus = bin_points[m];
        let f_m = bin_points[m + 1];
        let f_m_plus = bin_points[m + 2];

        for k in 0..n_fft_bins {
            let k_f = k as f64;
            if k_f >= f_m_minus && k_f <= f_m {
                filterbank[[m, k]] = ((k_f - f_m_minus) / (f_m - f_m_minus)) as f32;
            } else if k_f >= f_m && k_f <= f_m_plus {
                filterbank[[m, k]] = ((f_m_plus - k_f) / (f_m_plus - f_m)) as f32;
            }
        }
    }

    filterbank
}

/// Create a Hann window of given size.
fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let x = (PI * i as f64 / size as f64).sin();
            (x * x) as f32
        })
        .collect()
}

/// Dynamic range compression (log compression).
fn dynamic_range_compression(x: f32, clip_val: f32, c: f32) -> f32 {
    (x.max(clip_val) * c).ln()
}

/// Load audio from a WAV file.
///
/// Returns normalized f32 samples in range [-1.0, 1.0].
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let reader = WavReader::open(path.as_ref())?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // If stereo, convert to mono by averaging channels
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Compute Short-Time Fourier Transform (STFT).
///
/// Returns magnitude spectrogram as (n_frames, n_fft/2 + 1) array.
pub fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
    center: bool,
) -> Result<Array2<f32>> {
    let window = create_hann_window(win_size);

    // Pad audio if center=true or if needed
    let padded_audio: Vec<f32> = if center {
        let pad_len = n_fft / 2;
        let mut padded = vec![0.0f32; pad_len];
        padded.extend_from_slice(audio);
        padded.extend(vec![0.0f32; pad_len]);
        padded
    } else {
        // Reflect padding for non-centered STFT (like PyTorch)
        let pad_len = (n_fft - hop_size) / 2;
        let mut padded = Vec::with_capacity(audio.len() + 2 * pad_len);

        // Reflect padding at start
        for i in 0..pad_len {
            let idx = (pad_len - i).min(audio.len() - 1);
            padded.push(audio[idx]);
        }
        padded.extend_from_slice(audio);
        // Reflect padding at end
        for i in 0..pad_len {
            let idx = (audio.len() - 2 - i).max(0);
            padded.push(audio[idx]);
        }
        padded
    };

    let n_frames = (padded_audio.len() - n_fft) / hop_size + 1;
    let n_bins = n_fft / 2 + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut magnitudes = Array2::<f32>::zeros((n_frames, n_bins));
    let mut scratch = vec![Default::default(); fft.get_scratch_len()];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;

        // Apply window
        let mut windowed: Vec<f32> = (0..n_fft)
            .map(|i| {
                let audio_val = if start + i < padded_audio.len() {
                    padded_audio[start + i]
                } else {
                    0.0
                };
                let win_val = if i < win_size { window[i] } else { 0.0 };
                audio_val * win_val
            })
            .collect();

        // Perform FFT
        let mut spectrum = vec![Default::default(); n_bins];
        fft.process_with_scratch(&mut windowed, &mut spectrum, &mut scratch)
            .map_err(|e| anyhow!("FFT error: {:?}", e))?;

        // Compute magnitude
        for (bin_idx, c) in spectrum.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            magnitudes[[frame_idx, bin_idx]] = mag;
        }
    }

    Ok(magnitudes)
}

/// Compute mel spectrogram from audio samples.
///
/// Returns mel spectrogram as (num_mels, n_frames) array with log compression.
pub fn mel_spectrogram_ndarray(audio: &[f32], config: &MelConfig) -> Result<Array2<f32>> {
    // Compute STFT magnitude
    let spec = stft(audio, config.n_fft, config.hop_size, config.win_size, config.center)?;

    // Create mel filterbank
    let mel_basis = create_mel_filterbank(config);

    // Apply mel filterbank: (num_mels, n_fft/2+1) @ (n_fft/2+1, n_frames) = (num_mels, n_frames)
    let n_frames = spec.nrows();
    let mut mel_spec = Array2::<f32>::zeros((config.num_mels, n_frames));

    for frame in 0..n_frames {
        for mel_bin in 0..config.num_mels {
            let mut sum = 0.0f32;
            for fft_bin in 0..spec.ncols() {
                sum += mel_basis[[mel_bin, fft_bin]] * spec[[frame, fft_bin]];
            }
            // Apply spectral normalization (log compression)
            mel_spec[[mel_bin, frame]] = dynamic_range_compression(sum, 1e-5, 1.0);
        }
    }

    Ok(mel_spec)
}

/// Compute mel spectrogram and return as Candle Tensor.
///
/// Returns tensor of shape (1, num_mels, n_frames).
pub fn mel_spectrogram(audio: &[f32], config: &MelConfig, device: &Device) -> Result<Tensor> {
    let mel = mel_spectrogram_ndarray(audio, config)?;

    // Convert to tensor: (num_mels, n_frames) -> (1, num_mels, n_frames)
    let (num_mels, n_frames) = (mel.nrows(), mel.ncols());
    let data: Vec<f32> = mel.into_raw_vec();

    let tensor = Tensor::from_vec(data, (1, num_mels, n_frames), device)?;
    Ok(tensor)
}

/// Compute mel spectrogram from WAV file path.
///
/// Returns tensor of shape (1, num_mels, n_frames).
pub fn mel_spectrogram_from_file(
    path: impl AsRef<Path>,
    config: &MelConfig,
    device: &Device,
) -> Result<Tensor> {
    let (samples, sample_rate) = load_wav(path)?;

    // Verify sample rate matches config
    if sample_rate as usize != config.sampling_rate {
        return Err(anyhow!(
            "Sample rate mismatch: file has {} Hz, config expects {} Hz",
            sample_rate,
            config.sampling_rate
        ));
    }

    mel_spectrogram(&samples, config, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_mel_conversion() {
        // Test round-trip
        let freq = 1000.0;
        let mel = hz_to_mel(freq);
        let freq_back = mel_to_hz(mel);
        assert!((freq - freq_back).abs() < 1e-6);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let config = MelConfig::default();
        let fb = create_mel_filterbank(&config);
        assert_eq!(fb.nrows(), config.num_mels);
        assert_eq!(fb.ncols(), config.n_fft / 2 + 1);
    }

    #[test]
    fn test_hann_window() {
        let window = create_hann_window(1024);
        assert_eq!(window.len(), 1024);
        // Check endpoints are near zero
        assert!(window[0] < 1e-6);
        // Check center is near 1
        assert!((window[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stft_shape() {
        // Create dummy audio
        let audio: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = MelConfig::inference();

        let spec = stft(&audio, config.n_fft, config.hop_size, config.win_size, config.center)
            .expect("STFT should succeed");

        // Check output shape
        assert_eq!(spec.ncols(), config.n_fft / 2 + 1);
        assert!(spec.nrows() > 0);
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        // Create 1 second of audio at 24kHz
        let audio: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = MelConfig::inference();

        let mel = mel_spectrogram_ndarray(&audio, &config).expect("Mel spectrogram should succeed");

        assert_eq!(mel.nrows(), config.num_mels);
        assert!(mel.ncols() > 0);
    }
}
