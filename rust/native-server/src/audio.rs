//! Native audio processing module for CosyVoice.
//!
//! Provides mel spectrogram computation and audio I/O without Python dependencies.

use anyhow::{anyhow, Result};
use tracing::debug;
use candle_core::{npy::NpzTensors, Device, Tensor};
use crate::utils::StftModule;
use hound::WavReader;
use ndarray::Array2;
use realfft::RealFftPlanner;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::OnceLock;

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

/// Convert frequency in Hz to mel scale (Slaney).
fn hz_to_mel(freq: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let log_step = (6.4f64).ln() / 27.0; // ~0.06875

    if freq < min_log_hz {
        // Linear part: freq / (200.0 / 3.0)
        freq / (200.0 / 3.0)
    } else {
        // Log part
        min_log_mel + (freq / min_log_hz).ln() / log_step
    }
}

/// Convert mel scale to frequency in Hz (Slaney).
fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let log_step = (6.4f64).ln() / 27.0;

    if mel < min_log_mel {
        // Linear part
        mel * (200.0 / 3.0)
    } else {
        // Log part
        min_log_hz * (log_step * (mel - min_log_mel)).exp()
    }
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

        // Slaney-style area normalization: divide by width of the band in Hz
        // Factor = 2.0 / (f_m_plus - f_m_minus)
        let width = f_m_plus - f_m_minus;
        let norm_factor = 2.0 / width;
        if m == 0 {
            debug!("[Audio] Mel band 0: width={:.4} Hz, norm_factor={:.6}", width, norm_factor);
        }
        for k in 0..n_fft_bins {
            filterbank[[m, k]] *= norm_factor as f32;
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

const WHISPER_N_FFT: usize = 400;
const WHISPER_HOP_LENGTH: usize = 160;
const WHISPER_N_MELS: usize = 128;

static WHISPER_MEL_FILTERS_128: OnceLock<(Vec<f32>, usize, usize)> = OnceLock::new();

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn build_sinc_resample_kernels(
    orig: usize,
    new: usize,
    lowpass_filter_width: f64,
    rolloff: f64,
) -> (Vec<f32>, usize, usize) {
    let base_freq = (orig.min(new) as f64) * rolloff;
    let width = ((lowpass_filter_width * orig as f64) / base_freq).ceil() as isize;
    let kernel_len = (orig as isize + 2 * width) as usize;
    let scale = base_freq / orig as f64;

    let mut kernels = vec![0.0f32; new * kernel_len];
    for phase in 0..new {
        let phase_offset = -(phase as f64) / (new as f64);
        for k in 0..kernel_len {
            let idx = (k as isize - width) as f64 / orig as f64;
            let mut t = (phase_offset + idx) * base_freq;
            if t < -lowpass_filter_width {
                t = -lowpass_filter_width;
            } else if t > lowpass_filter_width {
                t = lowpass_filter_width;
            }
            let window = (t * PI / lowpass_filter_width / 2.0).cos().powi(2);
            let t_pi = t * PI;
            let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };
            let val = sinc * window * scale;
            kernels[phase * kernel_len + k] = val as f32;
        }
    }

    (kernels, kernel_len, width as usize)
}

fn reflect_pad_tensor(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let rank = x.rank();
    if rank < 2 {
        return Err(anyhow!("reflect_pad_tensor expects rank >= 2"));
    }
    let t = x.dim(rank - 1)?;
    if t <= pad {
        return Err(anyhow!("reflect_pad_tensor pad >= signal length"));
    }

    let mut indices = Vec::with_capacity(t + 2 * pad);
    for i in 0..pad {
        indices.push((pad - i) as u32);
    }
    for i in 0..t {
        indices.push(i as u32);
    }
    for i in 0..pad {
        indices.push((t - 2 - i) as u32);
    }

    let idx = Tensor::from_vec(indices, (t + 2 * pad,), x.device())?;
    x.index_select(&idx, rank - 1).map_err(Into::into)
}

/// Resample audio to match torchaudio.transforms.Resample (sinc_interp_hann).
pub fn resample_audio(samples: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(samples.to_vec());
    }
    if src_rate == 0 || dst_rate == 0 {
        return Err(anyhow!("Invalid sample rate: {} -> {}", src_rate, dst_rate));
    }
    let gcd = gcd_u32(src_rate, dst_rate);
    let orig = (src_rate / gcd) as usize;
    let new = (dst_rate / gcd) as usize;

    let (kernels, kernel_len, width) = build_sinc_resample_kernels(orig, new, 6.0, 0.99);

    let pad_left = width;
    let pad_right = width + orig;
    let mut padded = Vec::with_capacity(samples.len() + pad_left + pad_right);
    padded.extend(std::iter::repeat(0.0f32).take(pad_left));
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat(0.0f32).take(pad_right));

    let length = samples.len();
    let num_steps = length / orig + 1;
    let mut out = Vec::with_capacity(num_steps * new);
    for step in 0..num_steps {
        let start = step * orig;
        let frame = &padded[start..start + kernel_len];
        for phase in 0..new {
            let mut acc = 0.0f32;
            let kernel = &kernels[phase * kernel_len..(phase + 1) * kernel_len];
            for (k, w) in kernel.iter().enumerate() {
                acc += w * frame[k];
            }
            out.push(acc);
        }
    }

    let target_len = ((new as u64) * (length as u64) + (orig as u64) - 1) / (orig as u64);
    out.truncate(target_len as usize);
    Ok(out)
}

/// Compute Whisper-style log-mel spectrogram (n_mels=128, 16kHz) on CUDA.
///
/// Input tensor should be [batch, time] or [time] on CUDA.
/// Returns tensor of shape (batch, 128, frames).
pub fn whisper_log_mel_spectrogram_cuda(samples: &Tensor) -> Result<Tensor> {
    let device = samples.device();
    if !device.is_cuda() {
        return Err(anyhow!("CUDA device required for whisper_log_mel_spectrogram_cuda"));
    }

    let samples = if samples.rank() == 1 {
        samples.unsqueeze(0)?
    } else {
        samples.clone()
    };
    let (_b, t) = samples.dims2()?;
    if t < WHISPER_N_FFT {
        return Err(anyhow!("Audio too short for Whisper STFT"));
    }

    let padded = reflect_pad_tensor(&samples, WHISPER_N_FFT / 2)?;
    let stft = StftModule::new(WHISPER_N_FFT, WHISPER_HOP_LENGTH, false, device)?;
    let (real, imag) = stft.transform(&padded)?;
    let mag = real.sqr()?.add(&imag.sqr()?)?;

    let frames = mag.dim(2)?;
    if frames == 0 {
        return Err(anyhow!("Whisper STFT produced no usable frames"));
    }
    let mag = mag.narrow(2, 0, frames - 1)?;
    let mag = mag.contiguous()?;

    let (mel_filters, n_mels, n_freq) = {
        let filters = load_whisper_mel_filters_128()?;
        (filters.0.as_slice(), filters.1, filters.2)
    };
    if n_mels != WHISPER_N_MELS {
        return Err(anyhow!(
            "Whisper mel filter mismatch: expected {}, got {}",
            WHISPER_N_MELS,
            n_mels
        ));
    }

    let mel_filters = Tensor::from_vec(mel_filters.to_vec(), (n_mels, n_freq), device)?;
    let mag = mag.squeeze(0)?.contiguous()?;
    let mel_spec = mel_filters.matmul(&mag)?;

    let log_spec = mel_spec.clamp(1e-10f32, f32::INFINITY)?.log()?;
    let ln10 = Tensor::from_vec(vec![(10.0f32).ln()], (1,), device)?;
    let log_spec = log_spec.broadcast_div(&ln10)?;

    let max_val = log_spec.max_all()?.to_scalar::<f32>()?;
    let log_spec = log_spec.maximum(max_val - 8.0)?;
    let log_spec = log_spec.broadcast_add(&Tensor::from_vec(vec![4.0f32], (1,), device)?)?;
    let log_spec = log_spec.broadcast_div(&Tensor::from_vec(vec![4.0f32], (1,), device)?)?;

    log_spec.unsqueeze(0).map_err(Into::into)
}

/// Compute mel spectrogram on CUDA (matcha_compat settings).
///
/// Input tensor should be [batch, time] or [time] on CUDA.
/// Returns tensor of shape (batch, num_mels, frames).
pub fn mel_spectrogram_cuda(samples: &Tensor, config: &MelConfig) -> Result<Tensor> {
    let device = samples.device();
    if !device.is_cuda() {
        return Err(anyhow!("CUDA device required for mel_spectrogram_cuda"));
    }

    let samples = if samples.rank() == 1 {
        samples.unsqueeze(0)?
    } else {
        samples.clone()
    };
    let (_b, t) = samples.dims2()?;
    if t < config.win_size {
        return Err(anyhow!("Audio too short for mel spectrogram"));
    }

    let pad = (config.n_fft - config.hop_size) / 2;
    let padded = reflect_pad_tensor(&samples, pad)?;

    let stft = StftModule::new(config.n_fft, config.hop_size, false, device)?;
    let (real, imag) = stft.transform(&padded)?;
    let mag = real.sqr()?.add(&imag.sqr()?)?;
    let mag = mag.broadcast_add(&Tensor::from_vec(vec![1e-9f32], (1,), device)?)?;
    let mag = mag.sqrt()?;

    let mel_basis = create_mel_filterbank(config);
    let (num_mels, n_freq) = (mel_basis.nrows(), mel_basis.ncols());
    let (mel_data, offset) = mel_basis.into_raw_vec_and_offset();
    let mel_data = match offset {
        Some(off) => mel_data[off..].to_vec(),
        None => mel_data,
    };
    let mel_filters = Tensor::from_vec(mel_data, (num_mels, n_freq), device)?;

    let mag = mag.squeeze(0)?.contiguous()?;
    let mel_spec = mel_filters.matmul(&mag)?;
    let mel_spec = mel_spec.clamp(1e-5f32, f32::INFINITY)?;
    let mel_spec = mel_spec.log()?;

    mel_spec.unsqueeze(0).map_err(Into::into)
}

/// Resample audio using CUDA conv1d with torchaudio-compatible kernels.
/// Returns a Tensor of shape [1, target_len] on the given device.
pub fn resample_audio_cuda(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    device: &Device,
) -> Result<Tensor> {
    if src_rate == dst_rate {
        return Tensor::from_vec(samples.to_vec(), (1, samples.len()), device).map_err(Into::into);
    }
    if src_rate == 0 || dst_rate == 0 {
        return Err(anyhow!("Invalid sample rate: {} -> {}", src_rate, dst_rate));
    }
    if !device.is_cuda() {
        return Err(anyhow!("CUDA device required for resample_audio_cuda"));
    }

    let gcd = gcd_u32(src_rate, dst_rate);
    let orig = (src_rate / gcd) as usize;
    let new = (dst_rate / gcd) as usize;
    let (kernels, kernel_len, width) = build_sinc_resample_kernels(orig, new, 6.0, 0.99);

    let pad_left = width;
    let pad_right = width + orig;
    let mut padded = Vec::with_capacity(samples.len() + pad_left + pad_right);
    padded.extend(std::iter::repeat(0.0f32).take(pad_left));
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat(0.0f32).take(pad_right));

    let x = Tensor::from_vec(
        padded,
        (1, 1, pad_left + samples.len() + pad_right),
        device,
    )?;
    let kernel = Tensor::from_vec(kernels, (new, 1, kernel_len), device)?;
    let y = x.conv1d(&kernel, 0, orig, 1, 1)?;
    let (b, c, t) = y.dims3()?;
    let y = y.transpose(1, 2)?.reshape((b, t * c))?;

    let target_len = ((new as u64) * (samples.len() as u64) + (orig as u64) - 1) / (orig as u64);
    y.narrow(1, 0, target_len as usize).map_err(Into::into)
}

fn reflect_pad(samples: &[f32], pad: usize) -> Result<Vec<f32>> {
    if pad == 0 {
        return Ok(samples.to_vec());
    }
    if samples.len() <= pad {
        return Err(anyhow!(
            "Cannot reflect-pad {} samples with pad {}",
            samples.len(),
            pad
        ));
    }
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for i in 0..pad {
        let idx = pad - i;
        padded.push(samples[idx]);
    }
    padded.extend_from_slice(samples);
    for i in 0..pad {
        let idx = samples.len() - 2 - i;
        padded.push(samples[idx]);
    }
    Ok(padded)
}

fn load_whisper_mel_filters_128() -> Result<&'static (Vec<f32>, usize, usize)> {
    if let Some(filters) = WHISPER_MEL_FILTERS_128.get() {
        return Ok(filters);
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("mel_filters.npz");
    let npz = NpzTensors::new(&path)?;
    let tensor = npz
        .get("mel_128")?
        .ok_or_else(|| anyhow!("mel_128 not found in {:?}", path))?;
    let (n_mels, n_freq) = tensor.dims2()?;
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    let _ = WHISPER_MEL_FILTERS_128.set((data, n_mels, n_freq));
    WHISPER_MEL_FILTERS_128
        .get()
        .ok_or_else(|| anyhow!("Failed to cache mel_128 filters"))
}

/// Compute Whisper-style log-mel spectrogram (n_mels=128, 16kHz).
///
/// Returns tensor of shape (1, 128, frames).
pub fn whisper_log_mel_spectrogram(samples: &[f32], device: &Device) -> Result<Tensor> {
    if samples.len() < WHISPER_N_FFT {
        return Err(anyhow!("Audio too short for Whisper STFT"));
    }

    let padded = reflect_pad(samples, WHISPER_N_FFT / 2)?;
    let total_frames = (padded.len() - WHISPER_N_FFT) / WHISPER_HOP_LENGTH + 1;
    if total_frames == 0 {
        return Err(anyhow!("No frames available for Whisper STFT"));
    }
    let out_frames = total_frames.saturating_sub(1);
    if out_frames == 0 {
        return Err(anyhow!("Whisper STFT produced no usable frames"));
    }

    let window = create_hann_window(WHISPER_N_FFT);
    let n_freq = WHISPER_N_FFT / 2 + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WHISPER_N_FFT);
    let mut scratch = vec![Default::default(); fft.get_scratch_len()];
    let mut magnitudes = vec![0.0f32; out_frames * n_freq];

    for frame_idx in 0..total_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let mut frame: Vec<f32> = (0..WHISPER_N_FFT)
            .map(|i| padded[start + i] * window[i])
            .collect();

        let mut spectrum = vec![Default::default(); n_freq];
        fft.process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
            .map_err(|e| anyhow!("FFT error: {:?}", e))?;

        if frame_idx == total_frames - 1 {
            break;
        }

        let out_idx = frame_idx;
        for (bin_idx, c) in spectrum.iter().enumerate() {
            let mag = c.re * c.re + c.im * c.im;
            magnitudes[out_idx * n_freq + bin_idx] = mag;
        }
    }

    let (mel_filters, n_mels, filter_bins) = {
        let filters = load_whisper_mel_filters_128()?;
        (filters.0.as_slice(), filters.1, filters.2)
    };
    if n_mels != WHISPER_N_MELS || filter_bins != n_freq {
        return Err(anyhow!(
            "Whisper mel filter shape mismatch: expected {}x{}, got {}x{}",
            WHISPER_N_MELS,
            n_freq,
            n_mels,
            filter_bins
        ));
    }

    let mut log_spec = vec![0.0f32; n_mels * out_frames];
    let mut max_log = f32::NEG_INFINITY;
    for frame in 0..out_frames {
        for mel in 0..n_mels {
            let mut sum = 0.0f32;
            let filter_row = &mel_filters[mel * n_freq..(mel + 1) * n_freq];
            let mags = &magnitudes[frame * n_freq..(frame + 1) * n_freq];
            for (f, m) in filter_row.iter().zip(mags.iter()) {
                sum += f * m;
            }
            let log_val = sum.max(1e-10).log10();
            log_spec[mel * out_frames + frame] = log_val;
            if log_val > max_log {
                max_log = log_val;
            }
        }
    }

    let floor = max_log - 8.0;
    for v in log_spec.iter_mut() {
        let clipped = (*v).max(floor);
        *v = (clipped + 4.0) / 4.0;
    }

    Tensor::from_vec(log_spec, (1, n_mels, out_frames), device).map_err(Into::into)
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

        // Compute magnitude (match torch.stft + sqrt(sum + 1e-9))
        for (bin_idx, c) in spectrum.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
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
    let spec = stft(
        audio,
        config.n_fft,
        config.hop_size,
        config.win_size,
        config.center,
    )?;

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
    let (data, offset) = mel.into_raw_vec_and_offset();
    let data: Vec<f32> = match offset {
        Some(0) | None => data,
        Some(offset) => data.into_iter().skip(offset).collect(),
    };

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

// =============================================================================
// Kaldi-style Fbank (for Campplus speaker embedding)
// =============================================================================

fn kaldi_mel_scale(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn kaldi_mel_filterbank(
    num_bins: usize,
    window_length_padded: usize,
    sample_freq: f64,
    low_freq: f64,
    high_freq: f64,
) -> Result<Vec<f32>> {
    if num_bins <= 3 {
        return Err(anyhow!("num_bins must be > 3"));
    }
    if !window_length_padded.is_multiple_of(2) {
        return Err(anyhow!("window_length_padded must be even"));
    }

    let nyquist = 0.5 * sample_freq;
    let mut high_freq = high_freq;
    if high_freq <= 0.0 {
        high_freq += nyquist;
    }
    if !(0.0 <= low_freq && low_freq < nyquist && 0.0 < high_freq && high_freq <= nyquist) {
        return Err(anyhow!(
            "Bad mel freq bounds: low={} high={} nyquist={}",
            low_freq,
            high_freq,
            nyquist
        ));
    }

    let fft_bin_width = sample_freq / window_length_padded as f64;
    let mel_low = kaldi_mel_scale(low_freq);
    let mel_high = kaldi_mel_scale(high_freq);
    let mel_delta = (mel_high - mel_low) / (num_bins + 1) as f64;

    let num_fft_bins = window_length_padded / 2;
    let mut bins = vec![0.0f32; num_bins * (num_fft_bins + 1)];

    for mel_bin in 0..num_bins {
        let left_mel = mel_low + mel_bin as f64 * mel_delta;
        let center_mel = mel_low + (mel_bin + 1) as f64 * mel_delta;
        let right_mel = mel_low + (mel_bin + 2) as f64 * mel_delta;

        for fft_bin in 0..num_fft_bins {
            let freq = fft_bin_width * fft_bin as f64;
            let mel = kaldi_mel_scale(freq);
            let up = (mel - left_mel) / (center_mel - left_mel);
            let down = (right_mel - mel) / (right_mel - center_mel);
            let val = up.min(down).max(0.0);
            bins[mel_bin * (num_fft_bins + 1) + fft_bin] = val as f32;
        }
    }

    Ok(bins)
}

fn create_povey_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0];
    }
    (0..size)
        .map(|i| {
            let angle = 2.0 * PI * i as f64 / (size as f64 - 1.0);
            let hann = 0.5 - 0.5 * angle.cos();
            hann.powf(0.85) as f32
        })
        .collect()
}

/// Compute Kaldi-compatible log-fbank features (num_mel_bins=80, 16kHz).
///
/// Returns tensor of shape (1, frames, 80).
pub fn kaldi_fbank(samples: &[f32], sample_rate: u32, device: &Device) -> Result<Tensor> {
    let window_size = (sample_rate as f64 * 0.025) as usize;
    let window_shift = (sample_rate as f64 * 0.01) as usize;
    let padded_window_size = window_size.next_power_of_two();

    if samples.len() < window_size {
        return Err(anyhow!("Audio too short for Kaldi fbank"));
    }

    let num_frames = 1 + (samples.len() - window_size) / window_shift;
    let num_mel_bins = 80;
    let num_freq_bins = padded_window_size / 2 + 1;

    let mel_filters = kaldi_mel_filterbank(
        num_mel_bins,
        padded_window_size,
        sample_rate as f64,
        20.0,
        0.0,
    )?;

    let window = create_povey_window(window_size);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(padded_window_size);
    let mut scratch = vec![Default::default(); fft.get_scratch_len()];

    let mut feats = vec![0.0f32; num_frames * num_mel_bins];
    let mut spectrum = vec![Default::default(); num_freq_bins];

    for frame_idx in 0..num_frames {
        let start = frame_idx * window_shift;
        let mut frame = vec![0.0f32; padded_window_size];
        let slice = &samples[start..start + window_size];

        let mean = slice.iter().copied().sum::<f32>() / window_size as f32;
        for i in 0..window_size {
            let mut v = slice[i] - mean;
            if i > 0 {
                v -= 0.97 * (slice[i - 1] - mean);
            } else {
                v *= 1.0 - 0.97;
            }
            frame[i] = v * window[i];
        }

        fft.process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
            .map_err(|e| anyhow!("FFT error: {:?}", e))?;

        let mut power = vec![0.0f32; num_freq_bins];
        for (i, c) in spectrum.iter().enumerate() {
            power[i] = c.re * c.re + c.im * c.im;
        }

        for mel in 0..num_mel_bins {
            let filter_row = &mel_filters[mel * num_freq_bins..(mel + 1) * num_freq_bins];
            let mut sum = 0.0f32;
            for (f, p) in filter_row.iter().zip(power.iter()) {
                sum += f * p;
            }
            let log_val = sum.max(f32::EPSILON).ln();
            feats[frame_idx * num_mel_bins + mel] = log_val;
        }
    }

    // Mean-normalize across frames (per mel bin)
    for mel in 0..num_mel_bins {
        let mut mean = 0.0f32;
        for frame in 0..num_frames {
            mean += feats[frame * num_mel_bins + mel];
        }
        mean /= num_frames as f32;
        for frame in 0..num_frames {
            let idx = frame * num_mel_bins + mel;
            feats[idx] -= mean;
        }
    }

    Tensor::from_vec(feats, (1, num_frames, num_mel_bins), device).map_err(Into::into)
}

/// Compute Kaldi-compatible log-fbank features on CUDA.
///
/// Input tensor should be [batch, time] or [time] on CUDA.
/// Returns tensor of shape (batch, frames, 80).
pub fn kaldi_fbank_cuda(samples: &Tensor, sample_rate: u32) -> Result<Tensor> {
    let device = samples.device();
    if !device.is_cuda() {
        return Err(anyhow!("CUDA device required for kaldi_fbank_cuda"));
    }

    let samples = if samples.rank() == 1 {
        samples.unsqueeze(0)?
    } else {
        samples.clone()
    };
    let (b, t) = samples.dims2()?;

    let window_size = (sample_rate as f64 * 0.025) as usize;
    let window_shift = (sample_rate as f64 * 0.01) as usize;
    let padded_window_size = window_size.next_power_of_two();

    if t < window_size {
        return Err(anyhow!("Audio too short for Kaldi fbank"));
    }

    let num_frames = 1 + (t - window_size) / window_shift;
    let num_mel_bins = 80;
    let num_freq_bins = padded_window_size / 2 + 1;

    let mel_filters = kaldi_mel_filterbank(
        num_mel_bins,
        padded_window_size,
        sample_rate as f64,
        20.0,
        0.0,
    )?;

    let base = Tensor::arange(0u32, window_size as u32, device)?;
    let offsets = Tensor::arange(0u32, num_frames as u32, device)?;
    let shift = Tensor::from_vec(vec![window_shift as u32], (1,), device)?;
    let offsets = offsets.broadcast_mul(&shift)?;
    let idx = offsets.unsqueeze(1)?.broadcast_add(&base.unsqueeze(0)?)?;
    let idx = idx.unsqueeze(0)?.broadcast_as((b, num_frames, window_size))?.contiguous()?;

    let x = samples.unsqueeze(1)?.broadcast_as((b, num_frames, t))?.contiguous()?;
    let frames = x.gather(&idx, 2)?;

    let denom = Tensor::from_vec(vec![window_size as f32], (1,), device)?;
    let mean = frames.sum(2)?.broadcast_div(&denom)?;
    let frames = frames.broadcast_sub(&mean.unsqueeze(2)?)?;

    let coeff = Tensor::from_vec(vec![0.97f32], (1,), device)?;
    let first = frames.narrow(2, 0, 1)?;
    let rest = frames.narrow(2, 1, window_size - 1)?;
    let prev = frames.narrow(2, 0, window_size - 1)?;
    let rest_emph = rest.broadcast_sub(&prev.broadcast_mul(&coeff)?)?;
    let first_emph = first.broadcast_mul(&Tensor::from_vec(vec![1.0f32 - 0.97], (1,), device)?)?;
    let frames = Tensor::cat(&[&first_emph, &rest_emph], 2)?;

    let window = create_povey_window(window_size);
    let window = Tensor::from_vec(window, (1, 1, window_size), device)?;
    let frames = frames.broadcast_mul(&window)?;

    let pad_len = padded_window_size - window_size;
    let frames = if pad_len > 0 {
        let zeros = Tensor::zeros((b, num_frames, pad_len), candle_core::DType::F32, device)?;
        Tensor::cat(&[&frames, &zeros], 2)?
    } else {
        frames
    };

    let frames = frames.reshape((b * num_frames, padded_window_size))?;

    let mut real = Vec::with_capacity(num_freq_bins * padded_window_size);
    let mut imag = Vec::with_capacity(num_freq_bins * padded_window_size);
    for k in 0..num_freq_bins {
        for n_idx in 0..padded_window_size {
            let theta = -2.0 * PI * (k as f64) * (n_idx as f64) / (padded_window_size as f64);
            real.push(theta.cos() as f32);
            imag.push(theta.sin() as f32);
        }
    }
    let dft_real = Tensor::from_vec(real, (num_freq_bins, padded_window_size), device)?;
    let dft_imag = Tensor::from_vec(imag, (num_freq_bins, padded_window_size), device)?;

    let real = frames.matmul(&dft_real.transpose(0, 1)?)?;
    let imag = frames.matmul(&dft_imag.transpose(0, 1)?)?;
    let power = real.sqr()?.add(&imag.sqr()?)?;

    let mel_filters = Tensor::from_vec(mel_filters, (num_mel_bins, num_freq_bins), device)?;
    let power_t = power.transpose(0, 1)?;
    let feats = mel_filters.matmul(&power_t)?;
    let feats = feats.transpose(0, 1)?;
    let feats = feats.clamp(f32::EPSILON, f32::INFINITY)?.log()?;
    let feats = feats.reshape((b, num_frames, num_mel_bins))?;

    let mean = feats.mean(1)?;
    feats.broadcast_sub(&mean.unsqueeze(1)?).map_err(Into::into)
}

// =============================================================================
// Audio Post-Processing Functions
// =============================================================================

/// Configuration for audio post-processing.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    /// Normalize audio to target peak level (0.0 to 1.0)
    pub normalize: bool,
    /// Target peak level for normalization (default: 0.95)
    pub target_peak: f32,
    /// Upsample from 24kHz to 48kHz
    pub upsample_48k: bool,
    /// Apply soft clipping to prevent harsh distortion
    pub soft_clip: bool,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            target_peak: 0.95,
            upsample_48k: false,
            soft_clip: true,
        }
    }
}

/// Normalize audio to a target peak level.
///
/// # Arguments
/// * `samples` - Audio samples (modified in place)
/// * `target_peak` - Target peak amplitude (0.0 to 1.0)
///
/// # Returns
/// The gain applied
pub fn normalize_audio(samples: &mut [f32], target_peak: f32) -> f32 {
    if samples.is_empty() {
        return 1.0;
    }

    // Find current peak
    let current_peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));

    if current_peak < 1e-6 {
        return 1.0; // Avoid division by zero for silence
    }

    // Calculate and apply gain
    let gain = target_peak / current_peak;
    debug!(
        "normalize_audio: current_peak={:.6}, target_peak={:.6}, gain={:.6}",
        current_peak, target_peak, gain
    );
    for sample in samples.iter_mut() {
        *sample *= gain;
    }

    gain
}

/// Apply soft clipping to prevent harsh digital clipping.
///
/// Uses tanh-based soft clipping which compresses values approaching the limit.
pub fn soft_clip(samples: &mut [f32], threshold: f32) {
    let inv_threshold = 1.0 / threshold;
    for sample in samples.iter_mut() {
        if sample.abs() > threshold {
            // Soft clip using tanh
            *sample = sample.signum() * threshold * (*sample * inv_threshold).tanh();
        }
    }
}

/// Upsample audio from 24kHz to 48kHz using linear interpolation.
///
/// This is a simple upsampling method. For higher quality, consider
/// using sinc interpolation or polyphase filters.
pub fn upsample_2x_linear(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(samples.len() * 2);

    for i in 0..samples.len() - 1 {
        let s0 = samples[i];
        let s1 = samples[i + 1];
        output.push(s0);
        output.push((s0 + s1) / 2.0); // Linear interpolation
    }
    // Handle last sample
    output.push(samples[samples.len() - 1]);
    output.push(samples[samples.len() - 1]); // Duplicate last

    output
}

/// Upsample audio from 24kHz to 48kHz using cubic interpolation.
///
/// Higher quality than linear interpolation.
pub fn upsample_2x_cubic(samples: &[f32]) -> Vec<f32> {
    if samples.len() < 4 {
        return upsample_2x_linear(samples);
    }

    let mut output = Vec::with_capacity(samples.len() * 2);

    // Pad first sample
    output.push(samples[0]);
    output.push((samples[0] + samples[1]) / 2.0);

    for i in 1..samples.len() - 2 {
        let s_m1 = samples[i - 1];
        let s_0 = samples[i];
        let s_1 = samples[i + 1];
        let s_2 = samples[i + 2];

        output.push(s_0);

        // Cubic interpolation at t=0.5
        let t = 0.5f32;
        let interpolated = s_0
            + 0.5
                * t
                * (s_1 - s_m1
                    + t * (2.0 * s_m1 - 5.0 * s_0 + 4.0 * s_1 - s_2
                        + t * (3.0 * (s_0 - s_1) + s_2 - s_m1)));
        output.push(interpolated);
    }

    // Handle last samples
    let n = samples.len();
    output.push(samples[n - 2]);
    output.push((samples[n - 2] + samples[n - 1]) / 2.0);
    output.push(samples[n - 1]);
    output.push(samples[n - 1]);

    output
}

/// Apply full post-processing pipeline to audio samples.
///
/// # Arguments
/// * `samples` - Input audio samples (consumed)
/// * `config` - Post-processing configuration
///
/// # Returns
/// Processed audio samples and the new sample rate
pub fn post_process_audio(mut samples: Vec<f32>, config: &PostProcessConfig) -> (Vec<f32>, u32) {
    let mut sample_rate = 24000u32;

    // 1. Normalize
    if config.normalize {
        normalize_audio(&mut samples, config.target_peak);
    }

    // 2. Soft clip
    if config.soft_clip {
        soft_clip(&mut samples, 0.99);
    }

    // 3. Upsample (if requested)
    if config.upsample_48k {
        samples = upsample_2x_cubic(&samples);
        sample_rate = 48000;
    }

    (samples, sample_rate)
}

/// Convert f32 samples to i16 for WAV output.
pub fn samples_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
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

        let spec = stft(
            &audio,
            config.n_fft,
            config.hop_size,
            config.win_size,
            config.center,
        )
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

    #[test]
    fn test_normalize_audio() {
        let mut samples = vec![0.5, -0.3, 0.2, -0.1];
        let gain = normalize_audio(&mut samples, 0.95);

        // Peak should now be 0.95
        let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!((peak - 0.95).abs() < 1e-6);
        assert!(gain > 1.0); // Should have increased volume
    }

    #[test]
    fn test_soft_clip() {
        let mut samples = vec![1.5, -1.2, 0.5, -0.3];
        soft_clip(&mut samples, 0.99);

        // All values should be within [-0.99, 0.99] range after clipping
        for &s in &samples {
            assert!(s.abs() <= 0.99);
        }
        // Values below threshold should be unchanged
        assert!((samples[2] - 0.5).abs() < 1e-6);
        assert!((samples[3] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_upsample_2x_linear() {
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let upsampled = upsample_2x_linear(&samples);

        // Should be roughly 2x length
        assert_eq!(upsampled.len(), 8);
        // Original samples should be preserved at even indices
        assert!((upsampled[0] - 0.0).abs() < 1e-6);
        assert!((upsampled[2] - 1.0).abs() < 1e-6);
        // Interpolated values at odd indices
        assert!((upsampled[1] - 0.5).abs() < 1e-6); // (0.0 + 1.0) / 2
    }

    #[test]
    fn test_upsample_2x_cubic() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let upsampled = upsample_2x_cubic(&samples);

        // Should be roughly 2x length
        assert!(upsampled.len() >= samples.len() * 2 - 2);
    }

    #[test]
    fn test_post_process_audio_default() {
        let samples: Vec<f32> = (0..24000).map(|i| 0.3 * (i as f32 * 0.1).sin()).collect();
        let config = PostProcessConfig::default();

        let (processed, sample_rate) = post_process_audio(samples, &config);

        // Default doesn't upsample
        assert_eq!(sample_rate, 24000);
        // Should normalize to 0.95 peak
        let peak = processed.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_post_process_audio_with_upsample() {
        let samples: Vec<f32> = (0..24000).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        let config = PostProcessConfig {
            upsample_48k: true,
            ..Default::default()
        };

        let (processed, sample_rate) = post_process_audio(samples, &config);

        // Should upsample to 48kHz
        assert_eq!(sample_rate, 48000);
        // Length should approximately double
        assert!(processed.len() >= 24000 * 2 - 10);
    }

    #[test]
    fn test_samples_to_i16() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5];
        let i16_samples = samples_to_i16(&samples);

        assert_eq!(i16_samples[0], 0);
        assert_eq!(i16_samples[1], 16383); // 0.5 * 32767
        assert_eq!(i16_samples[2], -16383);
        assert_eq!(i16_samples[3], 32767);
        assert_eq!(i16_samples[4], -32767);
        // Values > 1.0 should be clamped
        assert_eq!(i16_samples[5], 32767);
        assert_eq!(i16_samples[6], -32768);
    }
}
