use candle_core::{Device, Result, Tensor};
use std::f64::consts::PI;

#[allow(dead_code)]
pub struct STFT {
    n_fft: usize,
    hop_length: usize,
    window: Tensor,
    dft_mat_real: Tensor,
    dft_mat_imag: Tensor,
    device: Device,
}

impl STFT {
    pub fn new(n_fft: usize, hop_length: usize, device: &Device) -> Result<Self> {
        let window = hann_window(n_fft, device)?;
        let (dft_mat_real, dft_mat_imag) = dft_matrix(n_fft, device)?;

        Ok(Self {
            n_fft,
            hop_length,
            window,
            dft_mat_real,
            dft_mat_imag,
            device: device.clone(),
        })
    }

    /// Short-Time Fourier Transform using Matrix Multiplication
    /// x: [Batch, Time]
    /// Returns: (magnitude, phase) or (real, imag)?
    /// HiFT expects: pure STFT return.
    /// Returns: (real, imag) each of shape [Batch, Freq, Frames]
    #[allow(dead_code)]
    fn forward(&self, _x: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Pad signal if needed (reflection padding is usually handled outside or assume input is padded)
        // For HiFT, input 's' is source excitation, usually matches hop size logic.

        // 2. Unfold/Frame the signal
        // Candle doesn't have `unfold` for 1D yet easily, but we can reshape if we assume exact framing.
        // Actually for small n_fft=16, we might need a custom framing kernel or just loop if efficient?
        // No, looping is bad.
        // Let's assume we can use `conv1d` to do STFT!
        // STFT is equivalent to Conv1d with kernel=window * basis.

        // However, since we defined this struct, let's use the Conv1d approach as it is standard and fast in Candle.
        // We need to fuse window * dft_basis into filters.

        // Re-do Init to pre-compute filters
        Err(candle_core::Error::Msg("Use StftModule instead".into()))
    }
}

pub struct StftModule {
    n_fft: usize,
    hop_length: usize,
    filters_real: Tensor, // [n_fft/2 + 1, 1, n_fft]
    filters_imag: Tensor, // [n_fft/2 + 1, 1, n_fft]
    center: bool,
    _device: Device,
}

impl StftModule {
    pub fn new(n_fft: usize, hop_length: usize, center: bool, device: &Device) -> Result<Self> {
        let window = hann_window(n_fft, device)?; // [n_fft]
        let (dft_real, dft_imag) = dft_matrix(n_fft, device)?; // [n_fft/2+1, n_fft]

        // Apply window to filters
        // filter = dft_basis * window
        let filters_real = dft_real.broadcast_mul(&window.unsqueeze(0)?)?;
        let filters_imag = dft_imag.broadcast_mul(&window.unsqueeze(0)?)?;

        // Reshape for Conv1d: [OutCh, InCh, Kernel] -> [n_fft/2+1, 1, n_fft]
        let filters_real = filters_real.unsqueeze(1)?;
        let filters_imag = filters_imag.unsqueeze(1)?;

        Ok(Self {
            n_fft,
            hop_length,
            filters_real,
            filters_imag,
            center,
            _device: device.clone(),
        })
    }

    pub fn transform(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x_in = if self.center {
            reflect_pad_1d(x, self.n_fft / 2)?
        } else {
            x.clone()
        };

        // x: [Batch, Time] -> need [Batch, 1, Time] for conv1d
        let x_unsqueezed = if x_in.rank() == 2 {
            x_in.unsqueeze(1)?
        } else {
            x_in
        };

        let real = x_unsqueezed.conv1d(
            &self.filters_real,
            0,
            self.hop_length,
            1, // dilation
            1, // groups
        )?;

        let imag = x_unsqueezed.conv1d(&self.filters_imag, 0, self.hop_length, 1, 1)?;

        // Output: [Batch, Freq, Frames]
        Ok((real, imag))
    }
}

use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, Module};
use rustfft::{num_complex::Complex, FftPlanner};

pub struct InverseStftModule {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    center: bool,
    device: Device,
}

impl InverseStftModule {
    pub fn new(n_fft: usize, hop_length: usize, center: bool, device: &Device) -> Result<Self> {
        let window_cpu = hann_window(n_fft, &Device::Cpu)?;
        let window_vec = window_cpu.to_vec1::<f32>()?;

        // Debug: print first few window values
        eprintln!(
            "    [InverseStftModule] DC Weights (Re): {:?}",
            &window_vec[0..n_fft.min(16)]
        );

        Ok(Self {
            n_fft,
            hop_length,
            window: window_vec,
            center,
            device: device.clone(),
        })
    }

    /// Proper ISTFT using iRFFT + overlap-add, matching torch.istft
    /// Input: Magnitude, Phase [Batch, Freq, Frames]
    /// Output: Audio [Batch, 1, Time]
    pub fn forward(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let device = magnitude.device();
        let (batch, n_freq, frames) = magnitude.dims3()?;

        if n_freq != self.n_fft / 2 + 1 {
            return Err(candle_core::Error::Msg(format!(
                "ISTFT: expected {} freq bins, got {}",
                self.n_fft / 2 + 1,
                n_freq
            )));
        }

        let real = magnitude.broadcast_mul(&phase.cos()?)?;
        let imag = magnitude.broadcast_mul(&phase.sin()?)?;

        // Move to CPU
        let real_cpu = real
            .to_device(&Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        let imag_cpu = imag
            .to_device(&Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        let real_vec: Vec<f32> = real_cpu.flatten_all()?.to_vec1()?;
        let imag_vec: Vec<f32> = imag_cpu.flatten_all()?.to_vec1()?;

        let out_len = (frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0f32; batch * out_len];

        // Pre-compute window sum for normalization (NOLA) - computed once as it's the same for all batches
        let mut window_sum = vec![0.0f32; out_len];
        for frame in 0..frames {
            let start = frame * self.hop_length;
            for i in 0..self.n_fft {
                if start + i < out_len {
                    window_sum[start + i] += self.window[i] * self.window[i];
                }
            }
        }

        // Setup full complex iFFT using rustfft
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);
        let mut scratch = vec![Complex::new(0.0f32, 0.0f32); ifft.get_inplace_scratch_len()];
        let scale = 1.0 / self.n_fft as f32;

        for b in 0..batch {
            for frame in 0..frames {
                // Construct spectrum
                let mut spectrum = Vec::with_capacity(self.n_fft);

                // 0..n_freq
                for k in 0..n_freq {
                    let idx = (b * n_freq * frames) + (k * frames) + frame;
                    spectrum.push(Complex::new(real_vec[idx], imag_vec[idx]));
                }

                // Conjugates n_freq-2 down to 1 (reconstruct full Hermitian spectrum)
                for k in (1..self.n_fft / 2).rev() {
                    let src_idx = (b * n_freq * frames) + (k * frames) + frame;
                    spectrum.push(Complex::new(real_vec[src_idx], -imag_vec[src_idx]));
                }

                ifft.process_with_scratch(&mut spectrum, &mut scratch);

                // Overlap-add
                let start = frame * self.hop_length;
                for (i, c) in spectrum.iter().enumerate() {
                    let val = c.re * scale;
                    output[b * out_len + start + i] += val * self.window[i];
                }
            }

            // Normalize by window sum (NOLA)
            for i in 0..out_len {
                let ws = window_sum[i];
                if ws > 1e-8 {
                    output[b * out_len + i] /= ws;
                }
            }
        }

        let pad = self.n_fft / 2;
        let mut trimmed = Vec::with_capacity(batch * ((frames - 1) * self.hop_length));

        for b in 0..batch {
            let start = b * out_len + pad;
            let trimmed_len = out_len - 2 * pad;
            trimmed.extend_from_slice(&output[start..start + trimmed_len]);
        }

        let out_t = trimmed.len() / batch;
        Tensor::from_vec(trimmed, (batch, out_t), device)
    }
}

/// Old ConvTranspose-based ISTFT (kept for reference/comparison)
#[allow(dead_code)]
pub struct InverseStftModuleConv {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    conv_real: ConvTranspose1d,
    conv_imag: ConvTranspose1d,
    center: bool,
    _device: Device,
}

#[allow(dead_code)]
impl InverseStftModuleConv {
    pub fn new(n_fft: usize, hop_length: usize, center: bool, device: &Device) -> Result<Self> {
        let _window = hann_window(n_fft, device)?;
        let window_cpu = hann_window(n_fft, &Device::Cpu)?;
        let window_vec = window_cpu.to_vec1::<f32>()?;

        let n_bins = n_fft / 2 + 1;
        let mut real_weights = Vec::with_capacity(n_bins * n_fft);
        let mut imag_weights = Vec::with_capacity(n_bins * n_fft);

        let scale = 1.0 / (n_fft as f64);

        for k in 0..n_bins {
            let factor = if k == 0 || k == n_fft / 2 { 1.0 } else { 2.0 };

            for n in 0..n_fft {
                let theta = 2.0 * PI * (k as f64) * (n as f64) / (n_fft as f64);
                let win_val = 0.5 * (1.0 - (2.0 * PI * n as f64 / n_fft as f64).cos());

                let cos_val = theta.cos() * factor * scale * win_val;
                let sin_val = theta.sin() * factor * scale * win_val;

                real_weights.push(cos_val as f32);
                imag_weights.push((-sin_val) as f32);
            }
        }

        let w_real = Tensor::from_vec(real_weights, (n_bins, 1, n_fft), device)?;
        let w_imag = Tensor::from_vec(imag_weights, (n_bins, 1, n_fft), device)?;

        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride: hop_length,
            dilation: 1,
            groups: 1,
        };

        let conv_real = ConvTranspose1d::new(w_real, None, cfg);
        let conv_imag = ConvTranspose1d::new(w_imag, None, cfg);

        Ok(Self {
            n_fft,
            hop_length,
            window: window_vec,
            conv_real,
            conv_imag,
            center,
            _device: device.clone(),
        })
    }

    pub fn forward(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let real = magnitude.broadcast_mul(&phase.cos()?)?;
        let imag = magnitude.broadcast_mul(&phase.sin()?)?;

        let y_real = self.conv_real.forward(&real)?;
        let y_imag = self.conv_imag.forward(&imag)?;

        let y = (y_real + y_imag)?;

        let frames = magnitude.dim(2)?;
        let out_len = (frames - 1) * self.hop_length + self.n_fft;
        let mut window_sums = vec![0.0f32; out_len];
        for frame in 0..frames {
            let start = frame * self.hop_length;
            for n in 0..self.n_fft {
                let idx = start + n;
                if idx < out_len {
                    let win = self.window[n];
                    window_sums[idx] += win * win;
                }
            }
        }
        for v in window_sums.iter_mut() {
            if *v < 1e-8 {
                *v = 1.0;
            }
        }

        let window_sums = Tensor::from_vec(window_sums, (1, 1, out_len), y.device())?;
        let window_sums = window_sums.broadcast_as(y.shape())?;
        let y = y.broadcast_div(&window_sums)?;

        if self.center {
            let pad = self.n_fft / 2;
            if out_len > 2 * pad {
                return y.narrow(2, pad, out_len - 2 * pad);
            }
        }

        Ok(y)
    }
}

fn reflect_pad_1d(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }

    let device = x.device();
    let (b, c, t, rank) = if x.rank() == 2 {
        let (b, t) = x.dims2()?;
        (b, 1usize, t, 2usize)
    } else if x.rank() == 3 {
        let (b, c, t) = x.dims3()?;
        (b, c, t, 3usize)
    } else {
        return Err(candle_core::Error::msg(
            "reflect_pad_1d expects rank-2 or rank-3 input",
        ));
    };

    if t <= pad {
        return Err(candle_core::Error::msg(
            "reflect_pad_1d pad >= signal length",
        ));
    }

    let x_cpu = x.to_device(&Device::Cpu)?;
    let x_vec = x_cpu.flatten_all()?.to_vec1::<f32>()?;

    let out_t = t + 2 * pad;
    let mut out = Vec::with_capacity(b * c * out_t);

    for bi in 0..b {
        for ci in 0..c {
            let offset = (bi * c + ci) * t;
            // Left pad
            for i in 0..pad {
                let idx = pad - i;
                out.push(x_vec[offset + idx]);
            }
            // Original
            out.extend_from_slice(&x_vec[offset..offset + t]);
            // Right pad
            for i in 0..pad {
                let idx = t - 2 - i;
                out.push(x_vec[offset + idx]);
            }
        }
    }

    if rank == 2 {
        Tensor::from_vec(out, (b, out_t), device)
    } else {
        Tensor::from_vec(out, (b, c, out_t), device)
    }
}

/// Generates Hann Window of size N (Periodic)
/// Matches `scipy.signal.get_window("hann", n, fftbins=True)` used in CosyVoice.
fn hann_window(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n);
    if n == 1 {
        data.push(1.0);
    } else {
        for i in 0..n {
            // Periodic: denominator is n
            let v = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
            data.push(v as f32);
        }
    }
    Tensor::from_vec(data, (n,), device)
}

/// Generates DFT Matrix for n_fft (only first n_fft/2 + 1 rows)
/// Returns (Real, Imag) matrices of shape [n_fft/2 + 1, n_fft]
fn dft_matrix(n: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let n_bins = n / 2 + 1;
    let mut real = Vec::with_capacity(n_bins * n);
    let mut imag = Vec::with_capacity(n_bins * n);

    for k in 0..n_bins {
        for n_idx in 0..n {
            let theta = -2.0 * PI * (k as f64) * (n_idx as f64) / (n as f64);
            real.push(theta.cos() as f32);
            imag.push(theta.sin() as f32);
        }
    }

    let real_t = Tensor::from_vec(real, (n_bins, n), device)?;
    let imag_t = Tensor::from_vec(imag, (n_bins, n), device)?;

    Ok((real_t, imag_t))
}
