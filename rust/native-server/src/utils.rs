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

pub struct InverseStftModule {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    conv_real: ConvTranspose1d,
    conv_imag: ConvTranspose1d,
    center: bool,
    _device: Device,
}

impl InverseStftModule {
    pub fn new(n_fft: usize, hop_length: usize, center: bool, device: &Device) -> Result<Self> {
        let _window = hann_window(n_fft, device)?; // [n_fft]
        let window_cpu = hann_window(n_fft, &Device::Cpu)?;
        let window_vec = window_cpu.to_vec1::<f32>()?;

        let n_bins = n_fft / 2 + 1;
        let mut real_weights = Vec::with_capacity(n_bins * n_fft);
        let mut imag_weights = Vec::with_capacity(n_bins * n_fft);

        // We construct weights for ConvTranspose1d: [InCh, OutCh/Group, Kernel]
        // InCh = n_bins, OutCh = 1, Kernel = n_fft.
        // Formula: x[n] = (1/N) * sum_k (X[k] * exp(j 2pi k n / N))
        // Real part logic:
        // k=0, k=N/2: Re[k]*cos(...) * window
        // 0<k<N/2: 2*Re[k]*cos(...) * window
        // Imag part logic:
        // k=0, k=N/2: 0 (Imag part of DC/Nyquist is 0 usually, or doesn't contribute to real signal if Hermitian)
        // 0<k<N/2: -2*Im[k]*sin(...) * window

        let scale = 1.0 / (n_fft as f64);

        for k in 0..n_bins {
            let factor = if k == 0 || k == n_fft / 2 { 1.0 } else { 2.0 };

            for n in 0..n_fft {
                let theta = 2.0 * PI * (k as f64) * (n as f64) / (n_fft as f64);
                let win_val = 0.5 * (1.0 - (2.0 * PI * n as f64 / n_fft as f64).cos()); // Manual Hann to be sure

                let cos_val = theta.cos() * factor * scale * win_val;
                let sin_val = theta.sin() * factor * scale * win_val;

                // For real input, we use cos basis
                real_weights.push(cos_val as f32);

                // For imag input, we use -sin basis
                // Im[k] * (cos + j sin) -> j * Im[k] * cos - Im[k] * sin
                // Real part is -Im[k]*sin
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

        // We use using_weights directly to avoid variable creating overhead if not needed,
        // but candle_nn::ConvTranspose1d::new expects variable or we construct struct manually.
        // Let's construct struct manually to use fixed weights.
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

    /// Input: Magnitude, Phase [Batch, Freq, Frames]
    /// Output: Audio [Batch, 1, Time]
    pub fn forward(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let real = magnitude.broadcast_mul(&phase.cos()?)?;
        let imag = magnitude.broadcast_mul(&phase.sin()?)?;

        // ConvTranspose1d expects [Batch, InChannels, Time/Frames]
        // Our input is [Batch, Freq, Frames].

        // Pass through filters
        let y_real = self.conv_real.forward(&real)?;
        let y_imag = self.conv_imag.forward(&imag)?;

        // Sum components
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

        // Output might be padded due to Centered STFT assumptions in PyTorch?
        // Match torch.istft(center=True) by trimming n_fft/2 on both sides.
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

/// Generates Hann Window of size N
fn hann_window(n: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let v = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
        data.push(v as f32);
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
