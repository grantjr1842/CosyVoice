// use crate::utils::StftModule; // Commented out until used
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;
#[cfg(feature = "f0-libtorch")]
use tch::{CModule, Device as TchDevice, Kind as TchKind, Tensor as TchTensor};

struct HiftParitySeeds {
    rand_ini: Option<Vec<f32>>,
    sine_noise_cache: Option<Tensor>,
    source_noise_cache: Option<Tensor>,
}

fn load_hift_parity_seeds_from_env() -> Result<Option<HiftParitySeeds>> {
    let path = match std::env::var("HIFT_PARITY_SEEDS_PATH") {
        Ok(value) if !value.is_empty() => value,
        _ => return Ok(None),
    };
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!(
            "    [HiFT] Parity seeds path not found: {}",
            path.display()
        );
        return Ok(None);
    }

    let cpu = Device::Cpu;
    let mut tensors = candle_core::safetensors::load(path, &cpu)?;
    let rand_ini = if let Some(t) = tensors.remove("rand_ini") {
        Some(t.flatten_all()?.to_vec1::<f32>()?)
    } else {
        None
    };
    let sine_noise_cache = tensors.remove("sine_noise_cache");
    let source_noise_cache = tensors.remove("source_noise_cache");

    if rand_ini.is_none() && sine_noise_cache.is_none() && source_noise_cache.is_none() {
        return Ok(None);
    }

    Ok(Some(HiftParitySeeds {
        rand_ini,
        sine_noise_cache,
        source_noise_cache,
    }))
}

fn load_hift_f0_override_from_env(device: &Device) -> Result<Option<Tensor>> {
    let path = match std::env::var("HIFT_F0_OVERRIDE_PATH") {
        Ok(value) if !value.is_empty() => value,
        _ => return Ok(None),
    };
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!(
            "    [HiFT] F0 override path not found: {}",
            path.display()
        );
        return Ok(None);
    }

    let cpu = Device::Cpu;
    let mut tensors = candle_core::safetensors::load(path, &cpu)?;
    let f0 = match tensors.remove("f0_output") {
        Some(tensor) => tensor,
        None => {
            eprintln!(
                "    [HiFT] F0 override missing f0_output in {}",
                path.display()
            );
            return Ok(None);
        }
    };

    let f0 = match f0.rank() {
        1 => f0.unsqueeze(0)?.unsqueeze(0)?,
        2 => f0.unsqueeze(1)?,
        3 => {
            let (_b, c, t) = f0.dims3()?;
            if c == 1 {
                f0
            } else if t == 1 {
                f0.transpose(1, 2)?
            } else {
                return Err(candle_core::Error::Msg(format!(
                    "f0_output must have channel dim 1 or last dim 1, got {:?}",
                    f0.shape()
                )));
            }
        }
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "f0_output must be rank 1-3, got rank {}",
                f0.rank()
            )))
        }
    };

    Ok(Some(f0.to_device(device)?))
}

#[cfg(feature = "f0-libtorch")]
struct TchF0Predictor {
    module: CModule,
}

#[cfg(feature = "f0-libtorch")]
impl TchF0Predictor {
    fn new(path: &Path) -> Result<Self> {
        let module = CModule::load(path).map_err(|err| {
            candle_core::Error::Msg(format!(
                "Failed to load torchscript f0 predictor from {}: {}",
                path.display(),
                err
            ))
        })?;
        Ok(Self { module })
    }

    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mel_cpu = mel.to_device(&Device::Cpu)?;
        let (b, c, t) = mel_cpu.dims3()?;
        let mel_vec = mel_cpu.flatten_all()?.to_vec1::<f32>()?;
        let input =
            TchTensor::from_slice(&mel_vec).reshape(&[b as i64, c as i64, t as i64]);
        let output = tch::no_grad(|| self.module.forward_ts(&[input])).map_err(|err| {
            candle_core::Error::Msg(format!("Torchscript f0 forward failed: {}", err))
        })?;

        let output = output.to_device(TchDevice::Cpu).to_kind(TchKind::Float);
        let sizes = output.size();
        let (out_b, out_t) = match sizes.as_slice() {
            [ob, ot] => (*ob as usize, *ot as usize),
            [ob, ot, 1] => (*ob as usize, *ot as usize),
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Torchscript f0 output shape {:?} not supported",
                    sizes
                )))
            }
        };
        let out_len = out_b * out_t;
        let mut out_vec = vec![0f32; out_len];
        output
            .view([-1])
            .f_copy_data(&mut out_vec, out_len)
            .map_err(|err| {
                candle_core::Error::Msg(format!("Torchscript f0 copy failed: {}", err))
            })?;

        Tensor::from_vec(out_vec, (out_b, 1, out_t), mel.device())
    }
}

#[cfg(feature = "f0-libtorch")]
fn load_hift_f0_torchscript_from_env() -> Result<Option<TchF0Predictor>> {
    let path = match std::env::var("HIFT_F0_TORCHSCRIPT_PATH") {
        Ok(value) if !value.is_empty() => value,
        _ => return Ok(None),
    };
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!(
            "    [HiFT] Torchscript f0 path not found: {}",
            path.display()
        );
        return Ok(None);
    }
    Ok(Some(TchF0Predictor::new(path)?))
}

fn load_hift_s_stft_override_from_env(device: &Device) -> Result<Option<(Tensor, Tensor)>> {
    let path = match std::env::var("HIFT_S_STFT_OVERRIDE_PATH") {
        Ok(value) if !value.is_empty() => value,
        _ => return Ok(None),
    };
    let path = Path::new(&path);
    if !path.exists() {
        eprintln!(
            "    [HiFT] s_stft override path not found: {}",
            path.display()
        );
        return Ok(None);
    }

    let cpu = Device::Cpu;
    let mut tensors = candle_core::safetensors::load(path, &cpu)?;
    let mut real = match tensors.remove("s_stft_real") {
        Some(tensor) => tensor,
        None => {
            eprintln!(
                "    [HiFT] s_stft override missing s_stft_real in {}",
                path.display()
            );
            return Ok(None);
        }
    };
    let mut imag = match tensors.remove("s_stft_imag") {
        Some(tensor) => tensor,
        None => {
            eprintln!(
                "    [HiFT] s_stft override missing s_stft_imag in {}",
                path.display()
            );
            return Ok(None);
        }
    };

    real = match real.rank() {
        2 => real.unsqueeze(0)?,
        3 => real,
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "s_stft_real must be rank 2-3, got rank {}",
                real.rank()
            )))
        }
    };
    imag = match imag.rank() {
        2 => imag.unsqueeze(0)?,
        3 => imag,
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "s_stft_imag must be rank 2-3, got rank {}",
                imag.rank()
            )))
        }
    };

    if real.shape() != imag.shape() {
        if real.rank() == 3 && imag.rank() == 3 {
            let (rb, rf, rt) = real.dims3()?;
            let (ib, ifreq, it) = imag.dims3()?;
            if rb == ib && rf == it && rt == ifreq {
                imag = imag.transpose(1, 2)?;
            }
        }
    }
    if real.shape() != imag.shape() {
        return Err(candle_core::Error::Msg(format!(
            "s_stft_real and s_stft_imag shape mismatch: {:?} vs {:?}",
            real.shape(),
            imag.shape()
        )));
    }

    Ok(Some((real.to_device(device)?, imag.to_device(device)?)))
}

/// Snake Activation: x + (1/alpha) * sin^2(alpha * x)
/// Actually, the paper usually uses: x + (1/alpha) * sin(alpha * x)^2 ?
/// BigVGAN implementation: x + (1 / alpha) * sin(alpha * x) ^ 2
/// Check Python code:
/// Snake(channels, alpha_logscale=False)
/// forward:
///   norm = torch.exp(log_alpha) if log_scale else alpha
///   x + (1/norm) * torch.sin(norm * x).pow(2)
pub struct Snake {
    alpha: Tensor,
}

impl Snake {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = if vb.contains_tensor("alpha") {
            let a = vb.get(channels, "alpha")?;
            a.reshape((1, channels, 1))?
        } else {
            vb.get((1, channels, 1), "alpha")?
        };
        Ok(Self { alpha })
    }
}

impl Module for Snake {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [Batch, Channels, Time]
        // alpha: [1, Channels, 1]
        let norm = &self.alpha;
        let one_over_norm = (norm.recip())?;
        let scaled_x = xs.broadcast_mul(norm)?;
        let sin_sq = scaled_x.sin()?.sqr()?;

        xs + one_over_norm.broadcast_mul(&sin_sq)?
    }
}

pub struct SineGen {
    harmonic_num: usize,
    sine_amp: f64,
    noise_std: f64,
    sampling_rate: f64,
    voiced_threshold: f32,
    upsample_scale: usize,
    use_interpolated: bool,
    causal: bool,
    rand_ini: Option<Vec<f32>>,
    sine_noise_cache: Option<Tensor>,
}

impl SineGen {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        upsample_scale: usize,
        causal: bool,
        rand_ini_override: Option<Vec<f32>>,
        sine_noise_cache: Option<Tensor>,
        _vb: VarBuilder,
    ) -> Result<Self> {
        let use_interpolated = sampling_rate != 22050;
        let num = harmonic_num + 1;
        let mut rand_ini = if causal { rand_ini_override } else { None };
        if let Some(values) = rand_ini.as_ref() {
            if values.len() != num {
                eprintln!(
                    "    [SineGen] rand_ini length {} does not match {}, ignoring override.",
                    values.len(),
                    num
                );
                rand_ini = None;
            }
        }
        if causal && rand_ini.is_none() {
            let mut rng = rand::thread_rng();
            let mut values = Vec::with_capacity(num);
            for h in 0..num {
                if h == 0 {
                    values.push(0.0);
                } else {
                    values.push(rng.gen::<f32>());
                }
            }
            rand_ini = Some(values);
        }
        let sine_noise_cache = if causal { sine_noise_cache } else { None };
        Ok(Self {
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate: sampling_rate as f64,
            voiced_threshold,
            upsample_scale: upsample_scale.max(1),
            use_interpolated,
            causal,
            rand_ini,
            sine_noise_cache,
        })
    }

    fn cached_sine_noise(
        &self,
        b: usize,
        l: usize,
        num_harmonics: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let cache = match &self.sine_noise_cache {
            Some(cache) => cache,
            None => return Ok(None),
        };
        let mut cache = cache.to_device(device)?;
        if cache.rank() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "sine_noise_cache must be 3D, got rank {}",
                cache.rank()
            )));
        }
        let (cb, c1, c2) = cache.dims3()?;
        if c2 == num_harmonics {
            if c1 < l {
                return Err(candle_core::Error::Msg(format!(
                    "sine_noise_cache too short: need {}, have {}",
                    l, c1
                )));
            }
            cache = cache.narrow(1, 0, l)?;
            cache = cache.transpose(1, 2)?;
        } else if c1 == num_harmonics {
            if c2 < l {
                return Err(candle_core::Error::Msg(format!(
                    "sine_noise_cache too short: need {}, have {}",
                    l, c2
                )));
            }
            cache = cache.narrow(2, 0, l)?;
        } else {
            return Err(candle_core::Error::Msg(format!(
                "sine_noise_cache dims do not match num_harmonics {} (got {:?})",
                num_harmonics,
                (c1, c2)
            )));
        }

        if cb == 1 && b > 1 {
            cache = cache.repeat((b, 1, 1))?;
        } else if cb != b {
            return Err(candle_core::Error::Msg(format!(
                "sine_noise_cache batch mismatch: cache {}, input {}",
                cb, b
            )));
        }

        Ok(Some(cache))
    }

    // forward(f0) -> (sine_waves, uv, noise)
    // f0: [Batch, Length, 1] (Wait, Python input is usually [Batch, Length, 1] or [Batch, 1, Length]?)
    // Python SineGen forward(f0): f0 is [Batch, 1, Length] (implied by transpose(1,2) at start of forward)
    // Wait, Python: f0 = f0.transpose(1, 2) -> [Batch, Length, 1] ?
    // No. PyTorch Transpose swaps dims.
    // If input is [Batch, 1, Length], transpose(1,2) -> [Batch, Length, 1].
    // Then F_mat init: [Batch, Harm+1, Length].
    // Wait, Python code:
    // f0 = f0.transpose(1, 2)
    // F_mat = zeros((batch, harm+1, f0.size(-1))) -> [Batch, Harm, 1] ?
    // No, f0.size(-1) is now 1.
    // Line 167: F_mat[:, i: i+1, :] = f0 * (i+1) ...
    // This implies f0 is [Batch, Length, 1].
    // Let's assume input f0 to `SineGen.forward` in Python is [Batch, 1, Length].
    // And internal calculation transposes it.

    // We will stick to tensor shapes [Batch, Channels, Length] for Candle Conv1d compatibility.
    // f0 input: [Batch, 1, Length].
    /// Forward pass with optional noise injection for parity testing.
    ///
    /// # Arguments
    /// * `f0` - Fundamental frequency [Batch, 1, Length]
    /// * `phase_inject` - Optional pre-generated harmonic phases [Batch, harmonic_num, Length] for parity testing
    /// * `noise_inject` - Optional pre-generated noise [Batch, harmonic_num+1, Length] for parity testing
    pub fn forward(
        &self,
        f0: &Tensor,
        _phase_inject: Option<&Tensor>,
        noise_inject: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // f0: [Batch, 1, Length] (or [Batch, Length, 1])
        let f0_2d = if f0.dim(1)? == 1 {
            f0.squeeze(1)?
        } else if f0.dim(2)? == 1 {
            f0.squeeze(2)?
        } else {
            return Err(candle_core::Error::Msg(
                "SineGen expects f0 with a singleton channel dimension".into(),
            ));
        };
        let (b, l) = f0_2d.dims2()?;
        let f0_cpu = f0_2d.to_device(&Device::Cpu)?;
        let f0_vec = f0_cpu.flatten_all()?.to_vec1::<f32>()?;

        let num_harmonics = self.harmonic_num + 1;
        let mut sine_waves_vec = vec![0f32; b * num_harmonics * l];

        let sampling_rate = self.sampling_rate as f32;
        let two_pi = 2.0 * std::f32::consts::PI;
        let sine_amp = self.sine_amp as f32;

        if self.use_interpolated {
            if l % self.upsample_scale != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "SineGen2 expects length {} divisible by upsample_scale {}",
                    l, self.upsample_scale
                )));
            }
            let down_len = l / self.upsample_scale;
            let scale = l as f32 / down_len as f32;
            let mut rng = rand::thread_rng();

            for b_idx in 0..b {
                let offset_f0 = b_idx * l;
                for h in 0..num_harmonics {
                    let mult = (h + 1) as f32;
                    let mut rad_values = vec![0f32; l];
                    for t in 0..l {
                        let current_f0 = f0_vec[offset_f0 + t];
                        let rad = (current_f0 * mult / sampling_rate).rem_euclid(1.0);
                        rad_values[t] = rad;
                    }

                    let rand_ini = if self.causal {
                        self.rand_ini
                            .as_ref()
                            .and_then(|vals| vals.get(h))
                            .cloned()
                            .unwrap_or(0.0)
                    } else if h == 0 {
                        0.0
                    } else {
                        rng.gen::<f32>()
                    };
                    if rand_ini != 0.0 {
                        rad_values[0] += rand_ini;
                    }

                    let mut rad_down = vec![0f32; down_len];
                    for out_idx in 0..down_len {
                        let in_index = (out_idx as f32 + 0.5) * scale - 0.5;
                        let i0 = in_index.floor();
                        let i1 = i0 + 1.0;
                        let w = in_index - i0;
                        let idx0 = if i0 < 0.0 {
                            0
                        } else if i0 as usize >= l {
                            l - 1
                        } else {
                            i0 as usize
                        };
                        let idx1 = if i1 < 0.0 {
                            0
                        } else if i1 as usize >= l {
                            l - 1
                        } else {
                            i1 as usize
                        };
                        let v0 = rad_values[idx0];
                        let v1 = rad_values[idx1];
                        rad_down[out_idx] = v0 + (v1 - v0) * w;
                    }

                    let mut phase = 0f32;
                    for down_idx in 0..down_len {
                        phase += rad_down[down_idx];
                        let phase_val = phase * two_pi * self.upsample_scale as f32;
                        let start = down_idx * self.upsample_scale;
                        for t in 0..self.upsample_scale {
                            let idx = start + t;
                            let dst = (b_idx * num_harmonics + h) * l + idx;
                            sine_waves_vec[dst] = phase_val.sin() * sine_amp;
                        }
                    }
                }
            }
        } else {
            for b_idx in 0..b {
                let offset_f0 = b_idx * l;
                for h in 0..num_harmonics {
                    let mult = (h + 1) as f32;
                    let mut running_phase = 0.0f32;
                    for t in 0..l {
                        let current_f0 = f0_vec[offset_f0 + t];
                        let phase_step = current_f0 * mult / sampling_rate;
                        running_phase += phase_step;
                        running_phase %= 1.0;
                        let val = (running_phase * two_pi).sin() * sine_amp;
                        let idx = (b_idx * num_harmonics + h) * l + t;
                        sine_waves_vec[idx] = val;
                    }
                }
            }
        }

        let sine_waves = Tensor::from_vec(sine_waves_vec, (b, num_harmonics, l), f0.device())?;

        // 4. UV
        let uv = f0.gt(self.voiced_threshold as f64)?.to_dtype(DType::F32)?;

        // 5. Noise
        let term1 = (&uv * self.noise_std)?;
        let ones = Tensor::ones_like(&uv)?;
        let term2 = ((&ones - &uv)? * (self.sine_amp / 3.0))?;
        let noise_amp = (term1 + term2)?;

        // Use injected noise or generate random noise
        let noise = match noise_inject {
            Some(n) => n.broadcast_mul(&noise_amp)?,
            None => {
                if self.causal {
                    if let Some(cache) =
                        self.cached_sine_noise(b, l, num_harmonics, f0.device())?
                    {
                        cache.broadcast_mul(&noise_amp)?
                    } else {
                        Tensor::randn_like(&sine_waves, 0.0, 1.0)?
                            .broadcast_mul(&noise_amp)?
                    }
                } else {
                    Tensor::randn_like(&sine_waves, 0.0, 1.0)?
                        .broadcast_mul(&noise_amp)?
                }
            }
        };

        // 6. Merge
        let output = ((sine_waves.broadcast_mul(&uv))? + noise.clone())?;

        Ok((output.transpose(1, 2)?, uv.transpose(1, 2)?, noise))
    }
}

fn causal_padding(kernel_size: usize, dilation: usize) -> usize {
    let numerator = kernel_size * dilation - dilation;
    (numerator / 2) * 2 + (kernel_size + 1) % 2
}

fn load_conv1d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> Result<Conv1d> {
    let weight = if vb.contains_tensor("weight_g") {
        let g = vb.get((out_c, 1, 1), "weight_g")?;
        let v = vb.get((out_c, in_c, k), "weight_v")?;
        let norm_v = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm_v)?;
        v.broadcast_mul(&scale)?
    } else if vb.contains_tensor("weight.original0") {
        let g = vb.get((out_c, 1, 1), "weight.original0")?;
        let v = vb.get((out_c, in_c, k), "weight.original1")?;
        let norm_v = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm_v)?;
        v.broadcast_mul(&scale)?
    } else if vb.contains_tensor("parametrizations.weight.original0") {
        let g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
        let v = vb.get((out_c, in_c, k), "parametrizations.weight.original1")?;
        let norm_v = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm_v)?;
        v.broadcast_mul(&scale)?
    } else {
        vb.get((out_c, in_c, k), "weight")?
    };

    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };

    let cfg = Conv1dConfig {
        padding,
        stride,
        dilation,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    Ok(Conv1d::new(weight, bias, cfg))
}

struct PaddedConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl PaddedConv1d {
    fn new(conv: Conv1d, pad_left: usize, pad_right: usize) -> Self {
        Self {
            conv,
            pad_left,
            pad_right,
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if self.pad_left == 0 && self.pad_right == 0 {
            return self.conv.forward(xs);
        }
        let padded = xs.pad_with_zeros(2, self.pad_left, self.pad_right)?;
        self.conv.forward(&padded)
    }
}

pub struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: candle_nn::Linear,
    causal: bool,
    noise_cache: Option<Tensor>,
}

impl SourceModuleHnNSF {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        upsample_scale: usize,
        causal: bool,
        vb: VarBuilder,
        parity_seeds: Option<HiftParitySeeds>,
    ) -> Result<Self> {
        let (rand_ini_override, sine_noise_cache, source_noise_cache) = match parity_seeds {
            Some(seeds) => (
                seeds.rand_ini,
                seeds.sine_noise_cache,
                seeds.source_noise_cache,
            ),
            None => (None, None, None),
        };

        let sine_gen = SineGen::new(
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate,
            voiced_threshold,
            upsample_scale,
            causal,
            rand_ini_override,
            sine_noise_cache,
            vb.clone(),
        )?;
        // l_linear: Linear(harmonic_num + 1, 1)
        // Python: nn.Linear(harmonic_num + 1, 1)
        // Check if weight_norm is on logic? generator.py: SourceModuleHnNSF uses regular Linear.
        let l_linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        let noise_cache = if causal {
            if let Some(cache) = source_noise_cache {
                Some(cache)
            } else {
                let mut rng = rand::thread_rng();
                let len = sampling_rate.saturating_mul(300);
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(rng.gen::<f32>());
                }
                Some(Tensor::from_vec(values, (1, len, 1), &Device::Cpu)?)
            }
        } else {
            None
        };

        Ok(Self {
            sine_gen,
            l_linear,
            causal,
            noise_cache,
        })
    }

    fn forward_with_sine(
        &self,
        f0: &Tensor,
        phase_inject: Option<&Tensor>,
        sine_noise_inject: Option<&Tensor>,
        source_noise_inject: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (sine_wavs, uv, _) = self.sine_gen.forward(f0, phase_inject, sine_noise_inject)?;

        // sine_merge = tanh(linear(sine_waves))
        // sine_wavs: [Batch, Length, Harmonics+1]
        let sine_merge = self.l_linear.forward(&sine_wavs)?;
        let sine_merge = sine_merge.tanh()?;
        // Transpose to [Batch, 1, Length]
        let sine_merge = sine_merge.transpose(1, 2)?;

        // Noise branch - use injected noise or generate random noise
        let noise = match source_noise_inject {
            Some(n) => (n * (self.sine_gen.sine_amp / 3.0))?,
            None if self.causal => {
                let len = uv.dim(1)?;
                let cache = self.noise_cache.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("Missing causal noise cache".into())
                })?;
                let mut noise_tensor = cache.to_device(uv.device())?;
                if noise_tensor.rank() == 2 {
                    noise_tensor = noise_tensor.unsqueeze(2)?;
                }
                if noise_tensor.rank() != 3 {
                    return Err(candle_core::Error::Msg(format!(
                        "Noise cache must be 3D, got rank {}",
                        noise_tensor.rank()
                    )));
                }
                let (cb, cl, cc) = noise_tensor.dims3()?;
                if cc != 1 {
                    return Err(candle_core::Error::Msg(format!(
                        "Noise cache expected last dim 1, got {}",
                        cc
                    )));
                }
                if len > cl {
                    return Err(candle_core::Error::Msg(format!(
                        "Noise cache too small: need {}, have {}",
                        len, cl
                    )));
                }
                noise_tensor = noise_tensor.narrow(1, 0, len)?;
                if cb == 1 && uv.dim(0)? > 1 {
                    noise_tensor = noise_tensor.repeat((uv.dim(0)?, 1, 1))?;
                } else if cb != uv.dim(0)? {
                    return Err(candle_core::Error::Msg(format!(
                        "Noise cache batch mismatch: cache {}, input {}",
                        cb,
                        uv.dim(0)?
                    )));
                }
                let noise_tensor = noise_tensor.broadcast_as(uv.shape())?;
                (noise_tensor * (self.sine_gen.sine_amp / 3.0))?
            }
            None => (Tensor::randn_like(&uv, 0.0, 1.0)? * (self.sine_gen.sine_amp / 3.0))?,
        };

        Ok((sine_merge, noise, uv, sine_wavs))
    }

    /// Forward pass with optional noise injection for parity testing.
    ///
    /// # Arguments
    /// * `f0` - Fundamental frequency [Batch, 1, Length]
    /// * `phase_inject` - Optional pre-generated harmonic phases for parity testing
    /// * `sine_noise_inject` - Optional pre-generated sine noise for parity testing
    /// * `source_noise_inject` - Optional pre-generated source noise for parity testing
    pub fn forward(
        &self,
        f0: &Tensor,
        phase_inject: Option<&Tensor>,
        sine_noise_inject: Option<&Tensor>,
        source_noise_inject: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (sine_merge, noise, uv, _sine_wavs) =
            self.forward_with_sine(f0, phase_inject, sine_noise_inject, source_noise_inject)?;
        Ok((sine_merge, noise, uv))
    }
}

pub struct ResBlock {
    convs1: Vec<PaddedConv1d>,
    convs2: Vec<PaddedConv1d>,
    acti1: Vec<Snake>,
    acti2: Vec<Snake>,
}

impl ResBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut acti1 = Vec::new();
        let mut acti2 = Vec::new();

        let vb_c1 = vb.pp("convs1");
        let vb_c2 = vb.pp("convs2");
        let vb_a1 = vb.pp("activations1");
        let vb_a2 = vb.pp("activations2");

        for (i, &dil) in dilations.iter().enumerate() {
            let pad1 = causal_padding(kernel_size, dil);
            let c1 = load_conv1d(vb_c1.pp(i), channels, channels, kernel_size, 1, dil, 0)?;
            convs1.push(PaddedConv1d::new(c1, pad1, 0));

            let pad2 = causal_padding(kernel_size, 1);
            let c2 = load_conv1d(vb_c2.pp(i), channels, channels, kernel_size, 1, 1, 0)?;
            convs2.push(PaddedConv1d::new(c2, pad2, 0));

            // acti
            acti1.push(Snake::new(channels, vb_a1.pp(i))?);
            acti2.push(Snake::new(channels, vb_a2.pp(i))?);
        }

        Ok(Self {
            convs1,
            convs2,
            acti1,
            acti2,
        })
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for i in 0..self.convs1.len() {
            let xt = self.acti1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.acti2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

pub struct F0Predictor {
    condnet: Vec<Conv1d>, // Sequential with ELU
    classifier: candle_nn::Linear,
}

impl F0Predictor {
    pub fn new(in_channels: usize, cond_channels: usize, vb: VarBuilder) -> Result<Self> {
        // condnet: 5 layers of Conv1d + WeightNorm
        let mut condnet = Vec::new();
        let vb_net = vb.pp("condnet");

        // 0: Conv1d(4, causal_type='right')
        // 2, 4, 6, 8: Conv1d(3, causal_type='left')
        for i in 0..5 {
            let in_c = if i == 0 { in_channels } else { cond_channels };
            let vb_layer = vb_net.pp(i * 2);
            let k = if i == 0 { 4 } else { 3 };
            // We use padding=0 here and pad manually in forward to achieve causal behavior
            let layer = load_conv1d(vb_layer, in_c, cond_channels, k, 1, 1, 0)?;

            // Debug weight stats
            if i == 0 {
                let w = layer.weight();
                if let Ok(flat) = w.flatten_all() {
                    if let Ok(vec) = flat.to_vec1::<f32>() {
                         let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                         let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                         let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                         eprintln!("    [F0Predictor] Layer 0 weight (after norm): min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
                    }
                }

                if let Some(bias) = layer.bias() {
                    if let Ok(flat) = bias.flatten_all() {
                        if let Ok(vec) = flat.to_vec1::<f32>() {
                            let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                            eprintln!("    [F0Predictor] Layer 0 bias: mean={:.6}", mean);
                        }
                    }
                }
            }

            condnet.push(layer);
        }

        // Classifier
        let classifier = candle_nn::linear(cond_channels, 1, vb.pp("classifier"))?;

        Ok(Self {
            condnet,
            classifier,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [Batch, 80, Time]
        eprintln!("  [F0Predictor] input shape: {:?}", x.shape());

        let mut h = x.clone();
        for (i, conv) in self.condnet.iter().enumerate() {
            if i == 0 {
                // F0 predictor in CosyVoice3.0 uses kernel 4, causal type 'right'
                // This means pad 3 on RIGHT.
                h = h.pad_with_zeros(2, 0, 3)?;
            } else {
                // Subsequent ones are kernel 3, causal 'left'
                // This means pad 2 on LEFT.
                h = h.pad_with_zeros(2, 2, 0)?;
            }
            h = conv.forward(&h)?;
            h = h.elu(1.0)?; // ELU

            // Debug each layer
            if let Ok(flat) = h.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = vec.iter().sum();
                    let mean = sum / vec.len() as f32;
                    eprintln!("    Layer {} after ELU: min={:.6}, max={:.6}, mean={:.6}", i, min, max, mean);
                }
            }
        }
        // h: [Batch, Cond, Time] -> transpose -> [Batch, Time, Cond]
        let h_t = h.transpose(1, 2)?;
        let out = if std::env::var("HIFT_F0_MANUAL_LINEAR")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            self.linear_manual(&h_t)?
        } else {
            self.classifier.forward(&h_t)?
        };

        // Debug classifier output
        if let Ok(flat) = out.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!("    Classifier out (pre-abs): min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
        }

        // out: [Batch, Time, 1]
        let f0 = out.transpose(1, 2)?.abs()?; // [Batch, 1, Time]
        Ok(f0)
    }

    pub fn forward_with_debug(
        &self,
        x: &Tensor,
        debug: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        // x: [Batch, 80, Time]
        eprintln!("  [F0Predictor] input shape: {:?}", x.shape());

        let mut h = x.clone();
        for (i, conv) in self.condnet.iter().enumerate() {
            if i == 0 {
                // F0 predictor in CosyVoice3.0 uses kernel 4, causal type 'right'
                // This means pad 3 on RIGHT.
                h = h.pad_with_zeros(2, 0, 3)?;
            } else {
                // Subsequent ones are kernel 3, causal 'left'
                // This means pad 2 on LEFT.
                h = h.pad_with_zeros(2, 2, 0)?;
            }
            h = conv.forward(&h)?;
            h = h.elu(1.0)?; // ELU
            debug.insert(format!("f0_layer{}", i), h.clone());

            if let Ok(flat) = h.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = vec.iter().sum();
                    let mean = sum / vec.len() as f32;
                    eprintln!("    Layer {} after ELU: min={:.6}, max={:.6}, mean={:.6}", i, min, max, mean);
                }
            }
        }
        // h: [Batch, Cond, Time] -> transpose -> [Batch, Time, Cond]
        let h_t = h.transpose(1, 2)?;
        let out = if std::env::var("HIFT_F0_MANUAL_LINEAR")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            self.linear_manual(&h_t)?
        } else {
            self.classifier.forward(&h_t)?
        };
        debug.insert("f0_classifier_pre_abs".to_string(), out.clone());

        if let Ok(flat) = out.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!("    Classifier out (pre-abs): min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
        }

        // out: [Batch, Time, 1]
        let f0 = out.transpose(1, 2)?.abs()?; // [Batch, 1, Time]
        Ok(f0)
    }

    fn linear_manual(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let (out_c, in_c) = self.classifier.weight().dims2()?;
        if out_c != 1 || in_c != c {
            return self.classifier.forward(x);
        }
        if !x.device().is_cpu() {
            return self.classifier.forward(x);
        }

        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let w_vec = self.classifier.weight().flatten_all()?.to_vec1::<f32>()?;
        let bias = if let Some(bias) = self.classifier.bias() {
            let b_vec = bias.flatten_all()?.to_vec1::<f32>()?;
            b_vec.get(0).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let mut out = vec![0f32; b * t];
        for bi in 0..b {
            for ti in 0..t {
                let mut acc = bias;
                let base = (bi * t + ti) * c;
                for ci in 0..c {
                    acc += x_vec[base + ci] * w_vec[ci];
                }
                out[bi * t + ti] = acc;
            }
        }

        Tensor::from_vec(out, (b, t, 1), x.device())
    }
}

pub struct HiFTGenerator {
    conv_pre: PaddedConv1d,
    ups: Vec<Conv1d>,
    ups_rates: Vec<usize>,
    source_downs: Vec<PaddedConv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: PaddedConv1d,
    f0_predictor: F0Predictor,
    stft: crate::utils::InverseStftModule,
    analysis_stft: crate::utils::StftModule,
    f0_upsamp_scale: usize,
    num_kernels: usize,
    m_source: SourceModuleHnNSF,
    #[cfg(feature = "f0-libtorch")]
    f0_predictor_tch: Option<TchF0Predictor>,
    #[cfg(not(feature = "f0-libtorch"))]
    f0_predictor_tch: Option<()>,
}

impl HiFTGenerator {
    pub fn new(vb: VarBuilder, config: &HiFTConfig) -> Result<Self> {
        let ups_rates = &config.upsample_rates;
        let ups_kernels = &config.upsample_kernel_sizes;
        let base_ch = config.base_channels;

        // Conv Pre
        // Fun-CosyVoice3-0.5B has kernel_size=5, padding=2
        let conv_pre_k = if vb.pp("conv_pre").contains_tensor("weight.original1")
            || vb.pp("conv_pre")
                .contains_tensor("parametrizations.weight.original1")
        {
            5
        } else {
            13
        };
        let conv_pre_raw = load_conv1d(
            vb.pp("conv_pre"),
            config.in_channels,
            base_ch,
            conv_pre_k,
            1,
            1,
            0,
        )?;
        let conv_pre = if conv_pre_k == 5 {
            PaddedConv1d::new(conv_pre_raw, 0, conv_pre_k - 1)
        } else {
            let pad = conv_pre_k / 2;
            PaddedConv1d::new(conv_pre_raw, pad, pad)
        };

        // Ups
        let mut ups = Vec::new();
        let vb_ups = vb.pp("ups");
        for (i, (&_u, &k)) in ups_rates.iter().zip(ups_kernels).enumerate() {
            let in_c = base_ch / (1 << i);
            let out_c = base_ch / (1 << (i + 1));
            // CausalConv1dUpsample is Upsample + CausalConv1d with padding k-1
            let conv = load_conv1d(vb_ups.pp(i), in_c, out_c, k, 1, 1, 0)?;
            ups.push(conv);
        }

        // Source Downs & ResBlocks
        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();
        let vb_sd = vb.pp("source_downs");
        let vb_sr = vb.pp("source_resblocks");

        // Downsample rates construction
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // downsample_cum_rates logic:
        // i=0: 1. i=1: 8. i=2: 64...
        // Iterating logic matches len(ups).
        // Let's assume standard HiFT logic:
        // For i in 0..len(ups):
        //   u = cum_rates[len-1-i]
        //   kernel/dilation from lists

        // Replicating generator.py logic:
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // cum = cumprod(downsample_rates)
        // iter over zip(cum[::-1], k_list, d_list)

        // Let's implement this logic in Rust:
        let mut ds_rates = vec![1];
        let mut rev_ups = ups_rates.clone();
        rev_ups.reverse();
        if !rev_ups.is_empty() {
            ds_rates.extend_from_slice(&rev_ups[..rev_ups.len() - 1]);
        }

        let mut cum_rates = vec![];
        let mut acc = 1;
        for r in ds_rates {
            acc *= r;
            cum_rates.push(acc);
        }
        cum_rates.reverse(); // [64, 8, 1] e.g.

        let src_k = &config.source_resblock_kernel_sizes;
        let src_d = &config.source_resblock_dilation_sizes;

        for (i, ((&u, &k), d)) in cum_rates.iter().zip(src_k).zip(src_d).enumerate() {
            let ch = base_ch / (1 << (i + 1));
            let in_ch = config.istft_params_n_fft + 2; // nfft+2 channels source

            // source_down
            let sd = if u == 1 {
                let conv = load_conv1d(vb_sd.pp(i), in_ch, ch, 1, 1, 1, 0)?;
                PaddedConv1d::new(conv, 0, 0)
            } else {
                let conv = load_conv1d(vb_sd.pp(i), in_ch, ch, u * 2, u, 1, 0)?;
                PaddedConv1d::new(conv, u.saturating_sub(1), 0)
            };
            source_downs.push(sd);

            // source_resblock
            let sb = ResBlock::new(ch, k, d, vb_sr.pp(i))?;
            source_resblocks.push(sb);
        }

        // ResBlocks
        let mut resblocks = Vec::new();
        let vb_rb = vb.pp("resblocks");
        // i loops over ups
        let num_ups = ups_rates.len();
        let num_kernels = config.resblock_kernel_sizes.len();

        for i in 0..num_ups {
            let ch = base_ch / (1 << (i + 1));
            for (j, (&k, d)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(&config.resblock_dilation_sizes)
                .enumerate()
            {
                let idx = i * num_kernels + j;
                let rb = ResBlock::new(ch, k, d, vb_rb.pp(idx))?;
                resblocks.push(rb);
            }
        }

        // Post
        let last_ch = base_ch / (1 << num_ups);
        let conv_post_raw = load_conv1d(
            vb.pp("conv_post"),
            last_ch,
            config.istft_params_n_fft + 2,
            7,
            1,
            1,
            0,
        )?;
        let conv_post = PaddedConv1d::new(conv_post_raw, causal_padding(7, 1), 0);

        let stft = crate::utils::InverseStftModule::new(
            config.istft_params_n_fft,
            config.istft_params_hop_len,
            vb.device(),
        )?;
        let analysis_stft = crate::utils::StftModule::new(
            config.istft_params_n_fft,
            config.istft_params_hop_len,
            vb.device(),
        )?;

        let f0_upsamp_scale = ups_rates.iter().product::<usize>() * config.istft_params_hop_len;

        // Source
        let parity_seeds = load_hift_parity_seeds_from_env()?;
        let m_source = SourceModuleHnNSF::new(
            config.nb_harmonics,
            0.1,
            0.003,
            config.sampling_rate,
            config.voiced_threshold,
            f0_upsamp_scale,
            true,
            vb.pp("m_source"),
            parity_seeds,
        )?;

        let f0_predictor = F0Predictor::new(config.in_channels, base_ch, vb.pp("f0_predictor"))?;
        #[cfg(feature = "f0-libtorch")]
        let f0_predictor_tch = load_hift_f0_torchscript_from_env()?;
        #[cfg(not(feature = "f0-libtorch"))]
        let f0_predictor_tch = None;

        Ok(Self {
            conv_pre,
            ups,
            ups_rates: ups_rates.clone(),
            resblocks,
            source_downs,
            source_resblocks,
            num_kernels,
            conv_post,
            stft,
            analysis_stft,
            m_source,
            f0_predictor,
            f0_upsamp_scale,
            f0_predictor_tch,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // mel: [Batch, 80, Length]
        eprintln!("\n  [HiFT.forward] Starting...");
        eprintln!("    input mel shape: {:?}", mel.shape());

        // Print mel stats
        if let Ok(mel_flat) = mel.flatten_all() {
            if let Ok(mel_vec) = mel_flat.to_vec1::<f32>() {
                let min = mel_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mel_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = mel_vec.iter().sum();
                let mean = sum / mel_vec.len() as f32;
                eprintln!("    input mel stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
        }

        // 1. F0 Predictor
        let f0 = if let Some(f0_override) = load_hift_f0_override_from_env(mel.device())? {
            eprintln!("    [HiFT] Using f0 override from HIFT_F0_OVERRIDE_PATH");
            f0_override
        } else {
            #[cfg(feature = "f0-libtorch")]
            {
                if let Some(f0_tch) = self.f0_predictor_tch.as_ref() {
                    eprintln!("    [HiFT] Using torchscript f0 predictor");
                    f0_tch.forward(mel)?
                } else {
                    self.f0_predictor.forward(mel)?
                }
            }
            #[cfg(not(feature = "f0-libtorch"))]
            {
                self.f0_predictor.forward(mel)?
            }
        }; // [Batch, 1, Length_f0]
        let mel_len = mel.dim(2)?;
        let f0 = f0.narrow(2, 0, mel_len)?; // crop to mel length
        eprintln!("    F0 predictor output shape: {:?}", f0.shape());

        // Print F0 stats
        if let Ok(f0_flat) = f0.flatten_all() {
            if let Ok(f0_vec) = f0_flat.to_vec1::<f32>() {
                let min = f0_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = f0_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = f0_vec.iter().sum();
                let mean = sum / f0_vec.len() as f32;
                eprintln!("    F0 stats: min={:.6} Hz, max={:.6} Hz, mean={:.6} Hz", min, max, mean);
            }
        }

        // 2. Upsample F0 to Source Resolution
        // Nearest neighbor upsample: [B, 1, L] -> [B, 1, L, Scale] -> [B, 1, L*Scale]
        let (b, c, l) = f0.dims3()?;
        let s = f0
            .unsqueeze(3)?
            .repeat((1, 1, 1, self.f0_upsamp_scale))?
            .reshape((b, c, l * self.f0_upsamp_scale))?;
        eprintln!("    upsampled f0 shape: {:?} (scale={})", s.shape(), self.f0_upsamp_scale);

        // 3. Source Module
        let (s_source, _, _) = self.m_source.forward(&s, None, None, None)?; // [B, 1, L_up]
        eprintln!("    source output shape: {:?}", s_source.shape());

        // Print source stats
        if let Ok(src_flat) = s_source.flatten_all() {
            if let Ok(src_vec) = src_flat.to_vec1::<f32>() {
                let min = src_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = src_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = src_vec.iter().sum();
                let mean = sum / src_vec.len() as f32;
                eprintln!("    source stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
        }

        // 4. Decode
        eprintln!("    Running decode...");
        let audio = self.decode(mel, &s_source)?;

        // Print final audio stats
        eprintln!("    [HiFT.forward] output shape: {:?}", audio.shape());
        if let Ok(audio_flat) = audio.flatten_all() {
            if let Ok(audio_vec) = audio_flat.to_vec1::<f32>() {
                let min = audio_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = audio_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = audio_vec.iter().sum();
                let mean = sum / audio_vec.len() as f32;
                eprintln!("    [HiFT.forward] audio stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);

                // Check for problems
                if min < -1.0 || max > 1.0 {
                    eprintln!("    ⚠️  ISSUE: Audio out of [-1, 1] range!");
                }
                if mean.abs() > 0.1 {
                    eprintln!("    ⚠️  ISSUE: Large DC offset: {}", mean);
                }
            }
        }

        Ok(audio)
    }

    pub fn forward_with_debug(&self, mel: &Tensor) -> Result<(Tensor, HashMap<String, Tensor>)> {
        let mut debug = HashMap::new();
        debug.insert("input_mel".to_string(), mel.clone());

        let f0 = if let Some(f0_override) = load_hift_f0_override_from_env(mel.device())? {
            eprintln!("    [HiFT] Using f0 override from HIFT_F0_OVERRIDE_PATH");
            f0_override
        } else {
            #[cfg(feature = "f0-libtorch")]
            {
                if let Some(f0_tch) = self.f0_predictor_tch.as_ref() {
                    eprintln!("    [HiFT] Using torchscript f0 predictor");
                    f0_tch.forward(mel)?
                } else {
                    self.f0_predictor.forward_with_debug(mel, &mut debug)?
                }
            }
            #[cfg(not(feature = "f0-libtorch"))]
            {
                self.f0_predictor.forward_with_debug(mel, &mut debug)?
            }
        };
        let mel_len = mel.dim(2)?;
        let f0 = f0.narrow(2, 0, mel_len)?;
        if let Ok(f0_no_channel) = f0.squeeze(1) {
            debug.insert("f0_output".to_string(), f0_no_channel);
        } else {
            debug.insert("f0_output".to_string(), f0.clone());
        }

        let (b, c, l) = f0.dims3()?;
        let s = f0
            .unsqueeze(3)?
            .repeat((1, 1, 1, self.f0_upsamp_scale))?
            .reshape((b, c, l * self.f0_upsamp_scale))?;
        debug.insert("source_s".to_string(), s.transpose(1, 2)?);

        let (sine_merge, noise, uv, sine_waves) =
            self.m_source.forward_with_sine(&s, None, None, None)?;
        debug.insert("sine_merge".to_string(), sine_merge.transpose(1, 2)?);
        debug.insert("sine_waves".to_string(), sine_waves.clone());
        debug.insert("noise".to_string(), noise.clone());
        debug.insert("uv".to_string(), uv.clone());
        if let Some(rand_ini) = self.m_source.sine_gen.rand_ini.as_ref() {
            let rand_ini_tensor =
                Tensor::from_vec(rand_ini.clone(), (1, rand_ini.len()), mel.device())?;
            debug.insert("rand_ini".to_string(), rand_ini_tensor);
        }
        if let Some(cache) = self.m_source.sine_gen.sine_noise_cache.as_ref() {
            let s_len = s.dim(2)?;
            let bsz = s.dim(0)?;
            let num_harmonics = self.m_source.sine_gen.harmonic_num + 1;
            let mut cache = cache.to_device(mel.device())?;
            if cache.rank() != 3 {
                return Err(candle_core::Error::Msg(format!(
                    "sine_noise_cache must be 3D, got rank {}",
                    cache.rank()
                )));
            }
            let (cb, c1, c2) = cache.dims3()?;
            let mut cache = if c2 == num_harmonics {
                if c1 < s_len {
                    return Err(candle_core::Error::Msg(format!(
                        "sine_noise_cache too short: need {}, have {}",
                        s_len, c1
                    )));
                }
                cache.narrow(1, 0, s_len)?
            } else if c1 == num_harmonics {
                if c2 < s_len {
                    return Err(candle_core::Error::Msg(format!(
                        "sine_noise_cache too short: need {}, have {}",
                        s_len, c2
                    )));
                }
                cache.narrow(2, 0, s_len)?.transpose(1, 2)?
            } else {
                return Err(candle_core::Error::Msg(format!(
                    "sine_noise_cache dims do not match num_harmonics {} (got {:?})",
                    num_harmonics,
                    (c1, c2)
                )));
            };
            if cb == 1 && bsz > 1 {
                cache = cache.repeat((bsz, 1, 1))?;
            } else if cb != bsz {
                return Err(candle_core::Error::Msg(format!(
                    "sine_noise_cache batch mismatch: cache {}, input {}",
                    cb, bsz
                )));
            }
            debug.insert("sine_noise_cache".to_string(), cache);
        }
        if let Some(cache) = self.m_source.noise_cache.as_ref() {
            let uv_len = uv.dim(1)?;
            let bsz = uv.dim(0)?;
            let mut cache = cache.to_device(mel.device())?;
            if cache.rank() == 2 {
                cache = cache.unsqueeze(2)?;
            }
            if cache.rank() != 3 {
                return Err(candle_core::Error::Msg(format!(
                    "source_noise_cache must be 3D, got rank {}",
                    cache.rank()
                )));
            }
            let (cb, cl, cc) = cache.dims3()?;
            if cc != 1 {
                return Err(candle_core::Error::Msg(format!(
                    "source_noise_cache expected last dim 1, got {}",
                    cc
                )));
            }
            if cl < uv_len {
                return Err(candle_core::Error::Msg(format!(
                    "source_noise_cache too short: need {}, have {}",
                    uv_len, cl
                )));
            }
            let mut cache = cache.narrow(1, 0, uv_len)?;
            if cb == 1 && bsz > 1 {
                cache = cache.repeat((bsz, 1, 1))?;
            } else if cb != bsz {
                return Err(candle_core::Error::Msg(format!(
                    "source_noise_cache batch mismatch: cache {}, input {}",
                    cb, bsz
                )));
            }
            debug.insert("source_noise_cache".to_string(), cache);
        }

        let audio = self.decode_internal(mel, &sine_merge, Some(&mut debug))?;
        let final_audio = if audio.rank() == 3 && audio.dim(1)? == 1 {
            audio.squeeze(1)?
        } else {
            audio.clone()
        };
        debug.insert("final_audio".to_string(), final_audio);

        Ok((audio, debug))
    }

    fn maybe_clamp_audio(&self, audio: Tensor) -> Result<Tensor> {
        let disable = std::env::var("HIFT_DISABLE_CLAMP")
            .map(|v| v != "0")
            .unwrap_or(false);
        if disable {
            return Ok(audio);
        }

        let audio_limit = 0.99f32;
        let min_val = Tensor::new(&[-audio_limit], audio.device())?;
        let max_val = Tensor::new(&[audio_limit], audio.device())?;
        let audio = audio.maximum(&min_val.broadcast_as(audio.shape())?)?;
        let audio = audio.minimum(&max_val.broadcast_as(audio.shape())?)?;
        Ok(audio)
    }

    fn decode(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        self.decode_internal(x, s, None)
    }

    fn decode_internal(
        &self,
        x: &Tensor,
        s: &Tensor,
        mut debug: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        // s STFT logic for fusion
        // s: [Batch, 1, Time]
        // We need STFT of s.
        // Wait, self.stft is INVERSE. We need ANALYSIS STFT.
        // I need to add `analysis_stft` to struct.
        // For now, let's assume I added it.
        let (s_real, s_imag) = if let Some((real, imag)) =
            load_hift_s_stft_override_from_env(s.device())?
        {
            eprintln!("    [HiFT] Using s_stft override from HIFT_S_STFT_OVERRIDE_PATH");
            (real, imag)
        } else {
            self.analysis_stft.transform(s)?
        }; // [Batch, Freq, Frames]
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("s_stft_real".to_string(), s_real.clone());
            debug_map.insert("s_stft_imag".to_string(), s_imag.clone());
        }
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?; // [Batch, 2*Freq, Frames]

        // x (mel): [Batch, 80, Length]
        let mut x = self.conv_pre.forward(x)?;
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("conv_pre_out".to_string(), x.clone());
        }

        let num_ups = self.ups.len();

        for i in 0..num_ups {
            x = candle_nn::ops::leaky_relu(&x, 0.1)?; // lrelu_slope=0.1
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("upsample_{}_pre", i), x.clone());
            }

            // Upsample
            let u = self.ups_rates[i];
            let (b, c, l) = x.dims3()?;
            x = x.unsqueeze(3)?
                .repeat((1, 1, 1, u))?
                .reshape((b, c, l * u))?;

            // Causal Padding k-1
            let k = self.ups[i].weight().dim(2)?;
            x = x.pad_with_zeros(2, k - 1, 0)?;
            x = self.ups[i].forward(&x)?;

            // Reflection-like padding at the end?
            // CausalHiFTGenerator.py: self.reflection_pad = nn.ReflectionPad1d((1, 0))
            if i == num_ups - 1 {
                let left = x.i((.., .., 1..2))?;
                x = Tensor::cat(&[&left, &x], 2)?;
            }
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("upsample_{}_out", i), x.clone());
            }

            // Fusion
            // si = source_downs[i](s_stft)
            let si_down = self.source_downs[i].forward(&s_stft)?;
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("source_down_{}_out", i), si_down.clone());
            }
            let si = self.source_resblocks[i].forward(&si_down)?;
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("source_resblock_{}_out", i), si.clone());
            }

            // Robust slice to handle boundary effects
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;
            let common_len = x_len.min(si_len);
            let x_slice = x.i((.., .., ..common_len))?;
            let si_slice = si.i((.., .., ..common_len))?;
            x = (x_slice + si_slice)?;
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("fusion_{}_out", i), x.clone());
            }

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let idx = i * self.num_kernels + j;
                let out = self.resblocks[idx].forward(&x)?;
                if let Some(debug_map) = debug.as_mut() {
                    debug_map.insert(format!("resblock_{}_{}_out", i, j), out.clone());
                }
                match xs {
                    None => xs = Some(out),
                    Some(prev) => xs = Some((prev + out)?),
                }
            }
            if let Some(val) = xs {
                x = (val / (self.num_kernels as f64))?;
            }
            if let Some(debug_map) = debug.as_mut() {
                debug_map.insert(format!("resblock_{}_out", i), x.clone());
            }
        }

        x = candle_nn::ops::leaky_relu(&x, 0.01)?; // Default slope matches PyTorch
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("post_lrelu_out".to_string(), x.clone());
        }
        x = self.conv_post.forward(&x)?;
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("conv_post_out".to_string(), x.clone());
        }

        // ISTFT Input: [B, n_fft+2, T]
        // Magnitude = exp(x[:, :mid])
        // Phase = sin(x[:, mid:])
        let dim = x.dim(1)?;
        let cutoff = dim / 2; // n_fft/2 + 1

        let mag_log = x.i((.., ..cutoff, ..))?;
        let phase_in = x.i((.., cutoff.., ..))?;

        // Debug conv_post output before exp
        if let Ok(flat) = mag_log.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!("    [decode] mag_log (before exp): min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
                if max > 50.0 {
                    eprintln!("      ⚠️  CRITICAL: mag_log has very large values, exp() will overflow!");
                }
            }
        }

        let magnitude = mag_log.exp()?;

        // Clip magnitude to prevent overflow - Python does: magnitude = torch.clip(magnitude, max=1e2)
        let max_mag = Tensor::new(&[100.0f32], magnitude.device())?;
        let magnitude = magnitude.minimum(&max_mag.broadcast_as(magnitude.shape())?)?;

        let phase = phase_in.sin()?;
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("magnitude".to_string(), magnitude.clone());
            debug_map.insert("phase".to_string(), phase.clone());
        }

        // Debug magnitude after exp
        if let Ok(flat) = magnitude.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!("    [decode] magnitude (after exp): min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
        }

        let audio = self.stft.forward(&magnitude, &phase)?;
        if let Some(debug_map) = debug.as_mut() {
            debug_map.insert("istft_audio".to_string(), audio.clone());
        }
        self.maybe_clamp_audio(audio)
    }
}

pub struct HiFTConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub sampling_rate: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub istft_params_n_fft: usize,
    pub istft_params_hop_len: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub source_resblock_kernel_sizes: Vec<usize>,
    pub source_resblock_dilation_sizes: Vec<Vec<usize>>,
    pub voiced_threshold: f32,
}

impl HiFTConfig {
    pub fn new(n_fft: usize) -> Self {
        Self {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 24000,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            istft_params_n_fft: n_fft,
            istft_params_hop_len: 4,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            source_resblock_kernel_sizes: vec![7, 7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            voiced_threshold: 10.0,
        }
    }

    pub fn fun_cosyvoice_3_0_5b(n_fft: usize) -> Self {
        Self {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 24000,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            istft_params_n_fft: n_fft,
            istft_params_hop_len: 4,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            source_resblock_kernel_sizes: vec![7, 7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            voiced_threshold: 5.0,
        }
    }
}

impl Default for HiFTConfig {
    fn default() -> Self {
        Self::new(16)
    }
}
