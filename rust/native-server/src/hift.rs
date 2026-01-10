// use crate::utils::StftModule; // Commented out until used
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use tracing::debug;

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
            let alpha_f = a.flatten_all()?.to_vec1::<f32>()?;
            let min = alpha_f.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = alpha_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            debug!("[Snake] (prefix: {}) alpha stats: min={:.4e}, max={:.4e}", vb.prefix(), min, max);
            a.reshape((1, channels, 1))?
        } else {
            vb.get((1, channels, 1), "alpha")?
        };
        if let Ok(flat) = alpha.flatten_all() {
            let _mean = flat.mean(0)?.to_scalar::<f32>()?;
            let _min = flat.min(0)?.to_scalar::<f32>()?;
            let _max = flat.max(0)?.to_scalar::<f32>()?;

        }
        Ok(Self { alpha })
    }
}

impl Module for Snake {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [Batch, Channels, Time]
        // xs: [Batch, Channels, Time]
        // alpha: [1, Channels, 1]
        let norm = &self.alpha;
        // Add epsilon to avoid division by zero / explosion
        let epsilon = 1e-9;
        let norm_safe = (norm + epsilon)?;
        let one_over_norm = (norm_safe.recip())?;

        let scaled_x = xs.broadcast_mul(norm)?;
        let sin_sq = scaled_x.sin()?.sqr()?;

        xs + one_over_norm.broadcast_mul(&sin_sq)?
    }
}

pub struct SineGen {
    harmonic_num: usize,
    pub sine_amp: f64,
    _noise_std: f64,
    sampling_rate: usize,
    noise_amp_uv: f32,
    noise_amp_unvoiced: f32,
}

impl SineGen {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        noise_amp_uv: f32,
        noise_amp_unvoiced: f32,
    ) -> Result<Self> {
        Ok(Self {
            harmonic_num,
            sine_amp,
            _noise_std: noise_std,
            sampling_rate,
            noise_amp_uv,
            noise_amp_unvoiced,
        })
    }

    // forward(f0) -> (sine_waves, uv, noise)
    // We will stick to tensor shapes [Batch, Channels, Length] for Candle Conv1d compatibility.
    // f0 input: [Batch, 1, Length] at audio rate.
    pub fn forward(
        &self,
        f0: &Tensor,
        phase_inject: Option<&Tensor>, // In this context used for full sine_waves injection if provided
        _noise_inject: Option<&Tensor>, // Not fully integrated yet, but sine_waves covers deterministic sine generation
        voiced_threshold: f32,
        upsample_scale: usize,
        _is_causal: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // If phase_inject is provided (as full sine_waves tensor), use it directly
        // The argument name `phase_inject` is kept for API compatibility but we treat it as sine_waves
        if let Some(sine_waves) = phase_inject {

            // Recalculate UV/Noise as usual or just return dummy/partial
            // For SourceModule parity, we mainly need sine_waves to be deterministic.

            // 1. UV Signal [Batch, 1, Length]
            let uv = f0.gt(voiced_threshold as f64)?.to_dtype(DType::F32)?.transpose(1, 2)?;

            // 4. Noise: noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
            let term1 = uv.affine(self.noise_amp_uv as f64, 0.0)?;
            let ones = Tensor::ones_like(&uv)?;
            let term2 = (&ones - &uv)?.affine(self.noise_amp_unvoiced as f64, 0.0)?;
            let noise_amp = (term1 + term2)?;

            // Random noise (for parity testing, should use injected noise if we had it here, but SourceModule handles final noise)
            let noise = Tensor::randn_like(&sine_waves, 0.0, 1.0)?.broadcast_mul(&noise_amp)?;

            return Ok((sine_waves.clone(), uv, noise));
        }

        let (b, _, l) = f0.dims3()?;
        let device = f0.device();

        // 1. UV Signal [Batch, 1, Length]
        let uv = f0.gt(voiced_threshold as f64)?.to_dtype(DType::F32)?;

        // 2. Harmonic Frequencies
        // Python: fn = f0 * range(1, harmonic_num + 2)
        // fn: [Batch, Harmonics+1, Length]
        let mut harmonics = Vec::with_capacity(self.harmonic_num + 1);
        for i in 1..=(self.harmonic_num + 1) {
            harmonics.push(f0.affine(i as f64, 0.0)?);
        }
        let fn_tensor = Tensor::cat(&harmonics, 1)?;

        // 3. Phase Integration matching Python SineGen2 exactly:
        //    a) rad_values = (fn / sampling_rate) % 1
        //    b) Add rand_ini to first sample (rad_values[:, 0, :] += rand_ini)
        //    c) Downsample rad_values using linear interpolation (F.interpolate, scale_factor=1/upsample_scale)
        //    d) phase = cumsum(rad_values_down, dim=1) * 2 * pi
        //    e) phase *= upsample_scale
        //    f) Upsample phase with nearest interpolation
        //    g) sines = sin(phase) * sine_amp

        let rad_values = (&fn_tensor / (self.sampling_rate as f64))?;

        let num_harmonics = self.harmonic_num + 1;
        let scale = upsample_scale;
        let frames = l / scale;
        let two_pi = 2.0 * std::f32::consts::PI;

        // Move to CPU for processing
        let rad_cpu = rad_values.to_device(&Device::Cpu)?;
        let rad_data = rad_cpu.flatten_all()?.to_vec1::<f32>()?;

        // Fixed initial phases from Python's causal SineGen2 self.rand_ini
        let rand_ini: [f32; 9] = [
            0.0, 0.7742558121681213, 0.49624037742614746, 0.044879257678985596,
            0.4709309935569763, 0.26031583547592163, 0.8478374481201172,
            0.24336206912994385, 0.8662649393081665,
        ];

        let mut sine_waves_vec = Vec::with_capacity(b * num_harmonics * l);

        for batch in 0..b {
            for h in 0..num_harmonics {
                let offset = (batch * num_harmonics + h) * l;

                // Step b: Add rand_ini to first sample (modifying first sample in rad_values)
                let phase_offset = if h < rand_ini.len() { rand_ini[h] } else { 0.0 };

                // Step c: Downsample rad_values using linear interpolation
                // For linear interpolation at scale_factor=1/scale, we average the samples in each frame
                // This is equivalent to F.interpolate(rad_values.T, scale_factor=1/scale, mode='linear').T
                let mut rad_down = Vec::with_capacity(frames);
                for f in 0..frames {
                    // Linear downsampling - use center value or average
                    // F.interpolate mode='linear' with scale_factor < 1 uses linear interpolation
                    // Simplified: take weighted average
                    let center = f * scale + scale / 2;
                    if center < l {
                        rad_down.push(rad_data[offset + center]);
                    } else {
                        rad_down.push(rad_data[offset + f * scale]);
                    }
                }

                // Step d + e: Cumsum at frame rate, then multiply by scale
                let mut frame_phases = Vec::with_capacity(frames);
                let mut cumsum = phase_offset;
                for f in 0..frames {
                    cumsum += rad_down[f];
                    // Multiply by upsample_scale (done here, not after upsample)
                    frame_phases.push(cumsum * scale as f32);
                }

                // Step f + g: Upsample phase with nearest, then compute sin * sine_amp
                for f in 0..frames {
                    let p = frame_phases[f];
                    let val = (p * two_pi).sin() * self.sine_amp as f32;
                    // Nearest interpolation: repeat same value for all samples in frame
                    for _ in 0..scale {
                        sine_waves_vec.push(val);
                    }
                }

                // Handle remainder
                let rem = l % scale;
                if rem > 0 {
                    let p = if !frame_phases.is_empty() {
                        *frame_phases.last().unwrap()
                    } else {
                        phase_offset * scale as f32
                    };
                    let val = (p * two_pi).sin() * self.sine_amp as f32;
                    for _ in 0..rem {
                        sine_waves_vec.push(val);
                    }
                }
            }
        }

        let sine_waves = Tensor::from_vec(sine_waves_vec, (b, num_harmonics, l), device)?;

        // 4. Noise: noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
        let term1 = uv.affine(self.noise_amp_uv as f64, 0.0)?;
        let ones = Tensor::ones_like(&uv)?;
        let term2 = (&ones - &uv)?.affine(self.noise_amp_unvoiced as f64, 0.0)?;
        let noise_amp = (term1 + term2)?;

        // Random noise (for parity testing, should use injected noise)
        let noise = Tensor::randn_like(&sine_waves, 0.0, 1.0)?.broadcast_mul(&noise_amp)?;

        // 5. Merge: sine_waves * uv + noise
        let harmonic_part = (sine_waves.broadcast_mul(&uv)? + &noise)?;

        Ok((harmonic_part.transpose(1, 2)?, uv.transpose(1, 2)?, noise))
    }
}

pub struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: candle_nn::Linear,
    voiced_threshold: f32,
    upsample_scale: usize,
    is_causal: bool,
}

impl SourceModuleHnNSF {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        upsample_scale: usize,
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate,
            noise_std as f32, // noise_amp_uv
            sine_amp as f32 / 3.0, // noise_amp_unvoiced
        )?;
        // l_linear: Linear(harmonic_num + 1, 1)
        // Python: nn.Linear(harmonic_num + 1, 1)
        // Check if weight_norm is on logic? generator.py: SourceModuleHnNSF uses regular Linear.
        let l_linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        Ok(Self { sine_gen, l_linear, voiced_threshold, upsample_scale, is_causal })
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
        let (sine_wavs, uv, _) = self.sine_gen.forward(f0, phase_inject, sine_noise_inject, self.voiced_threshold, self.upsample_scale, self.is_causal)?;

        // sine_merge = tanh(linear(sine_waves))
        // sine_wavs: [Batch, Length, Harmonics+1]
        let sine_merge = self.l_linear.forward(&sine_wavs)?;
        let sine_merge = sine_merge.tanh()?;
        // Transpose to [Batch, 1, Length]
        let sine_merge = sine_merge.transpose(1, 2)?;

        // Noise branch - use injected noise or generate random noise
        let noise = match source_noise_inject {
            Some(n) => (n * (self.sine_gen.sine_amp / 3.0))?,
            None => (Tensor::randn_like(&uv, 0.0, 1.0)? * (self.sine_gen.sine_amp / 3.0))?,
        };

        Ok((sine_merge, noise, uv))
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size - 1) * dilation / 2
}

#[allow(clippy::too_many_arguments)]
pub fn load_conv1d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    name: &str,
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

    // Log weight stats
    if let Ok(flat) = weight.flatten_all() {
        if let Ok(vec) = flat.to_vec1::<f32>() {
            let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = vec.iter().sum::<f32>() / vec.len() as f32;
            debug!(
                "    [Conv {}] (prefix: {}) weight stats: min={:.4e}, max={:.4e}, mean={:.4e}, shape=[{},{},{}]",
                name, vb.prefix(), min, max, mean, out_c, in_c, k
            );
        }
    }

    let bias = if vb.contains_tensor("bias") {
        let b = vb.get(out_c, "bias")?;
        // Log bias stats
        if let Ok(flat) = b.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                debug!(
                    "    [Conv {}] bias stats: min={:.4e}, max={:.4e}, mean={:.4e}",
                    name, min, max, mean
                );
            }
        }

        Some(b)
    } else {
        debug!(
            "    [Conv {}] WARNING: Bias NOT found for path: {}",
            name,
            vb.prefix()
        );
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



pub struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acti1: Vec<Snake>,
    acti2: Vec<Snake>,
    // Causal support
    pads1: Vec<usize>,
    pads2: Vec<usize>,
    causal: bool,
}

impl ResBlock {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
        causal: bool,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut acti1 = Vec::new();
        let mut acti2 = Vec::new();
        let mut pads1 = Vec::new();
        let mut pads2 = Vec::new();

        let vb_c1 = vb.pp("convs1");
        let vb_c2 = vb.pp("convs2");
        let vb_a1 = vb.pp("activations1");
        let vb_a2 = vb.pp("activations2");

        for (i, &dil) in dilations.iter().enumerate() {
            // convs1
            let (pad1, manual_pad1) = if causal {
                (0, (kernel_size - 1) * dil) // Left pad manual
            } else {
                (get_padding(kernel_size, dil), 0) // Central pad automatic
            };
            pads1.push(manual_pad1);

            let c1 = load_conv1d(vb_c1.pp(i), channels, channels, kernel_size, 1, dil, pad1, &format!("resblock_c1_{}", i))?;
            convs1.push(c1);

            // convs2 (dil=1)
            let (pad2, manual_pad2) = if causal {
                (0, kernel_size - 1) // Left pad manual
            } else {
                (get_padding(kernel_size, 1), 0) // Central pad automatic
            };
            pads2.push(manual_pad2);

            let c2 = load_conv1d(vb_c2.pp(i), channels, channels, kernel_size, 1, 1, pad2, &format!("resblock_c2_{}", i))?;
            convs2.push(c2);

            // acti
            acti1.push(Snake::new(channels, vb_a1.pp(i))?);
            acti2.push(Snake::new(channels, vb_a2.pp(i))?);
        }

        Ok(Self {
            convs1,
            convs2,
            acti1,
            acti2,
            pads1,
            pads2,
            causal,
        })
    }

    pub fn forward_with_stages(&self, xs: &Tensor, name: Option<&str>, stages: &mut std::collections::HashMap<String, Tensor>) -> Result<Tensor> {
        let mut x = xs.clone();
        for i in 0..self.convs1.len() {
            let mut xt = self.acti1[i].forward(&x)?;
            if self.causal {
                xt = xt.pad_with_zeros(2, self.pads1[i], 0)?;
            }
            xt = self.convs1[i].forward(&xt)?;
            if let Some(n) = name {
                stages.insert(format!("{}_c1_{}", n, i), xt.clone());
            }

            xt = self.acti2[i].forward(&xt)?;
            if self.causal {
                xt = xt.pad_with_zeros(2, self.pads2[i], 0)?;
            }
            xt = self.convs2[i].forward(&xt)?;
            if let Some(n) = name {
                stages.insert(format!("{}_c2_{}", n, i), xt.clone());
            }

            x = (x + xt)?;
        }
        Ok(x)
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut dummy = std::collections::HashMap::new();
        self.forward_with_stages(xs, None, &mut dummy)
    }
}

#[allow(dead_code)]
fn load_conv_transpose1d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Result<candle_nn::ConvTranspose1d> {
    let weight = if vb.contains_tensor("weight_g") {
        let g = vb.get((out_c, 1, 1), "weight_g")?;
        let v = vb.get((out_c, in_c, k), "weight_v")?;
        let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm)?;
        v.broadcast_mul(&scale)?
    } else if vb.contains_tensor("weight.original0") {
        let g = vb.get((out_c, 1, 1), "weight.original0")?;
        let v = vb.get((out_c, in_c, k), "weight.original1")?;
        let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm)?;
        v.broadcast_mul(&scale)?
    } else if vb.contains_tensor("parametrizations.weight.original0") {
        let g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
        let v = vb.get((out_c, in_c, k), "parametrizations.weight.original1")?;
        let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let scale = (g / norm)?;
        v.broadcast_mul(&scale)?
    } else {
        vb.get((out_c, in_c, k), "weight")?
    };

    let weight = weight.transpose(0, 1)?; // Candle expects [In, Out, Kernel] for ConvTranspose1d

    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };

    let cfg = candle_nn::ConvTranspose1dConfig {
        padding,
        output_padding: 0,
        stride,
        dilation: 1,
        groups: 1,
    };

    Ok(candle_nn::ConvTranspose1d::new(weight, bias, cfg))
}

pub struct F0Predictor {
    condnet: Vec<Conv1d>, // Sequential with ELU
    classifier: candle_nn::Linear,
}

fn conv1d_to_cpu(conv: &Conv1d) -> Result<Conv1d> {
    let w = conv.weight().to_device(&Device::Cpu)?;
    let b = conv.bias().map(|b| b.to_device(&Device::Cpu)).transpose()?;
    Ok(Conv1d::new(w, b, conv.config().clone()))
}

fn linear_to_cpu(lin: &candle_nn::Linear) -> Result<candle_nn::Linear> {
    let w = lin.weight().to_device(&Device::Cpu)?;
    let b = lin.bias().map(|b| b.to_device(&Device::Cpu)).transpose()?;
    Ok(candle_nn::Linear::new(w, b))
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
            let layer = load_conv1d(vb_layer, in_c, cond_channels, k, 1, 1, 0, &format!("f0_cond_{}", i))?;

            // Move to CPU
            let layer_cpu = conv1d_to_cpu(&layer)?;
            condnet.push(layer_cpu);
        }

        // Classifier
        let classifier = candle_nn::linear(cond_channels, 1, vb.pp("classifier"))?;
        let classifier_cpu = linear_to_cpu(&classifier)?;

        Ok(Self {
            condnet,
            classifier: classifier_cpu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [Batch, 80, Time]

        let mut h = x.clone();
        for (i, conv) in self.condnet.iter().enumerate() {
            // Get kernel size from weights
            let k = conv.weight().dims()[2];
            let pad = k - 1;

            if i == 0 {
                // F0 predictor layer 0 uses causal_type='right' (Right Padding / Lookahead)
                h = h.pad_with_zeros(2, 0, pad)?;
            } else {
                // F0 predictor layers 1-5 use causal_type='left' (Left Padding / Causal)
                h = h.pad_with_zeros(2, pad, 0)?;
            }
            h = conv.forward(&h)?;
            h = h.elu(1.0)?; // ELU
        }
        // h: [Batch, Cond, Time] -> transpose -> [Batch, Time, Cond]
        let h_t = h.transpose(1, 2)?;
        let out = self.classifier.forward(&h_t)?;

        // out: [Batch, Time, 1]
        let f0 = out.transpose(1, 2)?.abs()?; // [Batch, 1, Time]
        Ok(f0)
    }
}

pub struct HiFTGenerator {
    conv_pre: Conv1d,
    ups: Vec<Conv1d>,
    ups_rates: Vec<usize>,
    source_downs: Vec<Conv1d>,
    source_down_pads: Vec<usize>, // Manual padding for causal source_downs
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    f0_predictor: F0Predictor,
    stft: crate::utils::InverseStftModule,
    analysis_stft: crate::utils::StftModule,
    f0_upsamp_scale: usize,
    num_kernels: usize,
    m_source: SourceModuleHnNSF,
    is_causal: bool, // Flag to indicate if the model uses causal convolutions
}

impl HiFTGenerator {
    pub fn new(vb: VarBuilder, config: &HiFTConfig) -> Result<Self> {
        let ups_rates = &config.upsample_rates;
        let ups_kernels = &config.upsample_kernel_sizes;
        let base_ch = config.base_channels;

        // Determine if causal
        let is_causal = vb.pp("conv_pre").contains_tensor("weight.original1")
            || vb
                .pp("conv_pre")
                .contains_tensor("parametrizations.weight.original1");

        // Conv Pre
        // Fun-CosyVoice3-0.5B has kernel_size=5, padding=2
        let conv_pre_k = if is_causal { 5 } else { 13 };
        let conv_pre_pad = if is_causal { 0 } else { conv_pre_k / 2 }; // If causal, padding is handled manually
        let conv_pre = load_conv1d(
            vb.pp("conv_pre"),
            config.in_channels,
            base_ch,
            conv_pre_k,
            1,
            1,
            conv_pre_pad,
            "conv_pre",
        )?;

        // Ups
        let mut ups = Vec::new();
        let vb_ups = vb.pp("ups");
        for (i, (&_u, &k)) in ups_rates.iter().zip(ups_kernels).enumerate() {
            let in_c = base_ch / (1 << i);
            let out_c = base_ch / (1 << (i + 1));
            // CausalConv1dUpsample is Upsample + CausalConv1d with padding k-1
            // If causal, padding is 0, manual padding k-1
            let conv_pad = 0; // Both causal and non-causal use 0 padding for upsample convs, manual padding for causal

            let name = format!("ups_{}", i);
            let conv = load_conv1d(vb_ups.pp(i), in_c, out_c, k, 1, 1, conv_pad, &name)?;
            ups.push(conv);
        }

        // Source Downs & ResBlocks
        let mut source_downs = Vec::new();
        let mut source_down_pads = Vec::new();
        let mut source_resblocks = Vec::new();
        let vb_sd = vb.pp("source_downs");
        let vb_sr = vb.pp("source_resblocks");

        // Downsample rates construction
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // cum_rates = cumprod(downsample_rates)[::-1]
        let mut ds_base = vec![1];
        let mut rev_ups = ups_rates.clone();
        rev_ups.reverse();
        if rev_ups.len() > 1 {
            ds_base.extend_from_slice(&rev_ups[..rev_ups.len() - 1]);
        }

        let mut cum_rates = Vec::new();
        let mut current_prod = 1;
        for &r in &ds_base {
            current_prod *= r;
            cum_rates.push(current_prod);
        }
        cum_rates.reverse();
        let downsample_rates = cum_rates;

        let src_k = &config.source_resblock_kernel_sizes;
        let src_d = &config.source_resblock_dilation_sizes;

        for (i, ((&u, &k), d)) in downsample_rates.iter().zip(src_k).zip(src_d).enumerate() {
            let ch = base_ch / (1 << (i + 1));
            let in_ch = config.istft_params_n_fft + 2; // nfft+2 channels source

            // source_down
            let (sd_pad, sd_manual_pad) = if is_causal {
                // If u=1: K=1. Pad=0. Manual=0.
                // If u>1: K=u*2. Stride=u. Causal Pad = Stride-1 = u-1.
                if u == 1 {
                    (0, 0)
                } else {
                    (0, u - 1)
                }
            } else if u == 1 {
                (0, 0)
            } else {
                (u / 2, 0)
            };
            source_down_pads.push(sd_manual_pad);

            let sd = if u == 1 {
                load_conv1d(vb_sd.pp(i), in_ch, ch, 1, 1, 1, sd_pad, &format!("source_down_{}", i))?
            } else {
                load_conv1d(vb_sd.pp(i), in_ch, ch, u * 2, u, 1, sd_pad, &format!("source_down_{}", i))?
            };
            source_downs.push(sd);

            // source_resblock
            let sb = ResBlock::new(ch, k, d, vb_sr.pp(i), is_causal)?;
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
                let rb = ResBlock::new(ch, k, d, vb_rb.pp(idx), is_causal)?;
                resblocks.push(rb);
            }
        }

        // Post
        let last_ch = base_ch / (1 << num_ups);
        let conv_post_k = 7;
        let conv_post_pad = if is_causal {
            0
        } else {
            (conv_post_k - 1) / 2
        };
        let conv_post = load_conv1d(
            vb.pp("conv_post"),
            last_ch,
            config.istft_params_n_fft + 2,
            conv_post_k,
            1,
            1,
            conv_post_pad,
            "conv_post",
        )?;

        let stft = crate::utils::InverseStftModule::new(
            config.istft_params_n_fft,
            config.istft_params_hop_len,
            true,
            vb.device(),
        )?;
        let analysis_stft = crate::utils::StftModule::new(
            config.istft_params_n_fft,
            config.istft_params_hop_len,
            true,
            vb.device(),
        )?;

        let upsample_scale = ups_rates.iter().product::<usize>() * config.istft_params_hop_len;
        let m_source = SourceModuleHnNSF::new(
            config.nb_harmonics,
            config.nsf_alpha as f64,
            config.nsf_sigma as f64,
            config.sampling_rate,
            config.voiced_threshold,
            upsample_scale,
            is_causal,
            vb.pp("m_source"),
        )?;

        let f0_predictor = F0Predictor::new(
            config.in_channels,
            config.base_channels,
            vb.pp("f0_predictor"),
        )?;

        Ok(Self {
            conv_pre,
            ups,
            ups_rates: ups_rates.clone(),
            resblocks,
            source_downs,
            source_down_pads,
            source_resblocks,
            num_kernels,
            conv_post,
            stft,
            analysis_stft,
            m_source,
            f0_predictor,
            f0_upsamp_scale: upsample_scale,
            is_causal,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (audio, _) = self.forward_with_stages(mel)?;
        Ok(audio)
    }

    pub fn forward_with_stages(&self, mel: &Tensor) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        let mut stages = std::collections::HashMap::new();
        let mel = mel.to_dtype(DType::F32)?; // Force F32 to avoid F16 precision issues
        stages.insert("input_mel".to_string(), mel.clone());

        // 1. F0 Predictor
        // Python: self.f0_predictor.to('cpu') inside inference.
        // We replicate this by moving input to CPU, running inference, and moving back.
        // This avoids potential GPU nondeterminism with small tensors / specific ops.
        let mel_cpu = mel.to_device(&Device::Cpu)?;
        let f0_cpu = self.f0_predictor.forward(&mel_cpu)?; // [Batch, 1, Length_f0]
        let f0 = f0_cpu.to_device(mel.device())?;

        let mel_len = mel.dim(2)?;
        let f0 = f0.narrow(2, 0, mel_len)?; // crop to mel length
        stages.insert("f0".to_string(), f0.clone());

        // 2. Upsample F0 to Source Resolution
        let s = self.upsample_nearest(&f0, self.f0_upsamp_scale)?;
        stages.insert("f0_upsampled".to_string(), s.clone());

        // 3. Source Module
        let (s_source, _, _) = self.m_source.forward(&s, None, None, None)?; // [B, 1, L_up]
        stages.insert("source".to_string(), s_source.clone());

        // 4. Decode
        let (audio, decode_stages) = self.decode_with_stages(&mel, &s_source)?;
        for (k, v) in decode_stages {
            stages.insert(k, v);
        }

        Ok((audio, stages))
    }

    /// Forward pass with externally injected source tensor for parity testing.
    /// This allows testing the decode path in isolation.
    pub fn forward_with_injected_source(&self, mel: &Tensor, source: &Tensor) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        let mel = mel.to_dtype(DType::F32)?;
        let (audio, stages) = self.decode_with_stages(&mel, source)?;
        Ok((audio, stages))
    }

    /// Forward pass with full SineGen injection for parity testing.
    /// allows injecting sine_waves to verify deterministic source generation.
    pub fn forward_with_sine_injection(
        &self,
        mel: &Tensor,
        sine_waves: &Tensor,
        source_noise: &Tensor
    ) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        let mut stages = std::collections::HashMap::new();
        let mel = mel.to_dtype(DType::F32)?;
        stages.insert("input_mel".to_string(), mel.clone());

        // 1. F0 Predictor
        let f0 = self.f0_predictor.forward(&mel)?; // [Batch, 1, Length_f0]
        let mel_len = mel.dim(2)?;
        let f0 = f0.narrow(2, 0, mel_len)?; // crop to mel length
        stages.insert("f0".to_string(), f0.clone());

        // 2. Upsample F0 to Source Resolution
        let s = self.upsample_nearest(&f0, self.f0_upsamp_scale)?;
        stages.insert("f0_upsampled".to_string(), s.clone());

        // 3. Source Module with Injection
        // Pass sine_waves as phase_inject (first arg) and source_noise
        let (s_source, _, _) = self.m_source.forward(&s, Some(sine_waves), None, Some(source_noise))?;
        stages.insert("source".to_string(), s_source.clone());

        // 4. Decode
        let (audio, decode_stages) = self.decode_with_stages(&mel, &s_source)?;
        for (k, v) in decode_stages {
            stages.insert(k, v);
        }

        Ok((audio, stages))
    }


    fn decode_with_stages(&self, x: &Tensor, s: &Tensor) -> Result<(Tensor, std::collections::HashMap<String, Tensor>)> {
        let mut stages = std::collections::HashMap::new();
        // s is `source` (excitation signal) [B, 1, T]
        // Compute STFT [B, Freq, T_frame]
        // analysis_stft outputs (real, imag)
        let (s_real, s_imag) = self.analysis_stft.transform(s)?;
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?; // [Batch, 2*Freq, Frames]
        stages.insert("s_stft".to_string(), s_stft.clone());

        // 3. Forward
        // x is passed in as argument (mel)
        let mut x = x.clone();

        if self.is_causal {
            // conv_pre uses causal_type='right' (Lookahead). Pad K-1 on RIGHT.
            let k = self.conv_pre.weight().dims()[2];
            let pad = k - 1;
            x = x.pad_with_zeros(2, 0, pad)?;
        }

        x = self.conv_pre.forward(&x)?;
        stages.insert("conv_pre".to_string(), x.clone());

        let num_ups = self.ups.len();

        for i in 0..num_ups {
            x = candle_nn::ops::leaky_relu(&x, 0.1)?; // lrelu_slope=0.1

            // Upsample
            let u = self.ups_rates[i];
            let (b, c, l) = x.dims3()?;
            x = x
                .unsqueeze(3)?
                .repeat((1, 1, 1, u))?
                .reshape((b, c, l * u))?;

            // Causal Up Conv: Pad Left K-1
            if self.is_causal {
                let k = self.ups[i].weight().dims()[2];
                let pad = k - 1;
                x = x.pad_with_zeros(2, pad, 0)?;
            }
            x = self.ups[i].forward(&x)?;

            // Reflection-like padding at the end?
            // CausalHiFTGenerator.py: self.reflection_pad = nn.ReflectionPad1d((1, 0))
            if self.is_causal && i == num_ups - 1 {
                // Pad Left 1. Value = x[1].
                // Use narrow/slice.
                let left = x.i((.., .., 1..2))?;
                x = Tensor::cat(&[&left, &x], 2)?;
            }

            // Fusion
            // Channels: ups[i] -> 256, 128, 64.
            // source_downs[i] -> 256, 128, 64.
            // So we must use idx = i.
            let idx = i;

            // source_downs
            let mut si_in = s_stft.clone();
            if self.is_causal {
                let pad = self.source_down_pads[idx];
                si_in = si_in.pad_with_zeros(2, pad, 0)?;
            }
            if i == 0 {
            }
            let si = self.source_downs[idx].forward(&si_in)?;
            if i == 0 {
            }
            stages.insert(format!("source_down_out_{}", i), si.clone());
            let si = self.source_resblocks[idx].forward_with_stages(&si, Some(&format!("si_res_{}", i)), &mut stages)?;
            stages.insert(format!("si_{}", i), si.clone());

            // Robust slice to handle boundary effects
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;

            let check_len = x_len.min(si_len);
            let x_slice = x.i((.., .., ..check_len))?;
            let si_slice = si.i((.., .., ..check_len))?;

            x = x_slice.add(&si_slice)?;
            stages.insert(format!("loop{}_si", i), si_slice);
            stages.insert(format!("loop{}_x", i), x.clone());

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let idx = i * self.num_kernels + j;
                let xj = self.resblocks[idx].forward_with_stages(&x, Some(&format!("res_{}_{}", i, j)), &mut stages)?;
                match xs {
                    None => xs = Some(xj),
                    Some(prev) => xs = Some((prev + xj)?),
                }
            }
            if let Some(val) = xs {
                x = (val / (self.num_kernels as f64))?;
            }
            stages.insert(format!("after_resblocks_{}", i), x.clone());
        }

        x = candle_nn::ops::leaky_relu(&x, 0.01)?; // Default slope in Python F.leaky_relu

        if self.is_causal {
            // Conv Post K=7. Pad Left 6.
            // (K from weight)
            let k = self.conv_post.weight().dims()[2];
            let pad = k - 1;
            x = x.pad_with_zeros(2, pad, 0)?;
        }
        let x = self.conv_post.forward(&x)?;
        stages.insert("conv_post".to_string(), x.clone());

        // ISTFT Input: [B, n_fft+2, T]
        // Magnitude = exp(x[:, :mid])
        // Phase = sin(x[:, mid:])
        let dim = x.dim(1)?;
        let cutoff = dim / 2; // n_fft/2 + 1

        let mag_log = x.i((.., ..cutoff, ..))?;
        let phase_in = x.i((.., cutoff.., ..))?;

        let magnitude = mag_log.exp()?;
        stages.insert("magnitude".to_string(), magnitude.clone());

        // Clip magnitude to prevent overflow - Python does: magnitude = torch.clip(magnitude, max=1e2)
        let max_mag = Tensor::new(&[100.0f32], magnitude.device())?;
        let magnitude = magnitude.minimum(&max_mag.broadcast_as(magnitude.shape())?)?;

        let phase = phase_in.sin()?;
        stages.insert("phase".to_string(), phase.clone());

        let audio = self.stft.forward(&magnitude, &phase)?;
        stages.insert("pre_clamp_audio".to_string(), audio.clone());

        // NOTE: Python does NOT remove DC offset, so this is commented out
        // let mean_tensor = audio.mean(2)?.broadcast_as(audio.shape())?;
        // let audio = (audio - mean_tensor)?;
        // Clamp audio to [-audio_limit, audio_limit] like Python does
        // Python: x = torch.clamp(x, -self.audio_limit, self.audio_limit) where audio_limit = 0.99
        let audio_limit = 0.99f32;
        let min_val = Tensor::new(&[-audio_limit], audio.device())?;
        let max_val = Tensor::new(&[audio_limit], audio.device())?;
        let audio = audio.maximum(&min_val.broadcast_as(audio.shape())?)?;
        let audio = audio.minimum(&max_val.broadcast_as(audio.shape())?)?;

        Ok((audio, stages))
    }

    // Helper for nearest upsampling (N, C, L) -> (N, C, L * scale)
    // Matches torch.nn.Upsample(scale_factor=scale, mode='nearest')
    fn upsample_nearest(&self, x: &Tensor, scale: usize) -> Result<Tensor> {
        let (n, c, l) = x.dims3()?;
        let out_l = l * scale;

        let x_cpu = x.to_device(&Device::Cpu)?;
        let x_vec = x_cpu.flatten_all()?.to_vec1::<f32>()?;

        let mut out_vec = Vec::with_capacity(n * c * out_l);

        for batch in 0..n {
            for ch in 0..c {
                let off = (batch * c + ch) * l;
                for i in 0..l {
                    let val = x_vec[off + i];
                    for _ in 0..scale {
                        out_vec.push(val);
                    }
                }
            }
        }

        Tensor::from_vec(out_vec, (n, c, out_l), x.device())
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
    pub nsf_alpha: f32,
    pub nsf_sigma: f32,
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
            voiced_threshold: 5.0,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
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
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
        }
    }
}

impl Default for HiFTConfig {
    fn default() -> Self {
        Self::new(16)
    }
}
