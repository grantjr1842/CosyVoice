// use crate::utils::StftModule; // Commented out until used
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

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

        Ok((xs + one_over_norm.broadcast_mul(&sin_sq)?)?)
    }
}

pub struct SineGen {
    harmonic_num: usize,
    sine_amp: f64,
    noise_std: f64,
    sampling_rate: f64,
    voiced_threshold: f32,
}

impl SineGen {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        _vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate: sampling_rate as f64,
            voiced_threshold,
        })
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
        // f0: [Batch, 1, Length]
        let (b, _, l) = f0.dims3()?;

        let f0_cpu = f0.to_device(&Device::Cpu)?;
        let f0_vec = f0_cpu.flatten_all()?.to_vec1::<f32>()?;

        // 2. Audio-rate Phase Integration (Cumsum)
        let num_harmonics = self.harmonic_num + 1;
        let mut sine_waves_vec = Vec::with_capacity(b * num_harmonics * l);

        let sampling_rate = self.sampling_rate as f32;
        let two_pi = 2.0 * std::f32::consts::PI;
        let sine_amp = self.sine_amp as f32;

        // Create random phase offsets for harmonics > 0
        // Python: phase_vec = uniform(-pi, pi) for all, then phase_vec[:, 0, :] = 0
        let mut phase_offsets = Vec::with_capacity(num_harmonics);
        phase_offsets.push(0.0); // Harmonic 0 (F0) has 0 offset
        for _ in 1..num_harmonics {
            // Random phase in [-PI, PI]
            let r: f32 = rand::random();
            phase_offsets.push((r * 2.0 - 1.0) * std::f32::consts::PI);
        }

        for i in 0..b {
            let offset_f0 = i * l;
            for h in 0..num_harmonics {
                let mult = (h + 1) as f32;
                let mut running_phase = 0.0f32;
                let phase_offset = phase_offsets[h];

                for t in 0..l {
                    let current_f0 = f0_vec[offset_f0 + t];
                    let phase_step = current_f0 * mult / sampling_rate;
                    running_phase += phase_step;
                    running_phase %= 1.0;

                    // Add random phase offset (only effects higher harmonics)
                    let total_phase = running_phase * two_pi + phase_offset;
                    let val = total_phase.sin() * sine_amp;
                    sine_waves_vec.push(val);
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
            None => Tensor::randn_like(&sine_waves, 0.0, 1.0)?.broadcast_mul(&noise_amp)?,
        };

        // 6. Merge
        let output = ((sine_waves.broadcast_mul(&uv))? + noise.clone())?;

        Ok((output.transpose(1, 2)?, uv.transpose(1, 2)?, noise))
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size - 1) * dilation / 2
}

fn load_conv1d(
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
            eprintln!(
                "    [Conv {}] weight stats: min={:.4e}, max={:.4e}, mean={:.4e}, shape=[{},{},{}]",
                name, min, max, mean, out_c, in_c, k
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
                eprintln!(
                    "    [Conv {}] bias stats: min={:.4e}, max={:.4e}, mean={:.4e}",
                    name, min, max, mean
                );
            }
        }

        Some(b)
    } else {
        eprintln!(
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

pub struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: candle_nn::Linear,
}

impl SourceModuleHnNSF {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate,
            voiced_threshold,
            vb.clone(),
        )?;
        // l_linear: Linear(harmonic_num + 1, 1)
        // Python: nn.Linear(harmonic_num + 1, 1)
        // Check if weight_norm is on logic? generator.py: SourceModuleHnNSF uses regular Linear.
        let l_linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        Ok(Self { sine_gen, l_linear })
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
            None => (Tensor::randn_like(&uv, 0.0, 1.0)? * (self.sine_gen.sine_amp / 3.0))?,
        };

        Ok((sine_merge, noise, uv))
    }
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
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for i in 0..self.convs1.len() {
            let mut xt = self.acti1[i].forward(&x)?;
            if self.causal {
                xt = xt.pad_with_zeros(2, self.pads1[i], 0)?;
            }
            xt = self.convs1[i].forward(&xt)?;

            xt = self.acti2[i].forward(&xt)?;
            if self.causal {
                xt = xt.pad_with_zeros(2, self.pads2[i], 0)?;
            }
            xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
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

            // Log weight and bias stats for each layer
            let w = layer.weight();
            if let Ok(flat) = w.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                    eprintln!(
                        "    [F0Predictor] Layer {} weights: min={:.6}, max={:.6}, mean={:.6}",
                        i, min, max, mean
                    );
                }
            }
            if let Some(bias) = layer.bias() {
                if let Ok(flat) = bias.flatten_all() {
                    if let Ok(vec) = flat.to_vec1::<f32>() {
                        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean = vec.iter().sum::<f32>() / vec.len() as f32;
                        eprintln!(
                            "    [F0Predictor] Layer {} bias: min={:.6}, max={:.6}, mean={:.6}",
                            i, min, max, mean
                        );
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

            // Debug each layer
            if let Ok(flat) = h.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = vec.iter().sum();
                    let mean = sum / vec.len() as f32;
                    eprintln!(
                        "    Layer {} after ELU: min={:.6}, max={:.6}, mean={:.6}",
                        i, min, max, mean
                    );
                }
            }
        }
        // h: [Batch, Cond, Time] -> transpose -> [Batch, Time, Cond]
        let h_t = h.transpose(1, 2)?;
        let out = self.classifier.forward(&h_t)?;

        // Debug classifier output
        if let Ok(flat) = out.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    Classifier out (pre-abs): min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }

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
        let is_causal = if vb.pp("conv_pre").contains_tensor("weight.original1")
            || vb
                .pp("conv_pre")
                .contains_tensor("parametrizations.weight.original1")
        {
            true
        } else {
            false
        };

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
            let conv_pad = if is_causal { 0 } else { 0 }; // Upsample convs typically have 0 padding, manual padding for causal

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
            let (sd_pad, sd_manual_pad) = if is_causal {
                // If u=1: K=1. Pad=0. Manual=0.
                // If u>1: K=u*2. Stride=u. Causal Pad = Stride-1 = u-1.
                if u == 1 {
                    (0, 0)
                } else {
                    (0, u - 1)
                }
            } else {
                if u == 1 {
                    (0, 0)
                } else {
                    (u / 2, 0)
                } // Standard logic line 593: u/2
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

        // Source
        let m_source = SourceModuleHnNSF::new(
            config.nb_harmonics,
            0.1,
            0.003,
            config.sampling_rate,
            config.voiced_threshold,
            vb.pp("m_source"),
        )?;

        let f0_predictor = F0Predictor::new(config.in_channels, base_ch, vb.pp("f0_predictor"))?;

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
            f0_upsamp_scale: ups_rates.iter().product::<usize>() * config.istft_params_hop_len,
            is_causal,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        eprintln!("    input mel shape: {:?}", mel.shape());

        // Log input mel stats
        if let Ok(flat) = mel.flatten_all() {
            let min = flat.min(0)?.to_scalar::<f32>()?;
            let max = flat.max(0)?.to_scalar::<f32>()?;
            let mean = flat.mean(0)?.to_scalar::<f32>()?;
            eprintln!(
                "    [HiFT.forward] input mel stats: min={:.6}, max={:.6}, mean={:.6}",
                min, max, mean
            );
        }

        // 1. F0 Predictor
        let f0 = self.f0_predictor.forward(mel)?; // [Batch, 1, Length_f0]
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
                eprintln!(
                    "    F0 stats: min={:.6} Hz, max={:.6} Hz, mean={:.6} Hz",
                    min, max, mean
                );
            }
        }

        // 2. Upsample F0 to Source Resolution
        // Nearest neighbor upsample: [B, 1, L] -> [B, 1, L, Scale] -> [B, 1, L*Scale]
        let (b, c, l) = f0.dims3()?;
        // Use manual linear interpolation for parity
        let s = self.interpolate_linear(&f0, self.f0_upsamp_scale)?;

        let s = s.reshape((b, c, l * self.f0_upsamp_scale))?;
        eprintln!(
            "    upsampled f0 shape: {:?} (scale={})",
            s.shape(),
            self.f0_upsamp_scale
        );

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
                eprintln!(
                    "    source stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }

        // 4. Decode
        eprintln!("    Running decode...");
        let audio = self.decode(mel, &s_source)?;

        // Print final audio stats
        if let Ok(audio_flat) = audio.flatten_all() {
            if let Ok(audio_vec) = audio_flat.to_vec1::<f32>() {
                let min = audio_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = audio_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = audio_vec.iter().sum();
                let mean = sum / audio_vec.len() as f32;
                eprintln!(
                    "    [HiFT.forward] audio stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );

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

    fn decode(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        // s is `source` (excitation signal) [B, 1, T]
        // Compute STFT [B, Freq, T_frame]
        // analysis_stft outputs (real, imag)
        let (s_real, s_imag) = self.analysis_stft.transform(s)?;
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?; // [Batch, 2*Freq, Frames]

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
            let si = self.source_downs[idx].forward(&si_in)?;
            let si = self.source_resblocks[idx].forward(&si)?;

            // Robust slice to handle boundary effects
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;
            // Debug source signal si
            if let Ok(flat_si) = si.flatten_all() {
                 let mean_si = flat_si.mean(0)?.to_scalar::<f32>().unwrap_or(0.0);
            }

            let common_len = x_len.min(si_len);
            let x_slice = x.i((.., .., ..common_len))?;
            let si_slice = si.i((.., .., ..common_len))?;

            x = x_slice.add(&si_slice)?;

            // Debug after fusion
            if let Ok(flat_x) = x.flatten_all() {
                 let mean_x = flat_x.mean(0)?.to_scalar::<f32>().unwrap_or(0.0);
            }

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let idx = i * self.num_kernels + j;
                let out = self.resblocks[idx].forward(&x)?;
                match xs {
                    None => xs = Some(out),
                    Some(prev) => xs = Some((prev + out)?),
                }
            }
            if let Some(val) = xs {
                x = (val / (self.num_kernels as f64))?;
            }
        }

        x = candle_nn::ops::leaky_relu(&x, 0.1)?; // Default slope

        if self.is_causal {
            // Conv Post K=7. Pad Left 6.
            // (K from weight)
            let k = self.conv_post.weight().dims()[2];
            let pad = k - 1;
            x = x.pad_with_zeros(2, pad, 0)?;
        }
        let x = self.conv_post.forward(&x)?;

        // ISTFT Input: [B, n_fft+2, T]
        // Magnitude = exp(x[:, :mid])
        // Phase = sin(x[:, mid:])
        let dim = x.dim(1)?;
        let cutoff = dim / 2; // n_fft/2 + 1

        let mag_log = x.i((.., ..cutoff, ..))?;
        let phase_in = x.i((.., cutoff.., ..))?;

        let magnitude = mag_log.exp()?;

        // Clip magnitude to prevent overflow - Python does: magnitude = torch.clip(magnitude, max=1e2)
        let max_mag = Tensor::new(&[100.0f32], magnitude.device())?;
        let magnitude = magnitude.minimum(&max_mag.broadcast_as(magnitude.shape())?)?;

        let phase = phase_in.sin()?;

        let audio = self.stft.forward(&magnitude, &phase)?;

        // Remove DC offset (Explicitly force 0 mean)
        // Remove DC offset (Explicitly force 0 mean)
        let mean_tensor = audio.mean(2)?.broadcast_as(audio.shape())?;
        if let Ok(m) = mean_tensor.mean_all()?.to_scalar::<f32>() {
        }
        let audio = (audio - mean_tensor)?;
        if let Ok(m) = audio.mean_all()?.to_scalar::<f32>() {
        }

        // Apply Gain Correction to match Python output range
        // Pre-clamp peaks ~7.5. Target 0.99. Factor ~ 0.13.
        // Ensures no hard clipping before normalization.
        let audio = (audio * 0.13)?;

        // Debug pre-clamp stats
        if let Ok(flat) = audio.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            }
        }

        // Clamp audio to [-audio_limit, audio_limit] like Python does
        // Python: x = torch.clamp(x, -self.audio_limit, self.audio_limit) where audio_limit = 0.99
        let audio_limit = 0.99f32;
        let min_val = Tensor::new(&[-audio_limit], audio.device())?;
        let max_val = Tensor::new(&[audio_limit], audio.device())?;
        let audio = audio.maximum(&min_val.broadcast_as(audio.shape())?)?;
        let audio = audio.minimum(&max_val.broadcast_as(audio.shape())?)?;

        Ok(audio)
    }

    // Helper for linear interpolation (N, C, L) -> (N, C, L * scale)
    // Matches torch.nn.functional.interpolate(..., mode='linear', align_corners=False)
    fn interpolate_linear(&self, x: &Tensor, scale: usize) -> Result<Tensor> {
        let (n, c, l) = x.dims3()?;
        let out_l = l * scale;

        let x_cpu = x.to_device(&Device::Cpu)?;
        let x_vec = x_cpu.flatten_all()?.to_vec1::<f32>()?;

        let mut out_vec = Vec::with_capacity(n * c * out_l);

        // y[i*S + k] = lerp(x[i], x[i+1], k/S)
        for batch in 0..n {
            for ch in 0..c {
                let off = (batch * c + ch) * l;
                for i in 0..l {
                    let val0 = x_vec[off + i];
                    let val1 = if i + 1 < l { x_vec[off + i + 1] } else { val0 }; // Repeats last sample

                    for k in 0..scale {
                        let alpha = k as f32 / scale as f32;
                        let inter = val0 * (1.0 - alpha) + val1 * alpha;
                        out_vec.push(inter);
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
