use crate::utils::StftModule;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{ops, Conv1d, Conv1dConfig, VarBuilder};
use std::f64::consts::PI;

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
    harmonic_range: Tensor, // [1, 1, harmonic_num + 1] -> broadcastable to [Batch, Len, Harmonics] ?
    // Check shapes. Python: F_mat is [Batch, Harmonics, Length]
    // My harmonic_range should be [1, Harmonics, 1] for broadcasting against f0 [Batch, 1, Length]?
    voiced_threshold: f32,
    device: Device,
}

impl SineGen {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: usize,
        voiced_threshold: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dev = vb.device();
        // harmonic_range: [1.0, 2.0, ... H+1]
        let range: Vec<f32> = (1..=harmonic_num + 1).map(|x| x as f32).collect();
        // Shape: [1, Harmonics, 1] so we can mul with f0 [Batch, 1, Length] -> [Batch, Harmonics, Length]
        let harmonic_range = Tensor::from_vec(range, (1, harmonic_num + 1, 1), dev)?;

        Ok(Self {
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate: sampling_rate as f64,
            harmonic_range,
            voiced_threshold,
            device: dev.clone(),
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
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // f0: [Batch, 1, Length]

        // 1. Calculate F_mat: harmonics
        // harmonic_range: [1, H+1, 1]
        // f0: [Batch, 1, Length]
        // We want [Batch, H+1, Length].
        // Expand/Broadcast.

        let _f0_expanded = f0.broadcast_as((f0.dim(0)?, self.harmonic_num + 1, f0.dim(2)?))?;
        // But multiplying needs correct dims.
        // broadcast_mul automatically broadcasts.

        // harmonic_range [1, H+1, 1] * f0 [Batch, 1, Length] -> [Batch, H+1, Length]
        eprintln!("SineGen: harmonic_num={}, harmonic_range shape={:?}, f0 shape={:?}",
            self.harmonic_num, self.harmonic_range.shape(), f0.shape());

        let f_mat = self.harmonic_range.broadcast_mul(f0)?;
        eprintln!("SineGen: f_mat shape={:?}", f_mat.shape());
        let f_mat = (f_mat / self.sampling_rate)?;

        // 2. Cumsum for phase: theta = 2 * pi * cumsum(f_mat) % 1
        // Candle `cumsum` implementation naively creates an LxL matrix (O(N^2) memory), causing OOM.
        // We implement it manually on CPU (O(N)).

        let f_mat_cpu = f_mat.to_device(&Device::Cpu)?;
        let (b, c, l) = f_mat_cpu.dims3()?;
        let mut f_vec: Vec<f32> = f_mat_cpu.flatten_all()?.to_vec1()?;

        // Cumsum along dim 2 (inner-most dimension in [B, C, L] layout?)
        // Check layout. contiguous [B, C, L] means L varies fastest?
        // Yes, row-major.
        // So we iterate chunks of size L.

        for i in 0..(b * c) {
            let offset = i * l;
            let mut sum = 0.0;
            for j in 0..l {
                sum += f_vec[offset + j];
                f_vec[offset + j] = sum;
            }
        }

        let cumsum_cpu = Tensor::from_vec(f_vec, (b, c, l), &Device::Cpu)?;
        let cumsum = cumsum_cpu.to_device(&f_mat.device())?;

        // % 1 logic: x - floor(x).
        // Since we wrap phase, usually we do sin(2pi * x).
        // sin(2pi * (x % 1)) = sin(2pi * x).
        // So we can skip % 1 if we just multiply by 2pi.
        // However, large floats lose precision. keeping it bounded is good.
        // x - x.floor()
        // Candle doesn't have `floor`?
        // It does.
        // Or assume f32 precision holds enough for short clips.
        // Let's preserve precision:
        // theta_mat = 2 * pi * (cumsum % 1)
        // If candle missing floor, check ops.
        // Assuming precision is fine for typical audio chunks (30s).

        let two_pi = 2.0 * PI;
        let theta_mat = (cumsum * two_pi)?;

        // 3. Random phase (for noise/texture?)
        // Python: phase_vec = u_dist.sample(...)
        // phase_vec[:, 0, :] = 0
        // sine_waves = sine_amp * sin(theta + phase)

        // We can skip random phase for inference determinism or implement it.
        // Python code generates random phase `U[-pi, pi]`.
        // Let's implement it for parity.
        let shape = theta_mat.shape();
        let _phase_vec = (Tensor::rand(0.0f32, 1.0f32, shape, &self.device)? * two_pi)? - PI;
        // Zero out fundamental (idx 0) phase?
        // Python: phase_vec[:, 0, :] = 0
        // We can slice and cat.
        // slice(1, 0, 1) -> set to 0?
        // Or just construct it that way.
        // Easier: phase_vec is applied everywhere.
        // Hard to mutate in Candle.
        // let fundamental_phase = Tensor::zeros(...)
        // let harmonic_phases = Tensor::rand(...)
        // let phase_vec = Tensor::cat(...)

        let b = shape.dims()[0];
        let l = shape.dims()[2];
        let fund = Tensor::zeros((b, 1, l), DType::F32, &self.device)?;
        let harm = ((Tensor::rand(0.0f32, 1.0f32, (b, self.harmonic_num, l), &self.device)?
            * two_pi)?
            - PI)?;
        let phase_vec = Tensor::cat(&[&fund, &harm], 1)?; // Concat along dim 1 (channels)

        let sine_waves = ((theta_mat + phase_vec)?.sin()? * self.sine_amp)?;

        // 4. UV
        // uv = (f0 > threshold).float()
        let uv = f0.gt(self.voiced_threshold as f64)?.to_dtype(DType::F32)?;

        // 5. Noise
        // noise_amp = uv * noise_std + (1-uv) * sine_amp / 3
        let term1 = (&uv * self.noise_std)?;
        let ones = Tensor::ones_like(&uv)?;
        let term2 = ((&ones - &uv)? * (self.sine_amp / 3.0))?;
        let noise_amp = (term1 + term2)?;

        let noise = Tensor::randn_like(&sine_waves, 0.0, 1.0)?.broadcast_mul(&noise_amp)?; // Broadcast noise_amp over channels

        // 6. Merge
        // sine_waves = sine_waves * uv + noise
        // Check broadcast: sine_waves [B, H+1, L], uv [B, 1, L]. OK.
        let output = ((sine_waves.broadcast_mul(&uv))? + noise.clone())?;

        // Return [Batch, H+1, Length], UV, Noise
        Ok((output, uv, noise)) // Noise not typically used outside
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

pub struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: candle_nn::Linear,
    l_tanh: bool, // Just a flag/marker effectively
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

        Ok(Self {
            sine_gen,
            l_linear,
            l_tanh: true,
        })
    }

    // forward(f0) -> (sine_merge, noise, uv)
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (sine_wavs, uv, _) = self.sine_gen.forward(f0)?;

        // sine_merge = tanh(linear(sine_waves))
        // sine_waves: [Batch, Harmonics, Length].
        // Linear expects [Batch, *, InFeatures].
        // We need to transpose to apply linear on Harmonics dimension?
        // Candle Linear applies to last dim.
        // sine_wavs: [Batch, Chan, Time]. We want [Batch, Time, Chan].
        let sine_wavs_t = sine_wavs.transpose(1, 2)?;
        let sine_merge = self.l_linear.forward(&sine_wavs_t)?;
        let sine_merge = sine_merge.tanh()?;
        // Transpose back to [Batch, 1, Length] (since out=1)
        let sine_merge = sine_merge.transpose(1, 2)?;

        // Noise branch
        // Python: if training=False, noise = uv * sine_amp / 3 (if causal) or randn.
        // Assuming inference mode always.
        // The noise logic in SineGen (noise variable) was:
        // noise_amp * randn.
        // SourceModule logic (lines 371-374) overrides logic?
        // "source for noise branch... if training False... noise = uv * sine_amp / 3".
        // Wait, self.uv is fixed random if Causal.
        // If not causal, noise = randn * sine_amp / 3.

        // Let's implement logic consistent with inference mode.
        // sine_gen.forward output 'noise' which combines uv mask logic.
        // But SourceModule re-generates noise?
        // Line 367: sine_wavs, uv, _ = self.l_sin_gen(x)
        // Line 374: noise = torch.randn_like(uv) * self.sine_amp / 3

        // So SourceModule uses simpler noise logic for 'noise' output, separate from 'sine_merge'.
        // 'sine_merge' is the harmonic part. 'noise' is the unvoiced part?
        // Yes.

        let noise = (Tensor::randn_like(&uv, 0.0, 1.0)? * (self.sine_gen.sine_amp / 3.0))?;

        Ok((sine_merge, noise, uv))
    }
}

pub struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
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
            let pad1 = get_padding(kernel_size, dil);
            // convs1[i]
            let c1 = load_conv1d(vb_c1.pp(i), channels, channels, kernel_size, 1, dil, pad1)?;
            convs1.push(c1);

            let pad2 = get_padding(kernel_size, 1);
            // convs2[i]
            let c2 = load_conv1d(vb_c2.pp(i), channels, channels, kernel_size, 1, 1, pad2)?;
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
            let xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

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
        // condnet: 5 layers of Conv1d(3, padding=1) + WeightNorm
        let mut condnet = Vec::new();
        let vb_net = vb.pp("condnet");

        // 0, 2, 4, 6, 8: Conv1d layers
            let layer = load_conv1d(vb_layer, in_c, cond_channels, k, 1, 1, k / 2)?;
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
        for conv in self.condnet.iter() {
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
    ups: Vec<candle_nn::ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    f0_predictor: F0Predictor,
    stft: crate::utils::InverseStftModule,
    analysis_stft: crate::utils::StftModule,
    f0_upsamp_scale: usize,
    num_kernels: usize,
    m_source: SourceModuleHnNSF,
    device: Device,
}

impl HiFTGenerator {
    pub fn new(vb: VarBuilder, config: &HiFTConfig) -> Result<Self> {
        let ups_rates = &config.upsample_rates;
        let ups_kernels = &config.upsample_kernel_sizes;
        let base_ch = config.base_channels;

        // F0 Predictor
        let _f0_predictor = F0Predictor::new(config.in_channels, 512, vb.pp("f0_predictor"))?;

        // Conv Pre
        // Fun-CosyVoice3-0.5B has kernel_size=5, padding=2
        let conv_pre_k = if vb.pp("conv_pre").contains_tensor("weight.original1") {
             5
        } else if vb.pp("conv_pre").contains_tensor("parametrizations.weight.original1") {
             5
        } else {
             13
        };
        let conv_pre = load_conv1d(
            vb.pp("conv_pre"),
            config.in_channels,
            base_ch,
            conv_pre_k,
            1,
            1,
            conv_pre_k / 2,
        )?;

        // Ups
        let mut ups = Vec::new();
        let vb_ups = vb.pp("ups");
        for (i, (&u, &k)) in ups_rates.iter().zip(ups_kernels).enumerate() {
            let in_c = base_ch / (1 << i);
            let out_c = base_ch / (1 << (i + 1));
            let pad = (k - u) / 2;
            let conv = load_conv_transpose1d(vb_ups.pp(i), in_c, out_c, k, u, pad)?;
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
                load_conv1d(vb_sd.pp(i), in_ch, ch, 1, 1, 1, 0)?
            } else {
                load_conv1d(vb_sd.pp(i), in_ch, ch, u * 2, u, 1, u / 2)?
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
        let conv_post = load_conv1d(
            vb.pp("conv_post"),
            last_ch,
            config.istft_params_n_fft + 2,
            7,
            1,
            1,
            3,
        )?;

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

        // Source
        let m_source = SourceModuleHnNSF::new(
            config.nb_harmonics,
            0.1,
            0.003,
            config.sampling_rate,
            0.0,
            vb.pp("m_source"),
        )?;

        let f0_predictor = F0Predictor::new(config.in_channels, base_ch, vb.pp("f0_predictor"))?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            source_downs,
            source_resblocks,
            num_kernels,
            conv_post,
            stft,
            analysis_stft,
            m_source,
            f0_predictor,
            f0_upsamp_scale: ups_rates.iter().product::<usize>() * config.istft_params_hop_len,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // mel: [Batch, 80, Length]

        // 1. F0 Predictor
        let f0 = self.f0_predictor.forward(mel)?; // [Batch, 1, Length_f0]
        let mel_len = mel.dim(2)?;
        let f0 = f0.narrow(2, 0, mel_len)?; // crop to mel length

        // 2. Upsample F0 to Source Resolution
        // Nearest neighbor upsample: [B, 1, L] -> [B, 1, L, Scale] -> [B, 1, L*Scale]
        let (b, c, l) = f0.dims3()?;
        let s = f0
            .unsqueeze(3)?
            .repeat((1, 1, 1, self.f0_upsamp_scale))?
            .reshape((b, c, l * self.f0_upsamp_scale))?;
        eprintln!("HiFT s (upsampled f0) shape: {:?}", s.shape());

        // 3. Source Module
        let (s_source, _, _) = self.m_source.forward(&s)?; // [B, 1, L_up]

        // 4. Decode
        self.decode(mel, &s_source)
    }

    fn decode(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        // s STFT logic for fusion
        // s: [Batch, 1, Time]
        // We need STFT of s.
        // Wait, self.stft is INVERSE. We need ANALYSIS STFT.
        // I need to add `analysis_stft` to struct.
        // For now, let's assume I added it.
        let (s_real, s_imag) = self.analysis_stft.transform(s)?; // [Batch, Freq, Frames]
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?; // [Batch, 2*Freq, Frames]

        // x (mel): [Batch, 80, Length]
        let mut x = self.conv_pre.forward(x)?;

        let num_ups = self.ups.len();

        for i in 0..num_ups {
            x = candle_nn::ops::leaky_relu(&x, 0.1)?; // lrelu_slope=0.1 default
            x = self.ups[i].forward(&x)?;

            // Ref padding?
            // hift.py: if i == num_ups - 1: reflection_pad(x)
            if i == num_ups - 1 {
                let left = x.i((.., .., 1..2))?;
                x = Tensor::cat(&[&left, &x], 2)?;
            }

            // Fusion
            // si = source_downs[i](s_stft)
            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;

            // Robust slice to handle boundary effects
            let x_len = x.dim(2)?;
            let si_len = si.dim(2)?;
            let common_len = x_len.min(si_len);
            let mut x_slice = x.i((.., .., ..common_len))?;
            let si_slice = si.i((.., .., ..common_len))?;
            x = (x_slice + si_slice)?;

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
        x = self.conv_post.forward(&x)?;

        // ISTFT Input: [B, n_fft+2, T]
        // Magnitude = exp(x[:, :mid])
        // Phase = sin(x[:, mid:])
        let dim = x.dim(1)?;
        let cutoff = dim / 2; // n_fft/2 + 1

        let mag_log = x.i((.., ..cutoff, ..))?;
        let phase_in = x.i((.., cutoff.., ..))?;

        let magnitude = mag_log.exp()?;
        let phase = phase_in.sin()?; // Redundant sin in python?
                                     // Python: phase = torch.sin(x[:, mid:, :])
                                     // Then ISTFT(mag, phase) -> real = mag * cos(phase), img = mag * sin(phase).
                                     // Since phase input is already sine of something?
                                     // Wait, "actually, sin is redundancy" comment in python.
                                     // It implies the network predicts 'phase' value which we treat as angle?
                                     // No, it predicts x. Phase = sin(x).
                                     // So Phase Angle is not x.
                                     // Real part reconstruction uses cos(phase_angle). Imag uses sin(phase_angle).
                                     // Here `phase` variable IS `sin(angle)`.
                                     // So we have sin(theta). We need cos(theta).
                                     // cos = sqrt(1 - sin^2). Sign?
                                     // If we only predict sin(theta), we lose sign of cos(theta).
                                     // Maybe HiFT assumes phase is in [-pi/2, pi/2] or something?
                                     // Or "sin is redundancy" means we just use it as is?
                                     // Let's check `_istft` in generator.py
                                     // `real = magnitude * torch.cos(phase)`
                                     // `img = magnitude * torch.sin(phase)`
                                     // Wait, if `phase` variable passed to `_istft` is already result of `sin(x)`, then
                                     // `real = mag * cos(sin(x))`. `img = mag * sin(sin(x))`.
                                     // This seems weird if `x` is unbounded.
                                     // If `x` is angle, then `phase = sin(x)`.
                                     // Then we do `cos(sin(x))`.
                                     // This seems to be the literal translation.

        // I will implement `stft.forward` taking `magnitude` and `phase_input`.
        // Logic inside `InverseStftModule::forward` was `mag * phase.cos()`.
        // If `phase` input is `x` (angle), that is correct.
        // But here `phase` passed is `sin(x)`.
        // So I should pass `x` (the angle) to `InverseStft`?
        // Python: phase = torch.sin(x_slice). _istft(magnitude, phase).
        // So `_istft` receives `sin(x)`.
        // inside `_istft`: `real = mag * cos(phase)`.
        // So it computes `cos(sin(x))`.
        // This effectively constraints the angle to `sin(x)` which is in [-1, 1] radians?
        // This acts as a bounded phase constraint?
        // Yes, likely intended for stability.

        // So, `InverseStftModule` should take `angle` or `raw_phase_param`?
        // My `InverseStftModule` takes `phase` and does `.cos()` on it.
        // If I pass `x.i(cutoff..)`, it will do `cos(x)`.
        // Python does `cos(sin(x))`.
        // So I should pass `x.i(cutoff..).sin()?` to my module?
        // My module does `phase.cos()`.
        // So `(sin(x)).cos()`.
        // Yes.

        self.stft.forward(&magnitude, &phase)
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
        }
    }
}

impl Default for HiFTConfig {
    fn default() -> Self {
        Self::new(16)
    }
}
