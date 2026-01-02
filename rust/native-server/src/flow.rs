use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, LayerNorm, Linear, VarBuilder};

// Force rebuild marker: REBUILD_V1
const _FORCE_REBUILD: u32 = 1;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub mel_dim: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            depth: 2,
            heads: 16,
            dim_head: 64,
            mel_dim: 80,
        }
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
    silu: candle_nn::Activation,
}

impl TimestepEmbedding {
    pub fn new(vb: VarBuilder, dim: usize, inner_dim: usize) -> Result<Self> {
    // PyTorch uses time_mlp.0 and time_mlp.2 (Sequential indices) instead of linear_1/linear_2
    // Try both naming conventions for compatibility
    let time_mlp = vb.pp("time_mlp");
    let linear_1 = if time_mlp.contains_tensor("0.weight") {
        linear(dim, inner_dim, time_mlp.pp("0"))?
    } else {
        linear(dim, inner_dim, vb.pp("linear_1"))?
    };
    let linear_2 = if time_mlp.contains_tensor("2.weight") {
        linear(inner_dim, inner_dim, time_mlp.pp("2"))?
    } else {
        linear(inner_dim, inner_dim, vb.pp("linear_2"))?
    };
    Ok(Self {
        linear_1,
        linear_2,
        silu: candle_nn::Activation::Silu,
    })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_emb = sinusoidal_embedding(t, 256)?; // 256 is default dim for SinusPositionEmbedding
        t_emb
            .apply(&self.linear_1)?
            .apply(&self.silu)?
            .apply(&self.linear_2)
    }
}

pub struct AdaLayerNormZero {
    pub norm: LayerNorm,
    pub linear: Linear,
    pub silu: candle_nn::Activation,
}

impl AdaLayerNormZero {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let device = vb.device();
        let weight = Tensor::ones((dim,), DType::F32, device)?;
        let bias = Tensor::zeros((dim,), DType::F32, device)?;
        let norm = LayerNorm::new(weight, bias, 1e-6);

        let linear = linear(dim, dim * 6, vb.pp("linear"))?;
        Ok(Self {
            norm,
            linear,
            silu: candle_nn::Activation::Silu,
        })
    }

    pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&emb.apply(&self.silu)?)?;
        let chunks = emb.chunk(6, 1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        let gate = gate.unsqueeze(1)?;
        let shift = shift.unsqueeze(1)?;
        let scale = scale.unsqueeze(1)?;

        let x = self.norm.forward(x)?;
        let x = x.broadcast_mul(&(scale.affine(1.0, 1.0)?))?;
        let x = x.broadcast_add(&shift)?;
        x.broadcast_mul(&gate)
    }
}

pub struct AdaLayerNormZeroFinal {
    pub norm: LayerNorm,
    pub linear: Linear,
    pub silu: candle_nn::Activation,
}

impl AdaLayerNormZeroFinal {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        // Python: elementwise_affine=False, so no weights/bias.
        // candle LayerNorm requires Tensors. We create them manually (weight=1, bias=0).
        let device = vb.device();
        let weight = Tensor::ones((dim,), DType::F32, device)?;
        let bias = Tensor::zeros((dim,), DType::F32, device)?;
        let norm = LayerNorm::new(weight, bias, 1e-6);

        let linear = linear(dim, dim * 2, vb.pp("linear"))?;
        Ok(Self {
            norm,
            linear,
            silu: candle_nn::Activation::Silu,
        })
    }

    pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&emb.apply(&self.silu)?)?;
        let chunks = emb.chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);

        let scale = scale.unsqueeze(1)?;
        let shift = shift.unsqueeze(1)?;

        let x = self.norm.forward(x)?;

        x.broadcast_mul(&(scale.affine(1.0, 1.0)?))?
            .broadcast_add(&shift)
    }
}

pub struct DiTBlock {
    attn_norm: AdaLayerNormZero,
    attn: Attention,
    ff_norm: LayerNorm,
    ff: FeedForward,
}

impl DiTBlock {
    pub fn new(vb: VarBuilder, dim: usize, heads: usize, dim_head: usize) -> Result<Self> {
        let device = vb.device();
        let attn_norm = AdaLayerNormZero::new(vb.pp("attn_norm"), dim)?;
        let attn = Attention::new(vb.pp("attn"), dim, heads, dim_head)?;

        let ff_norm_weight = Tensor::ones((dim,), DType::F32, device)?;
        let ff_norm_bias = Tensor::zeros((dim,), DType::F32, device)?;
        let ff_norm = LayerNorm::new(ff_norm_weight, ff_norm_bias, 1e-6);

        let ff = FeedForward::new(vb.pp("ff"), dim, 2)?;
        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        t_emb: &Tensor,
        mask: &Tensor,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let emb = self
            .attn_norm
            .linear
            .forward(&t_emb.apply(&self.attn_norm.silu)?)?;
        let chunks = emb.chunk(6, 1)?;

        let shift_msa = chunks[0].unsqueeze(1)?;
        let scale_msa = chunks[1].unsqueeze(1)?;
        let gate_msa = chunks[2].unsqueeze(1)?;
        let shift_mlp = chunks[3].unsqueeze(1)?;
        let scale_mlp = chunks[4].unsqueeze(1)?;
        let gate_mlp = chunks[5].unsqueeze(1)?;

        // MSA
        let res_msa = x.clone();
        let x_norm = self.attn_norm.norm.forward(x)?;
        eprintln!(
            "DIT MSA: x_norm={:?}, scale_msa={:?}, shift_msa={:?}",
            x_norm.shape(),
            scale_msa.shape(),
            shift_msa.shape()
        );
        let x_norm = x_norm
            .broadcast_mul(&(scale_msa.affine(1.0, 1.0)?))?
            .broadcast_add(&shift_msa)?;
        let x_attn = self.attn.forward(&x_norm, mask, rope)?;

        eprintln!(
            "DIT MSA ADD: res_msa={:?}, x_attn={:?}, gate_msa={:?}",
            res_msa.shape(),
            x_attn.shape(),
            gate_msa.shape()
        );
        let mul_msa = x_attn.broadcast_mul(&gate_msa)?;
        let x = res_msa.broadcast_add(&mul_msa)?;

        // MLP
        let res_mlp = x.clone();
        let x_norm = self.ff_norm.forward(&x)?;
        eprintln!(
            "DIT MLP: x_norm={:?}, scale_mlp={:?}, shift_mlp={:?}",
            x_norm.shape(),
            scale_mlp.shape(),
            shift_mlp.shape()
        );
        let x_norm = x_norm
            .broadcast_mul(&(scale_mlp.affine(1.0, 1.0)?))?
            .broadcast_add(&shift_mlp)?;
        let x_mlp = self.ff.forward(&x_norm)?;

        eprintln!(
            "DIT MLP ADD: res_mlp={:?}, x_mlp={:?}, gate_mlp={:?}",
            res_mlp.shape(),
            x_mlp.shape(),
            gate_mlp.shape()
        );
        let mul_mlp = x_mlp.broadcast_mul(&gate_mlp)?;
        let x = res_mlp.broadcast_add(&mul_mlp)?;

        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let sum: f32 = x_vec.iter().sum();
        let mean = sum / x_vec.len() as f32;
        eprintln!("DIT BLOCK mean={}, first 5={:?}", mean, &x_vec[0..5]);

        Ok(x)
    }
}

pub struct FeedForward {
    project_in: Linear,
    project_out: Linear,
    act: candle_nn::Activation,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        // FeedForward: Python Sequential(Linear, Gelu, Linear) -> Rust names match
        let inner_dim = dim * mult;
        let project_in = linear(dim, inner_dim, vb.pp("ff.0.0"))?;
        let project_out = linear(inner_dim, dim, vb.pp("ff.2"))?;
        Ok(Self {
            project_in,
            project_out,
            act: candle_nn::Activation::NewGelu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.project_in)?
            .apply(&self.act)?
            .apply(&self.project_out)
    }
}

pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
    scale: f64,
}

impl Attention {
    pub fn new(vb: VarBuilder, dim: usize, heads: usize, dim_head: usize) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let to_q = linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f64).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;

        let q = q
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;

        let (q, k) = if let Some((cos, sin)) = rope {
            (
                apply_rotary_pos_emb(&q, cos, sin)?,
                apply_rotary_pos_emb(&k, cos, sin)?,
            )
        } else {
            (q, k)
        };

        // Ensure contiguous layout for matmul
        let q = q.contiguous()?;
        let k = k.contiguous()?;

        let mut attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        let m = mask.unsqueeze(1)?.unsqueeze(1)?; // [B, 1, 1, N]
        let m_inv = (m.affine(-1.0, 1.0)? * 1e10)?;
        attn = candle_nn::ops::softmax(&attn.broadcast_sub(&m_inv)?, 3)?;

        let v = v.contiguous()?;
        let out = attn.matmul(&v)?;
        self.to_out.forward(
            &out.transpose(1, 2)?
                .reshape((b, n, self.heads * self.dim_head))?,
        )
    }
}

pub fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = x.dims4()?;

    // Split into Head 0 and others
    let x_h0 = x.narrow(1, 0, 1)?; // [B, 1, N, D]
    let x_rest = x.narrow(1, 1, h - 1)?; // [B, H-1, N, D]

    // Rotate Head 0
    // x_transformers matches frequencies to input shape by right-aligning
    // and uses [cos, cos, sin, sin] but interleave in pairs.
    // Actually x_transformers stack((freqs, freqs), -1).flatten(-2) -> [f1, f1, f2, f2]
    // And rotate_half: [-x2, x1, -x4, x3]

    // x_h0: [B, 1, N, D]
    // freqs (cos/sin): [1, 1, N, D]

    let cos = cos.narrow(0, 0, n)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, n)?.unsqueeze(0)?.unsqueeze(0)?;

    // GPT-J style rotation: adjacent pairs
    // Reshape [B, 1, N, D/2, 2]
    let x_reshaped = x_h0.reshape((b, 1, n, d / 2, 2))?;

    // x1 = x[..., 0], x2 = x[..., 1]
    let x1 = x_reshaped.narrow(4, 0, 1)?;
    let x2 = x_reshaped.narrow(4, 1, 1)?;

    // rotate_x = [-x2, x1]
    let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], 4)?.flatten_from(3)?; // Back to [B, 1, N, D]

    let x_cos = x_h0.broadcast_mul(&cos)?;
    let rot_sin = rotate_x.broadcast_mul(&sin)?;
    let x_h0_rot = x_cos.broadcast_add(&rot_sin)?;

    // Concatenate back
    Tensor::cat(&[&x_h0_rot, &x_rest], 1)
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / 10000.0f32.powf(i as f32 / dim as f32))
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, dim / 2), device)?;
        let t = Tensor::arange(0.0f32, max_seq_len as f32, device)?.reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&freqs)?;

        // Interleave frequencies: [f1, f1, f2, f2, ...]
        // freqs: [Seq, Dim/2]
        // unsqueeze(-1) -> [Seq, Dim/2, 1]
        // repeat -> [Seq, Dim/2, 2]
        // flatten -> [Seq, Dim]
        let freqs = freqs.unsqueeze(2)?.repeat((1, 1, 2))?.flatten_from(1)?;

        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }
}

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    let softplus = x.exp()?.broadcast_add(&Tensor::ones_like(x)?)?.log()?;
    let tanh = softplus.tanh()?;
    x.broadcast_mul(&tanh)
}

fn sinusoidal_embedding(x: &Tensor, dim: usize) -> Result<Tensor> {
    // x: [B] (time steps)
    // Output: [B, dim]
    let half_dim = dim / 2;
    let scale = 1000.0;
    let emb_factor = (10000.0f64).ln() / (half_dim as f64 - 1.0);

    // freqs = exp(arange(half_dim) * -emb_factor)
    let arange = Tensor::arange(0u32, half_dim as u32, x.device())?.to_dtype(DType::F32)?;
    let freqs = (arange * (-emb_factor))?.exp()?;

    // args = scale * x.unsqueeze(1) * freqs.unsqueeze(0)
    let x_uns = x.unsqueeze(1)?.to_dtype(DType::F32)?;
    let freqs_uns = freqs.unsqueeze(0)?;
    eprintln!(
        "SINUSOIDAL: x={:?}, x_uns={:?}, freqs_uns={:?}",
        x.shape(),
        x_uns.shape(),
        freqs_uns.shape()
    );
    let args = (x_uns * scale)?.broadcast_mul(&freqs_uns)?;

    let emb_sin = args.sin()?;
    let emb_cos = args.cos()?;

    Tensor::cat(&[emb_sin, emb_cos], 1)
}

pub struct CausalConvPositionEmbedding {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
}

impl CausalConvPositionEmbedding {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let conv_cfg = candle_nn::Conv1dConfig {
            groups: 16,
            padding: 0,
            ..Default::default()
        };
        // kernel_size=31. Python: padding=0 because we manually pad.
        // Weight names: conv1.0, conv2.0 because Sequential(Conv, Mish)
        // If weights are purely conv1.weight, conv1.bias, we need to check.
        // Python: self.conv1 = nn.Sequential(nn.Conv1d(...), nn.Mish())
        // So weights are conv1.0.weight.
        let conv1 = candle_nn::conv1d(dim, dim, 31, conv_cfg, vb.pp("conv1.0"))?;
        let conv2 = candle_nn::conv1d(dim, dim, 31, conv_cfg, vb.pp("conv2.0"))?;
        Ok(Self { conv1, conv2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, N, D]
        // Manual grouped convolution to check for backend issues
        let apply_manual_grouped_conv = |x: &Tensor, conv: &candle_nn::Conv1d| -> Result<Tensor> {
            let groups = 16;
            let weight = conv.weight();


            let bias = conv.bias();
            let x_chunks = x.chunk(groups, 1)?;
            let w_chunks = weight.chunk(groups, 0)?;
            let b_chunks = match bias {
                Some(b) => Some(b.chunk(groups, 0)?),
                None => None,
            };

            let mut outs = Vec::new();
            for i in 0..groups {
                let x_i = &x_chunks[i];
                let w_i = &w_chunks[i];
                // conv1d(weight, padding, stride, dilation, groups)
                // Here groups=1 because we split manually
                let out_i = x_i.conv1d(w_i, 0, 1, 1, 1)?;

                let out_i = match &b_chunks {
                    Some(bs) => out_i.broadcast_add(&bs[i].unsqueeze(0)?.unsqueeze(2)?)?,
                    None => out_i,
                };
                outs.push(out_i);
            }
            Tensor::cat(&outs, 1)
        };

        let x = x.transpose(1, 2)?; // [B, D, N]
        let (b, d, _) = x.dims3()?;
        let zeros = Tensor::zeros((b, d, 30), x.dtype(), x.device())?;

        let x = Tensor::cat(&[&zeros, &x], 2)?;
        let x = apply_manual_grouped_conv(&x, &self.conv1)?;
        let x = mish(&x)?;

        let x = Tensor::cat(&[&zeros, &x], 2)?;
        let x = apply_manual_grouped_conv(&x, &self.conv2)?;
        let x = mish(&x)?;

        x.transpose(1, 2) // [B, N, D]
    }
}

pub struct InputEmbedding {
    pub proj: Linear,
    pub conv_pos_embed: CausalConvPositionEmbedding,
}

impl InputEmbedding {
    pub fn new(vb: VarBuilder, mel_dim: usize, out_dim: usize) -> Result<Self> {
        let proj = linear(mel_dim * 4, out_dim, vb.pp("proj"))?;
        let conv_pos_embed = CausalConvPositionEmbedding::new(vb.pp("conv_pos_embed"), out_dim)?;
        Ok(Self {
            proj,
            conv_pos_embed,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor, mu: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let spks = spks.unsqueeze(2)?.repeat((1, 1, seq_len))?;
        let cat = Tensor::cat(&[x, cond, mu, &spks], 1)?;
        let x = self.proj.forward(&cat.transpose(1, 2)?)?;

        let pos = self.conv_pos_embed.forward(&x)?;
        x.add(&pos)
    }
}

pub struct DiT {
    input_embed: InputEmbedding,
    time_embed: TimestepEmbedding,
    transformer_blocks: Vec<DiTBlock>,
    pub norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
    rotary_embed: RotaryEmbedding,
}

impl DiT {
    pub fn new(vb: VarBuilder, cfg: &FlowConfig) -> Result<Self> {
        let input_embed = InputEmbedding::new(vb.pp("input_embed"), cfg.mel_dim, cfg.dim)?;
        let time_embed = TimestepEmbedding::new(vb.pp("time_embed"), 256, cfg.dim)?;
        let mut transformer_blocks = Vec::new();
        let vb_blocks = vb.pp("transformer_blocks");
        for i in 0..cfg.depth {
            transformer_blocks.push(DiTBlock::new(
                vb_blocks.pp(i),
                cfg.dim,
                cfg.heads,
                cfg.dim_head,
            )?);
        }
        let norm_out = AdaLayerNormZeroFinal::new(vb.pp("norm_out"), cfg.dim)?;
        let proj_out = linear(cfg.dim, cfg.mel_dim, vb.pp("proj_out"))?;

        let rotary_embed = RotaryEmbedding::new(cfg.dim_head, 4096, vb.device())?;
        Ok(Self {
            input_embed,
            time_embed,
            transformer_blocks,
            norm_out,
            proj_out,
            rotary_embed,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let x = self.input_embed.forward(x, cond, mu, spks)?;

        // TimestepEmbedding handles sinusoidal embedding internally now
        let t_emb = self.time_embed.forward(t)?;

        let rope = Some((self.rotary_embed.cos.clone(), self.rotary_embed.sin.clone()));

        let mut x = x;
        for block in self.transformer_blocks.iter() {
            x = block.forward(&x, &t_emb, mask, rope.as_ref())?;
        }

        let x = self.norm_out.forward(&x, &t_emb)?;

        let out = self.proj_out.forward(&x)?.transpose(1, 2)?;

        Ok(out)
    }
}

fn _t_to_sinusoidal(t: &Tensor, dim: usize) -> Result<Tensor> {
    let device = t.device();
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / 10000.0f32.powf(i as f32 / (half_dim as f32 - 1.0)))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;
    let outer = t
        .reshape((t.elem_count(), 1))?
        .matmul(&inv_freq.reshape((1, half_dim))?)?;
    Tensor::cat(&[outer.sin()?, outer.cos()?], 1)
}

pub struct ConditionalCFM {
    estimator: DiT,
    pub sigma: f64,
    pub cfg_strength: f64,
}

impl ConditionalCFM {
    pub fn new(
        _vb: VarBuilder,
        estimator: DiT,
        _ode_type: String,
        sigma: f64,
        cfg_strength: f64,
    ) -> Result<Self> {
        Ok(Self {
            estimator,
            sigma,
            cfg_strength,
        })
    }

    /// Flow matching forward pass with optional noise injection for parity testing.
    ///
    /// # Arguments
    /// * `mu` - Mean tensor from length regulator
    /// * `mask` - Attention mask
    /// * `n_timesteps` - ODE solver steps
    /// * `temperature` - Sampling temperature
    /// * `spks` - Optional speaker embedding
    /// * `cond` - Optional conditioning tensor
    /// * `noise` - Optional pre-generated noise for parity testing (use `None` for random)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        temperature: f64,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = mu.device();
        // Use injected noise for parity testing, or generate random noise
        // Clone inputs to owned handles to allow slicing/realignment
        let mut mu = mu.clone();
        let mut mask = mask.clone();
        let mut cond = cond.cloned(); // Option<Tensor> -> Option<Tensor>

        let mut x = match noise {
            Some(n) => (n * temperature)?,
            None => (mu.randn_like(0.0, 1.0)? * temperature)?,
        };
        let x_init = x.clone();

        // Align shapes if noise length differs from mu length
        let x_len = x.dim(2)?;
        let mu_len = mu.dim(2)?;
        if x_len != mu_len {
            let min_len = usize::min(x_len, mu_len);
            if x_len > min_len {
                x = x.narrow(2, 0, min_len)?;
            }
            if mu_len > min_len {
                mu = mu.narrow(2, 0, min_len)?;
            }
            // mask is [1, 1, T] or [1, T]?
            if mask.rank() >= 3 && mask.dim(2)? > min_len {
                 mask = mask.narrow(2, 0, min_len)?;
            } else if mask.rank() == 2 && mask.dim(1)? > min_len {
                 mask = mask.narrow(1, 0, min_len)?;
            }

            if let Some(c) = cond.take() {
                let c_len = c.dim(2)?;
                if c_len > min_len {
                    cond = Some(c.narrow(2, 0, min_len)?);
                } else {
                    cond = Some(c);
                }
            }
            // spks is [1, 80]. No time dim usually.
        }

        let mut t_span: Vec<f32> = (0..=n_timesteps)
            .map(|i| i as f32 / n_timesteps as f32)
            .collect();
        for t in t_span.iter_mut() {
            *t = 1.0 - (*t * 0.5 * std::f32::consts::PI).cos();
        }

        // 2. ODE Solver Loop (Euler)
        for i in 1..t_span.len() {
            let t_prev = t_span[i - 1];
            let t_curr = t_span[i];
            let dt = t_curr - t_prev; // Restore dt
            let t_tensor = Tensor::from_vec(vec![t_prev, t_prev], (2,), device)?; // [2]

            // CFG Guidance: Batch 2
            // inputs must be &[&Tensor]
            // x, mu, mask are owned Tensors -> use &x, &mu, &mask
            // spks is Option<&Tensor> (arg) -> need to handle
            // cond is Option<Tensor> (shadowed) -> need to handle

            let x_in = Tensor::cat(&[&x, &x], 0)?;
            let mask_in = Tensor::cat(&[&mask, &mask], 0)?;
            let mu_in = Tensor::cat(&[&mu, &mu.zeros_like()?], 0)?;

            // Handle spks (Option<&Tensor> arg -> Tensor)
            let spks_tensor = match spks {
                Some(s) => s.clone(),
                None => Tensor::zeros((1, 80), DType::F32, device)?,
            };
            // Ensure concatenated zeros match spks rank (2) not mu rank (3)
            let spks_in = Tensor::cat(&[&spks_tensor, &spks_tensor.zeros_like()?], 0)?;


            // Handle cond (Option<Tensor>)
            let cond_in_tensor = if let Some(c) = &cond {
                 Tensor::cat(&[c, &c.zeros_like()?], 0)?
            } else {
                 mu.zeros_like()? // Placeholder if cond is None
            };
            // forward expects &Tensor for cond.
            // cond_in_tensor is always initialized (zeros if None)
            let v = self
                .estimator
                .forward(&x_in, &mask_in, &mu_in, &t_tensor, &spks_in, &cond_in_tensor)?;
            let chunks = v.chunk(2, 0)?;
            let (v1, v2) = (&chunks[0], &chunks[1]);

            let cfg_rate = 0.7; // Hardcoded default or use config
            // v1 * (1.7) - v2 * (0.7)
            // Use broadcast_mul to be safe with Result vs Tensor return
            let v1_scaled = v1.broadcast_mul(&(Tensor::from_vec(vec![(1.0 + cfg_rate) as f32], 1, device)?))?;
            let v2_scaled = v2.broadcast_mul(&(Tensor::from_vec(vec![cfg_rate as f32], 1, device)?))?;
            let v_cfg = (v1_scaled - v2_scaled)?;

            let d = v_cfg.broadcast_mul(&(Tensor::from_vec(vec![dt], (1,), device)?))?;
            x = (x + d)?;
        }

        // Debug
        if std::env::var("SAVE_FLOW_DEBUG").is_ok() {
            eprintln!("Saving flow debug tensors to rust_flow_debug.safetensors...");
            let mut debug_map = std::collections::HashMap::new();
            debug_map.insert("mu".to_string(), mu.clone());
            debug_map.insert("mask".to_string(), mask.clone());
            // handle option spks
            if let Some(s) = spks {
                debug_map.insert("spks".to_string(), s.clone());
            }
            // handle option cond
            if let Some(c) = cond {
                 debug_map.insert("cond".to_string(), c);
            }
            debug_map.insert("x_init".to_string(), x_init);
            debug_map.insert("flow_output".to_string(), x.clone());
            candle_core::safetensors::save(&debug_map, "rust_flow_debug.safetensors")?;
        }

        Ok(x)
    }
}
