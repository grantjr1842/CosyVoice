use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::ops::sdpa;
use candle_nn::{linear, LayerNorm, Linear, VarBuilder};

// Force rebuild marker: REBUILD_V1
const _FORCE_REBUILD: u32 = 1;

use serde::Deserialize;

fn log_v_stats(name: &str, t: &Tensor) -> Result<()> {
    let t = if t.dtype() != DType::F32 {
        t.to_dtype(DType::F32)?
    } else {
        t.clone()
    };
    let flat = t.flatten_all()?;
    let vec = flat.to_vec1::<f32>()?;
    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
    eprintln!(
        "    [V DEBUG] {} stats: min={:.6}, max={:.6}, mean={:.6}, first 5={:?}",
        name,
        min,
        max,
        mean,
        &vec[0..usize::min(vec.len(), 5)]
    );
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub mel_dim: usize,
    pub mu_dim: usize,
    pub spk_dim: usize,
    pub static_chunk_size: usize,
    pub num_decoding_left_chunks: isize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            depth: 22,
            heads: 16,
            dim_head: 64,
            mel_dim: 80,
            mu_dim: 80,
            spk_dim: 80,
            static_chunk_size: 50,
            num_decoding_left_chunks: -1,
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
        chunk_mask: Option<&Tensor>,
        rope: Option<&Tensor>,
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
        let x_norm = x_norm
            .broadcast_mul(&(scale_msa.affine(1.0, 1.0)?))?
            .broadcast_add(&shift_msa)?;
        let x_attn = self.attn.forward(&x_norm, mask, chunk_mask, rope)?;
        let mul_msa = x_attn.broadcast_mul(&gate_msa)?;
        let x = res_msa.broadcast_add(&mul_msa)?;

        // MLP
        let res_mlp = x.clone();
        let x_norm = self.ff_norm.forward(&x)?;
        let x_norm = x_norm
            .broadcast_mul(&(scale_mlp.affine(1.0, 1.0)?))?
            .broadcast_add(&shift_mlp)?;
        let x_mlp = self.ff.forward(&x_norm)?;
        let mul_mlp = x_mlp.broadcast_mul(&gate_mlp)?;
        let x = res_mlp.broadcast_add(&mul_mlp)?;

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
    use_flash_attn: bool,
}

impl Attention {
    pub fn new(vb: VarBuilder, dim: usize, heads: usize, dim_head: usize) -> Result<Self> {
        let inner_dim = heads * dim_head;
        let to_q = linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, dim, vb.pp("to_out.0"))?;
        let use_flash_attn = vb.device().is_cuda() || vb.device().is_metal();
        static LOG_ONCE: std::sync::Once = std::sync::Once::new();
        LOG_ONCE.call_once(|| {
            eprintln!("    [Attn DEBUG] Attention::new called: device={:?}, use_flash_attn={}", vb.device(), use_flash_attn);
        });

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f64).sqrt(),
            use_flash_attn,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        chunk_mask: Option<&Tensor>,
        rope: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;

        let q = if let Some(rope_freqs) = rope {
            apply_rotary_pos_emb(&q, rope_freqs)?
        } else {
            q
        };
        let k = if let Some(rope_freqs) = rope {
            apply_rotary_pos_emb(&k, rope_freqs)?
        } else {
            k
        };

        eprintln!("    [Attn DEBUG] q after RoPE shape: {:?}", q.shape());
        let q = q
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, n, self.heads, self.dim_head))?
            .transpose(1, 2)?;
        eprintln!("    [Attn DEBUG] q after reshape/transpose: {:?}", q.shape());
        eprintln!("    [Attn DEBUG] k shape: {:?}, v shape: {:?}", k.shape(), v.shape());

        let scale_f32 = self.scale as f32;

        let attn_mask = if let Some(chunk_mask) = chunk_mask {
            eprintln!("    [Attn DEBUG] chunk_mask shape: {:?}", chunk_mask.shape());
            // chunk_mask is [B, 1, N, N]
            // For sdpa, we need an additive mask (0 for valid, -inf for masked)
            let chunk_inv = chunk_mask
                .affine(-1.0, 1.0)?   // 1 -> 0, 0 -> 1
                .affine(1e10, 0.0)?   // multiply by large value
                .neg()?;              // 0 -> 0, 1e10 -> -1e10
            Some(chunk_inv)
        } else {
            eprintln!("    [Attn DEBUG] mask shape: {:?}", mask.shape());
            let m = mask.unsqueeze(1)?.unsqueeze(1)?; // [B, 1, 1, N]
            let m_inv = (m.affine(-1.0, 1.0)? * 1e10)?.neg()?;
            Some(m_inv)
        };

        let out = if self.use_flash_attn {
            match sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attn_mask.as_ref(),
                false,
                scale_f32,
                1.0,
            ) {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("    [Attn WARNING] sdpa failed: {:?}, falling back to manual path", e);
                    let q = q.contiguous()?;
                    let k = k.contiguous()?;
                    let k_t = k.transpose(2, 3)?;
                    let attn = q.matmul(&k_t)?;
                    let attn = (attn * self.scale)?;

                    let attn = if let Some(mask) = attn_mask {
                        attn.broadcast_add(&mask)?
                    } else {
                        attn
                    };
                    let attn = candle_nn::ops::softmax(&attn, 3)?;
                    attn.matmul(&v.contiguous()?)?
                }
            }
        } else {
            let q = q.contiguous()?;
            let k = k.contiguous()?;
            let k_t = k.transpose(2, 3)?;
            let attn = q.matmul(&k_t)?;
            let mut attn = (attn * self.scale)?;

            let attn = if let Some(mask) = attn_mask {
                attn.broadcast_add(&mask)?
            } else {
                attn
            };
            let attn = candle_nn::ops::softmax(&attn, 3)?;
            attn.matmul(&v.contiguous()?)?
        };

        self.to_out.forward(
            &out.transpose(1, 2)?
                .reshape((b, n, self.heads * self.dim_head))?,
        )
    }
}


pub struct RotaryEmbedding {
    freqs: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        // Python x_transformers uses interleaved layout: [f0, f0, f1, f1, f2, f2, ...]
        // Each frequency appears twice consecutively
        let inv_freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / 10000.0f32.powf(i as f32 / dim as f32))
            .collect();
        // inv_freqs has dim/2 values

        // Build interleaved freqs: for each position, repeat each freq twice
        let mut freqs_data = Vec::with_capacity(max_seq_len * dim);
        for pos in 0..max_seq_len {
            let t = pos as f32;
            for &inv_f in &inv_freqs {
                let freq_val = t * inv_f;
                freqs_data.push(freq_val); // First copy
                freqs_data.push(freq_val); // Second copy (interleaved)
            }
        }

        let freqs = Tensor::from_vec(freqs_data, (max_seq_len, dim), device)?;
        Ok(Self { freqs })
    }
}


pub fn apply_rotary_pos_emb(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    match x.rank() {
        3 => apply_rotary_pos_emb_flat(x, freqs),
        4 => {
            let (b, h, n, d) = x.dims4()?;
            let inner = h * d;
            let x_flat = x.transpose(1, 2)?.reshape((b, n, inner))?;
            let rotated = apply_rotary_pos_emb_flat(&x_flat, freqs)?;
            rotated.reshape((b, n, h, d))?.transpose(1, 2)
        }
        _ => Err(candle_core::Error::Msg(
            "apply_rotary_pos_emb expects a rank-3 or rank-4 tensor".into(),
        )),
    }
}

fn apply_rotary_pos_emb_flat(x: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let (_b, n, d) = x.dims3()?;
    let device = x.device();
    eprintln!(
        "    [RoPE DEBUG] x shape: {:?}, freqs shape: {:?}",
        x.shape(),
        freqs.shape()
    );
    let freq = freqs
        .narrow(0, 0, n)?
        .to_dtype(x.dtype())?
        .to_device(&device)?
        .unsqueeze(0)?;
    eprintln!("    [RoPE DEBUG] freq (after unsqueeze(0)) shape: {:?}", freq.shape());
    let cos = freq.cos()?;
    let sin = freq.sin()?;
    let rot_dim = cos.dim(2)?;
    eprintln!("    [RoPE DEBUG] rot_dim={}, d={}", rot_dim, d);

    let x_rot = x.narrow(2, 0, rot_dim)?;
    eprintln!("    [RoPE DEBUG] x_rot shape: {:?}", x_rot.shape());
    let rotated = rotate_half(&x_rot)?;
    eprintln!("    [RoPE DEBUG] rotated shape: {:?}", rotated.shape());
    let rotated = rotated.broadcast_mul(&sin)?;
    let mut x_rotated = x_rot.broadcast_mul(&cos)?;
    x_rotated = x_rotated.broadcast_add(&rotated)?;
    eprintln!("    [RoPE DEBUG] x_rotated shape: {:?}", x_rotated.shape());

    if rot_dim < d {
        let rest = x.narrow(2, rot_dim, d - rot_dim)?;
        eprintln!("    [RoPE DEBUG] rest shape: {:?}", rest.shape());
        let result = Tensor::cat(&[&x_rotated, &rest], 2)?;
        eprintln!("    [RoPE DEBUG] result shape: {:?}", result.shape());
        Ok(result)
    } else {
        Ok(x_rotated)
    }
}


fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape().dims().to_vec();
    let last_dim = *dims.last().ok_or_else(|| {
        candle_core::Error::Msg("rotate_half requires at least one dimension".into())
    })?;
    if last_dim % 2 != 0 {
        return Err(candle_core::Error::Msg(
            "rotate_half expects an even number of channels".into(),
        ));
    }
    let half = last_dim / 2;
    let mut pair_shape = dims.clone();
    pair_shape.pop();
    pair_shape.push(half);
    pair_shape.push(2);
    let x_pairs = x.reshape(pair_shape.as_slice())?;
    let last_idx = pair_shape.len() - 1;
    let x_even = x_pairs.narrow(last_idx, 0, 1)?;
    let x_odd = x_pairs.narrow(last_idx, 1, 1)?;
    let rotated = Tensor::cat(&[&x_odd.neg()?, &x_even], last_idx)?;
    rotated.reshape(dims.as_slice())
}

fn collapse_mask(mask: &Tensor) -> Result<Tensor> {
    match mask.rank() {
        3 => {
            if mask.dim(1)? == 1 {
                let dims = (mask.dim(0)?, mask.dim(2)?);
                mask.reshape(&[dims.0, dims.1])
            } else {
                let idx = mask.rank() - 1;
                let dims = (mask.dim(0)?, mask.dim(idx)?);
                mask.reshape(&[dims.0, dims.1])
            }
        }
        2 => Ok(mask.clone()),
        _ => {
            let idx = mask.rank() - 1;
            let dims = (mask.dim(0)?, mask.dim(idx)?);
            mask.reshape(&[dims.0, dims.1])
        }
    }
}

fn subsequent_chunk_mask(
    size: usize,
    chunk_size: usize,
    num_left_chunks: isize,
    device: &Device,
) -> Result<Tensor> {
    if chunk_size == 0 {
        return Tensor::ones((size, size), DType::F32, device);
    }
    let mut data = vec![0f32; size * size];
    for i in 0..size {
        let block_idx = i / chunk_size;
        let start = if num_left_chunks < 0 {
            0
        } else {
            let left_chunks = block_idx.saturating_sub(num_left_chunks as usize);
            left_chunks * chunk_size
        };
        let start = start.min(size);
        let end = ((block_idx + 1) * chunk_size).min(size);
        for j in start..end {
            data[i * size + j] = 1.0;
        }
    }
    Tensor::from_vec(data, (size, size), device)
}

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    let softplus = x.exp()?.broadcast_add(&Tensor::ones_like(x)?)?.log()?;
    let tanh = softplus.tanh()?;
    x.broadcast_mul(&tanh)
}

fn sinusoidal_embedding(x: &Tensor, dim: usize) -> Result<Tensor> {
    // x: [B] (time steps in [0, 1])
    // Output: [B, dim]
    let half_dim = dim / 2;
    let device = x.device();

    // freqs = exp(-ln(10000.0) * arange(half_dim) / half_dim)
    let arange = Tensor::arange(0.0f32, half_dim as f32, device)?;
    let denom = (half_dim as f32 - 1.0).max(1.0);
    let inv_freq = arange
        .affine((-(10000.0f32.ln()) / denom) as f64, 0.0)?
        .exp()?;

    // args = x.unsqueeze(1) * inv_freq.unsqueeze(0)
    let x_uns = x.unsqueeze(1)?.to_dtype(DType::F32)?;
    let x_scaled = x_uns.affine(1000.0, 0.0)?;
    let args = x_scaled.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let emb_sin = args.sin()?;
    let emb_cos = args.cos()?;

    let mut emb = Tensor::cat(&[emb_sin, emb_cos], 1)?;
    if dim % 2 == 1 {
        emb = Tensor::cat(&[&emb, &emb.narrow(1, 0, 1)?.zeros_like()?], 1)?;
    }
    Ok(emb)
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
    pub fn new(
        vb: VarBuilder,
        mel_dim: usize,
        text_dim: usize,
        spk_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let in_dim = mel_dim * 2 + text_dim + spk_dim;
        let proj = linear(in_dim, out_dim, vb.pp("proj"))?;
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
    static_chunk_size: usize,
    num_decoding_left_chunks: isize,
}

impl DiT {
    pub fn new(vb: VarBuilder, cfg: &FlowConfig) -> Result<Self> {
        let input_embed = InputEmbedding::new(
            vb.pp("input_embed"),
            cfg.mel_dim,
            cfg.mu_dim,
            cfg.spk_dim,
            cfg.dim,
        )?;
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
            static_chunk_size: cfg.static_chunk_size,
            num_decoding_left_chunks: cfg.num_decoding_left_chunks,
        })
    }

    fn build_chunk_mask(&self, mask: &Tensor) -> Result<Option<Tensor>> {
        if self.static_chunk_size == 0 {
            return Ok(None);
        }
        let rank = mask.rank();
        let seq_len = match rank {
            2 => mask.dim(1)?,
            3 => mask.dim(2)?,
            _ => {
                return Err(candle_core::Error::Msg(
                    "Chunk mask expects mask of rank 2 or 3".into(),
                ))
            }
        };
        let chunk = subsequent_chunk_mask(
            seq_len,
            self.static_chunk_size,
            self.num_decoding_left_chunks,
            mask.device(),
        )?
        .unsqueeze(0)?
        .unsqueeze(1)?;
        let mask_float = mask.to_dtype(DType::F32)?;
        let mask_expand = if rank == 2 {
            mask_float.unsqueeze(1)?.unsqueeze(1)?
        } else {
            mask_float.unsqueeze(2)?
        };
        let combined = mask_expand.broadcast_mul(&chunk)?;
        Ok(Some(combined))
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
        // Default to non-streaming mode (no chunk masking) to match Python inference
        self.forward_with_streaming(x, mask, mu, t, spks, cond, false)
    }

    pub fn forward_with_streaming(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let mut x = self.input_embed.forward(x, cond, mu, spks)?;

        // TimestepEmbedding handles sinusoidal embedding internally now
        let t_emb = self.time_embed.forward(t)?;
        log_v_stats("t_emb", &t_emb)?;

        let rope = Some(&self.rotary_embed.freqs);

        // Only use chunk masking in streaming mode
        // In non-streaming mode (default), chunk_mask is None - matching Python streaming=False
        let chunk_mask = if streaming {
            self.build_chunk_mask(mask)?
        } else {
            None
        };

        for block in self.transformer_blocks.iter() {
            x = block.forward(&x, &t_emb, mask, chunk_mask.as_ref(), rope)?;
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
        let mut mask_flat = collapse_mask(mask)?;
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
            if mask_flat.dim(1)? > min_len {
                mask_flat = mask_flat.narrow(1, 0, min_len)?;
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

        let max_steps = std::env::var("FLOW_DEBUG_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let log_v = std::env::var("FLOW_DEBUG_LOG_V").is_ok();
        let save_steps = std::env::var("SAVE_FLOW_STEP_TENSORS").is_ok();
        let mut step_debug_map = if save_steps {
            Some(std::collections::HashMap::new())
        } else {
            None
        };

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

            if let Some(max) = max_steps {
                if i > max {
                    eprintln!(
                        "    [Flow parity] reached debug max steps ({}) -> stopping early",
                        max
                    );
                    break;
                }
            }
            eprintln!("    [Flow parity] solver step {}/{}", i, t_span.len() - 1);

            let x_in = Tensor::cat(&[&x, &x], 0)?;
            let mask_in = Tensor::cat(&[&mask_flat, &mask_flat], 0)?;
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
            let v = self.estimator.forward(
                &x_in,
                &mask_in,
                &mu_in,
                &t_tensor,
                &spks_in,
                &cond_in_tensor,
            )?;
            let chunks = v.chunk(2, 0)?;
            let (v1, v2) = (&chunks[0], &chunks[1]);

            let cfg_rate = 0.7;
            // v1 * (1.7) - v2 * (0.7)
            // Use broadcast_mul to be safe with Result vs Tensor return
            let v1_scaled =
                v1.broadcast_mul(&(Tensor::from_vec(vec![(1.0 + cfg_rate) as f32], 1, device)?))?;
            let v2_scaled =
                v2.broadcast_mul(&(Tensor::from_vec(vec![cfg_rate as f32], 1, device)?))?;
            let v_cfg = (v1_scaled - v2_scaled)?;

            if log_v {
                log_v_stats("rust_v1", v1)?;
                log_v_stats("rust_v2", v2)?;
                log_v_stats("rust_v_cfg", &v_cfg)?;
            }

            if let Some(map) = step_debug_map.as_mut() {
                map.insert(format!("step{}_v1", i), v1.clone());
                map.insert(format!("step{}_v2", i), v2.clone());
                map.insert(format!("step{}_v_cfg", i), v_cfg.clone());
                if let Ok(dt_tensor) = Tensor::from_vec(vec![dt], (1,), device) {
                    map.insert(format!("step{}_dt", i), dt_tensor);
                }
            }

            let d = v_cfg.broadcast_mul(&(Tensor::from_vec(vec![dt], (1,), device)?))?;
            x = (x + d)?;
            if let Some(map) = step_debug_map.as_mut() {
                map.insert(format!("step{}_x", i), x.clone());
            }
        }

        if let Some(map) = step_debug_map {
            candle_core::safetensors::save(&map, "rust_flow_steps.safetensors")?;
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
