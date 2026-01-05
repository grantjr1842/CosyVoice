use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding; // Using standard RotaryEmbedding if compatible, or usually manual for Qwen
use std::sync::Arc;

// Copying RotaryEmbedding from qwen.rs or defining a compatible one
#[derive(Debug, Clone)]
struct RotaryEmbeddingFull {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbeddingFull {
    fn new(rope_theta: f32, head_dim: usize, max_position_embeddings: usize, device: &Device) -> Result<Self> {
        let dim = head_dim;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(scale: Tensor, eps: f64) -> Self {
        Self { scale, eps }
    }

    fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let scale = qtensor.dequantize(device)?;
        Ok(Self::new(scale, eps))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(x.rank() - 1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(x.rank() - 1)? / (hidden_size as f64))?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.scale)
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct QMatMulWrapper {
    inner: QMatMul,
}

impl Module for QMatMulWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: QMatMulWrapper,
    up_proj: QMatMulWrapper,
    down_proj: QMatMulWrapper,
    act_fn: candle_nn::Activation,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Attention {
    q_proj: QMatMulWrapper,
    k_proj: QMatMulWrapper,
    v_proj: QMatMulWrapper,
    o_proj: QMatMulWrapper,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbeddingFull>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Standard attention
        let key_states = candle_transformers::utils::repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?;

        attn_output.apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub struct Model {
    pub embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    _device: Device,
}

impl Model {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Metadata extraction
        let hidden_size = ct.metadata["llama.embedding_length"].to_u32()? as usize;
        let num_hidden_layers = ct.metadata["llama.block_count"].to_u32()? as usize;
        let num_attention_heads = ct.metadata["llama.attention.head_count"].to_u32()? as usize;
        let num_key_value_heads = ct.metadata["llama.attention.head_count_kv"].to_u32()? as usize;
        let rope_theta = ct.metadata.get("llama.rope.freq_base").and_then(|v| v.to_f32().ok()).unwrap_or(10000.0);
        let max_position_embeddings = ct.metadata.get("llama.context_length").and_then(|v| v.to_u32().ok()).unwrap_or(4096) as usize;
        let rms_norm_eps = ct.metadata["llama.attention.layer_norm_rms_epsilon"].to_f32()? as f64;
        let vocab_size = ct.metadata.get("vocab_size").and_then(|v| v.to_u32().ok()).or_else(|| ct.metadata.get("llama.vocab_size").and_then(|v| v.to_u32().ok())).unwrap_or(151936) as usize; // Check typical qwen vocab size

        let head_dim = hidden_size / num_attention_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;

        // Load Embedding
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = Embedding::new(tok_embeddings_q.dequantize(device)?, vocab_size);

        // Load Norm
        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps, device)?;

        // Rotary
        let rotary_emb = Arc::new(RotaryEmbeddingFull::new(rope_theta, head_dim, max_position_embeddings, device)?);

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for layer_idx in 0..num_hidden_layers {
             let prefix = format!("blk.{}", layer_idx);

             // Attention
             let q_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.attn_q.weight", prefix), device)?)? };
             let k_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.attn_k.weight", prefix), device)?)? };
             let v_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.attn_v.weight", prefix), device)?)? };
             let o_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.attn_output.weight", prefix), device)?)? };

             let self_attn = Attention {
                 q_proj,
                 k_proj,
                 v_proj,
                 o_proj,
                 num_heads: num_attention_heads,
                 num_kv_heads: num_key_value_heads,
                 num_kv_groups,
                 head_dim,
                 hidden_size,
                 rotary_emb: rotary_emb.clone(),
                 kv_cache: None,
             };

             // MLP
             let gate_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.ffn_gate.weight", prefix), device)?)? };
             let down_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.ffn_down.weight", prefix), device)?)? };
             let up_proj = QMatMulWrapper { inner: QMatMul::from_qtensor(ct.tensor(reader, &format!("{}.ffn_up.weight", prefix), device)?)? };

             let mlp = Mlp {
                 gate_proj,
                 up_proj,
                 down_proj,
                 act_fn: candle_nn::Activation::Silu, // Qwen usually uses Silu
             };

             let input_layernorm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{}.attn_norm.weight", prefix), device)?, rms_norm_eps, device)?;
             let post_attention_layernorm = RmsNorm::from_qtensor(ct.tensor(reader, &format!("{}.ffn_norm.weight", prefix), device)?, rms_norm_eps, device)?;

             layers.push(DecoderLayer {
                 self_attn,
                 mlp,
                 input_layernorm,
                 post_attention_layernorm,
             });
        }

        Ok(Self {
            embed_tokens: tok_embeddings,
            layers,
            norm,
            _device: device.clone(),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn forward_embeds(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

pub struct ModelForCausalLM {
    pub base_model: Model,
    lm_head: QMatMulWrapper,
}

impl ModelForCausalLM {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device
    ) -> Result<Self> {
        let base_model = Model::from_gguf(ct, reader, device)?;
        let lm_head_q = ct.tensor(reader, "output.weight", device)?;
        let lm_head = QMatMulWrapper { inner: QMatMul::from_qtensor(lm_head_q)? };

        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn forward_embeds(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len, _hidden) = inputs_embeds.dims3()?;
        self.base_model
            .forward_embeds(inputs_embeds, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}
