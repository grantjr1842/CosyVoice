//! Complete CosyVoice Flow module with all layers for speech token to mel conversion.
//!
//! This module implements the full Flow pipeline:
//! 1. Token embedding
//! 2. PreLookahead processing
//! 3. Speaker embedding projection
//! 4. DiT decoder with ODE solver

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder};

use crate::flow::{ConditionalCFM, DiT, FlowConfig};

/// Configuration for the complete Flow model
#[derive(Debug, Clone)]
pub struct CosyVoiceFlowConfig {
    pub input_size: usize,         // Token embedding dim (80 for Fun-CosyVoice3-0.5B)
    pub output_size: usize,        // Mel dim (80)
    pub spk_embed_dim: usize,      // Speaker embedding dim (192)
    pub vocab_size: usize,         // Speech token vocab (6561 for Fun-CosyVoice3-0.5B)
    pub token_mel_ratio: usize,    // Upsampling ratio (2)
    pub pre_lookahead_len: usize,  // Lookahead context (3)
    pub pre_lookahead_channels: usize, // Intermediate channels in pre-lookahead (1024)
}

impl Default for CosyVoiceFlowConfig {
    fn default() -> Self {
        Self {
            input_size: 80,           // Fun-CosyVoice3-0.5B uses 80
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561,          // Fun-CosyVoice3-0.5B uses 6561
            token_mel_ratio: 2,
            pre_lookahead_len: 3,
            pre_lookahead_channels: 1024,
        }
    }
}

/// Pre-lookahead layer for causal processing of token embeddings
pub struct PreLookaheadLayer {
    conv1: Conv1d,
    conv2: Conv1d,
    pre_lookahead_len: usize,
}

impl PreLookaheadLayer {
    pub fn new(vb: VarBuilder, in_channels: usize, channels: usize, pre_lookahead_len: usize) -> Result<Self> {
        let conv1_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };
        let conv2_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };

        // kernel_size = pre_lookahead_len + 1 for conv1
        let conv1 = conv1d(in_channels, channels, pre_lookahead_len + 1, conv1_cfg, vb.pp("conv1"))?;
        // kernel_size = 3 for conv2
        let conv2 = conv1d(channels, in_channels, 3, conv2_cfg, vb.pp("conv2"))?;

        Ok(Self {
            conv1,
            conv2,
            pre_lookahead_len,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `inputs` - Input tensor [batch, seq_len, channels]
    /// * `context` - Optional lookahead context [batch, pre_lookahead_len, channels]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, channels]
    pub fn forward(&self, inputs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        // Transpose to [batch, channels, seq_len] for Conv1d
        let outputs = inputs.transpose(1, 2)?;

        // Build padded tensor with lookahead
        let outputs = if let Some(ctx) = context {
            let ctx = ctx.transpose(1, 2)?;
            let ctx_len = ctx.dim(2)?;
            let combined = Tensor::cat(&[&outputs, &ctx], 2)?;
            // Pad remaining lookahead with zeros
            if ctx_len < self.pre_lookahead_len {
                combined.pad_with_zeros(2, 0, self.pre_lookahead_len - ctx_len)?
            } else {
                combined
            }
        } else {
            // Pad with zeros on the right
            outputs.pad_with_zeros(2, 0, self.pre_lookahead_len)?
        };

        // Conv1 + LeakyReLU
        let outputs = self.conv1.forward(&outputs)?;
        let outputs = leaky_relu(&outputs, 0.01)?;

        // Pad left for causal conv2 (kernel_size - 1 = 2)
        let outputs = outputs.pad_with_zeros(2, 2, 0)?;
        let outputs = self.conv2.forward(&outputs)?;

        // Transpose back to [batch, seq_len, channels]
        let outputs = outputs.transpose(1, 2)?;

        // Residual connection
        outputs.add(inputs)
    }
}

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?.affine(negative_slope, 0.0)?;
    pos.add(&neg)
}

/// Complete CosyVoice Flow model
pub struct CosyVoiceFlow {
    /// Token embedding layer
    input_embedding: Embedding,
    /// Speaker embedding projection
    spk_embed_affine_layer: Linear,
    /// Pre-lookahead processing layer
    pre_lookahead_layer: PreLookaheadLayer,
    /// DiT decoder with CFM
    decoder: ConditionalCFM,
    /// Configuration
    config: CosyVoiceFlowConfig,
    /// Device
    device: Device,
}

impl CosyVoiceFlow {
    /// Create a new Flow model from safetensors weights
    pub fn new(
        flow_config: CosyVoiceFlowConfig,
        dit_config: &FlowConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Token embedding
        let input_embedding = embedding(
            flow_config.vocab_size,
            flow_config.input_size,
            vb.pp("input_embedding"),
        )?;

        // Speaker embedding projection
        let spk_embed_affine_layer = linear(
            flow_config.spk_embed_dim,
            flow_config.output_size,
            vb.pp("spk_embed_affine_layer"),
        )?;

        // Pre-lookahead layer with intermediate channels
        let pre_lookahead_layer = PreLookaheadLayer::new(
            vb.pp("pre_lookahead_layer"),
            flow_config.input_size,
            flow_config.pre_lookahead_channels,
            flow_config.pre_lookahead_len,
        )?;

        // DiT decoder
        let dit = DiT::new(vb.pp("decoder.estimator"), dit_config)?;
        let decoder = ConditionalCFM::new(vb.clone(), dit, "cosine".to_string(), 0.0, 0.7)?;

        Ok(Self {
            input_embedding,
            spk_embed_affine_layer,
            pre_lookahead_layer,
            decoder,
            config: flow_config,
            device,
        })
    }

    /// Run flow inference to generate mel spectrogram from speech tokens
    ///
    /// # Arguments
    /// * `token` - Speech tokens [batch, token_len]
    /// * `prompt_token` - Prompt speech tokens [batch, prompt_len]
    /// * `prompt_feat` - Prompt mel features [batch, mel_len, mel_dim]
    /// * `embedding` - Speaker embedding [batch, spk_embed_dim]
    /// * `n_timesteps` - Number of ODE solver steps (default 10)
    ///
    /// # Returns
    /// Mel spectrogram [batch, mel_dim, mel_len]
    pub fn inference(
        &self,
        token: &Tensor,
        prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        // Normalize speaker embedding
        let embedding_norm = l2_normalize(embedding)?;
        let embedding_proj = self.spk_embed_affine_layer.forward(&embedding_norm)?;

        // Concatenate prompt and target tokens
        let combined_token = Tensor::cat(&[prompt_token, token], 1)?;
        let token_len = combined_token.dim(1)?;

        // Embed tokens
        let token_emb = self.input_embedding.forward(&combined_token)?;

        // Apply pre-lookahead layer (finalize mode, no context)
        let h = self.pre_lookahead_layer.forward(&token_emb, None)?;

        // Repeat interleave for token_mel_ratio
        let h = repeat_interleave(&h, self.config.token_mel_ratio, 1)?;

        let prompt_mel_len = prompt_feat.dim(1)?;
        let total_mel_len = h.dim(1)?;
        let target_mel_len = total_mel_len - prompt_mel_len;

        // Build conditioning tensor
        let mut conds = Tensor::zeros(
            (1, prompt_mel_len + target_mel_len, self.config.output_size),
            DType::F32,
            &self.device,
        )?;

        // Copy prompt features into conds
        // conds[:, :prompt_mel_len] = prompt_feat
        if prompt_mel_len > 0 {
            // For simplicity, we'll build conds by concatenating
            let zeros_part = Tensor::zeros(
                (1, target_mel_len, self.config.output_size),
                DType::F32,
                &self.device,
            )?;
            conds = Tensor::cat(&[prompt_feat, &zeros_part], 1)?;
        }
        let conds = conds.transpose(1, 2)?; // [batch, mel_dim, mel_len]

        // Create mask (all ones for now)
        let mask = Tensor::ones((1, total_mel_len), DType::F32, &self.device)?;

        // mu = h transposed
        let mu = h.transpose(1, 2)?; // [batch, hidden_dim, mel_len]

        // Run CFM decoder
        let feat = self.decoder.forward(
            &mu,
            &mask,
            n_timesteps,
            1.0, // temperature
            Some(&embedding_proj),
            Some(&conds),
        )?;

        // Extract only the target portion (after prompt)
        let feat = feat.narrow(2, prompt_mel_len, target_mel_len)?;

        Ok(feat)
    }
}

/// L2 normalize along dimension 1
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norm = norm.broadcast_add(&Tensor::new(&[1e-8f32], x.device())?)?;
    x.broadcast_div(&norm)
}

/// Repeat interleave: repeat each element along dim
fn repeat_interleave(x: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape().dims();
    let n = shape[dim];

    // Reshape, expand, and flatten back
    // For dim=1: [B, N, D] -> [B, N, 1, D] -> [B, N, R, D] -> [B, N*R, D]
    if dim == 1 {
        let (b, n_dim, d) = (shape[0], shape[1], shape[2]);
        let x = x.reshape((b, n_dim, 1, d))?;
        let x = x.repeat((1, 1, repeats, 1))?;
        x.reshape((b, n_dim * repeats, d))
    } else {
        // Generic fallback
        Err(candle_core::Error::Msg(format!("repeat_interleave not implemented for dim={}", dim)))
    }
}
