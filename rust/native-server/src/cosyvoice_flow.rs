//! Complete CosyVoice Flow module with all layers for speech token to mel conversion.
//!
//! This module implements the full Flow pipeline:
//! 1. Token embedding
//! 2. PreLookahead processing
//! 3. Speaker embedding projection
//! 4. DiT decoder with ODE solver

use candle_core::{Device, Result, Tensor};
use candle_nn::{
    conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder,
};

use crate::flow::{ConditionalCFM, DiT, FlowConfig};

/// Configuration for the complete Flow model
#[derive(Debug, Clone)]
pub struct CosyVoiceFlowConfig {
    pub input_size: usize,      // Token embedding dim (80 for Fun-CosyVoice3-0.5B)
    pub output_size: usize,     // Mel dim (80)
    pub spk_embed_dim: usize,   // Speaker embedding dim (192)
    pub vocab_size: usize,      // Speech token vocab (6561 for Fun-CosyVoice3-0.5B)
    pub token_mel_ratio: usize, // Upsampling ratio (2)
    pub pre_lookahead_len: usize, // Lookahead context (3)
    pub pre_lookahead_channels: usize, // Intermediate channels in pre-lookahead (1024)
    pub chunk_size: usize,      // Base chunk size (25)
    pub num_decoding_left_chunks: isize, // -1 means unlimited
}

impl Default for CosyVoiceFlowConfig {
    fn default() -> Self {
        Self {
            input_size: 80, // Fun-CosyVoice3-0.5B uses 80
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561, // Fun-CosyVoice3-0.5B uses 6561
            token_mel_ratio: 2,
            pre_lookahead_len: 3,
            pre_lookahead_channels: 1024,
            chunk_size: 25,
            num_decoding_left_chunks: -1,
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
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        channels: usize,
        pre_lookahead_len: usize,
    ) -> Result<Self> {
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
        let conv1 = conv1d(
            in_channels,
            channels,
            pre_lookahead_len + 1,
            conv1_cfg,
            vb.pp("conv1"),
        )?;
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

        eprintln!("PreLookahead conv1 input shape: {:?}", outputs.shape());
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
        let mut dit_cfg = dit_config.clone();
        dit_cfg.spk_dim = flow_config.output_size;
        dit_cfg.static_chunk_size = flow_config.chunk_size * flow_config.token_mel_ratio;
        dit_cfg.num_decoding_left_chunks = flow_config.num_decoding_left_chunks;
        let dit = DiT::new(vb.pp("decoder.estimator"), &dit_cfg)?;
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
    /// * `noise` - Optional pre-generated noise for parity testing (use `None` for random)
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
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        eprintln!("\n  [Flow.inference] Starting...");
        eprintln!("    token shape: {:?}", token.shape());
        eprintln!("    prompt_token shape: {:?}", prompt_token.shape());
        eprintln!("    prompt_feat shape: {:?}", prompt_feat.shape());
        eprintln!("    embedding shape: {:?}", embedding.shape());

        // Normalize speaker embedding
        let embedding_norm = l2_normalize(embedding)?;
        let embedding_proj = self.spk_embed_affine_layer.forward(&embedding_norm)?;
        eprintln!("    embedding_proj shape: {:?}", embedding_proj.shape());

        // Concatenate prompt and target tokens
        let combined_token = Tensor::cat(&[prompt_token, token], 1)?;
        eprintln!("    combined_token shape: {:?}", combined_token.shape());

        // Embed tokens
        let token_emb = self.input_embedding.forward(&combined_token)?;
        eprintln!("    token_emb shape: {:?}", token_emb.shape());

        // Apply pre-lookahead layer (finalize mode, no context)
        let h = self.pre_lookahead_layer.forward(&token_emb, None)?;
        eprintln!("    pre_lookahead output shape: {:?}", h.shape());

        // Repeat interleave for token_mel_ratio
        let h = repeat_interleave(&h, self.config.token_mel_ratio, 1)?;
        eprintln!("    after repeat_interleave shape: {:?}", h.shape());

        let prompt_mel_len = prompt_feat.dim(2)?; // [B, D, T], use dim 2
        let total_mel_len = h.dim(1)?;
        let target_mel_len = total_mel_len - prompt_mel_len;
        eprintln!(
            "    prompt_mel_len={}, total_mel_len={}, target_mel_len={}",
            prompt_mel_len, total_mel_len, target_mel_len
        );

        // Build conditioning tensor: [1, 80, total_len]
        let mut conds = Tensor::zeros(
            (1, self.config.output_size, prompt_mel_len + target_mel_len),
            prompt_feat.dtype(),
            &self.device,
        )?;

        // Copy prompt features into conds
        if prompt_mel_len > 0 {
            // Build conds by concatenating [1, 80, prompt_len] and [1, 80, target_len]
            let zeros_part = Tensor::zeros(
                (1, self.config.output_size, target_mel_len),
                prompt_feat.dtype(),
                &self.device,
            )?;
            conds = Tensor::cat(&[prompt_feat, &zeros_part], 2)?; // Cat along dim 2
        }
        eprintln!("    conds shape: {:?}", conds.shape());

        // Create mask (all ones for now)
        let mask = Tensor::ones((1, total_mel_len), h.dtype(), &self.device)?;

        // mu = h transposed
        let mu = h.transpose(1, 2)?; // [batch, hidden_dim, mel_len]
        eprintln!("    mu (to decoder) shape: {:?}", mu.shape());

        // Print mu statistics
        if let Ok(mu_flat) = mu.flatten_all() {
            if let Ok(mu_vec) = mu_flat.to_vec1::<f32>() {
                let min = mu_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mu_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = mu_vec.iter().sum();
                let mean = sum / mu_vec.len() as f32;
                eprintln!(
                    "    mu stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }

        // Run CFM decoder
        eprintln!("    Running CFM decoder with {} timesteps...", n_timesteps);
        let feat = self.decoder.forward(
            &mu,
            &mask,
            n_timesteps,
            1.0, // temperature
            Some(&embedding_proj),
            Some(&conds),
            noise,
        )?;
        eprintln!("    decoder output shape: {:?}", feat.shape());

        // Print decoder output statistics
        if let Ok(feat_flat) = feat.flatten_all() {
            if let Ok(feat_vec) = feat_flat.to_vec1::<f32>() {
                let min = feat_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = feat_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = feat_vec.iter().sum();
                let mean = sum / feat_vec.len() as f32;
                eprintln!(
                    "    decoder output stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }

        // Extract only the target portion (after prompt)
        // Clamp length to available dim (in case truncation happened during parity test)
        let actual_dim = feat.dim(2)?;
        let available_len = actual_dim.saturating_sub(prompt_mel_len);
        let final_len = usize::min(target_mel_len, available_len);
        if final_len != target_mel_len {
            eprintln!(
                "    [Flow.inference] Warning: Output truncated from {} to {}",
                target_mel_len, final_len
            );
        }
        let feat = feat.narrow(2, prompt_mel_len, final_len)?;
        eprintln!("    final feat shape: {:?}", feat.shape());

        // Print final mel statistics
        if let Ok(feat_flat) = feat.flatten_all() {
            if let Ok(feat_vec) = feat_flat.to_vec1::<f32>() {
                let min = feat_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = feat_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = feat_vec.iter().sum();
                let mean = sum / feat_vec.len() as f32;
                eprintln!(
                    "    [Flow.inference] Final mel stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }

        Ok(feat)
    }
}

/// L2 normalize along dimension 1
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norm = norm.broadcast_add(&Tensor::new(&[1e-8f32], x.device())?.to_dtype(x.dtype())?)?;
    x.broadcast_div(&norm)
}

/// Repeat interleave: repeat each element along dim
fn repeat_interleave(x: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape().dims();
    // let n = shape[dim];

    // Reshape, expand, and flatten back
    // For dim=1: [B, N, D] -> [B, N, 1, D] -> [B, N, R, D] -> [B, N*R, D]
    if dim == 1 {
        let (b, n_dim, d) = (shape[0], shape[1], shape[2]);
        let x = x.reshape((b, n_dim, 1, d))?;
        let x = x.repeat((1, 1, repeats, 1))?;
        x.reshape((b, n_dim * repeats, d))
    } else {
        // Generic fallback
        Err(candle_core::Error::Msg(format!(
            "repeat_interleave not implemented for dim={}",
            dim
        )))
    }
}
