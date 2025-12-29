//! CosyVoice LLM module for speech token generation.
//!
//! This module wraps the Qwen2 model and adds speech token generation capabilities.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use rand::Rng;

use crate::qwen::{Config as QwenConfig, ModelForCausalLM};

/// Configuration for CosyVoice LLM
#[derive(Debug, Clone)]
pub struct CosyVoiceLLMConfig {
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,
    pub spk_embed_dim: usize,
}

impl Default for CosyVoiceLLMConfig {
    fn default() -> Self {
        Self {
            llm_input_size: 896,
            llm_output_size: 896,
            speech_token_size: 4096,
            spk_embed_dim: 192,
        }
    }
}

/// CosyVoice LLM for speech token generation
pub struct CosyVoiceLLM {
    /// Core Qwen2 model
    llm: ModelForCausalLM,
    /// LLM embedding for SOS and task_id tokens
    llm_embedding: Embedding,
    /// Speech token embedding
    speech_embedding: Embedding,
    /// Decoder head for speech tokens
    llm_decoder: Linear,
    /// Speaker embedding projection
    spk_embed_affine_layer: Linear,
    /// Configuration
    config: CosyVoiceLLMConfig,
    /// Device
    device: Device,
    /// Special token IDs
    sos: usize,
    task_id: usize,
    eos_token: usize,
}

impl CosyVoiceLLM {
    /// Create a new CosyVoice LLM from safetensors weights
    pub fn new(
        qwen_config: &QwenConfig,
        llm_config: CosyVoiceLLMConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Load core Qwen2 model
        let llm = ModelForCausalLM::new(qwen_config, vb.clone())?;

        // Load LLM embedding (SOS=0, task_id=1)
        let llm_embedding = embedding(2, llm_config.llm_input_size, vb.pp("llm_embedding"))?;

        // Load speech embedding (speech_token_size + 3 tokens including EOS, fill, etc.)
        let speech_embedding = embedding(
            llm_config.speech_token_size + 3,
            llm_config.llm_input_size,
            vb.pp("speech_embedding"),
        )?;

        // Load decoder head
        let llm_decoder = linear(
            llm_config.llm_output_size,
            llm_config.speech_token_size + 3,
            vb.pp("llm_decoder"),
        )?;

        // Load speaker embedding projection - try "spk_embed_affine_layer" first
        let spk_embed_affine_layer = if vb.contains_tensor("spk_embed_affine_layer.weight") {
            linear(llm_config.spk_embed_dim, llm_config.llm_input_size, vb.pp("spk_embed_affine_layer"))?
        } else {
            // Fallback: create identity-like initialization
            let weight = Tensor::randn(
                0.0f32,
                0.02,
                (llm_config.llm_input_size, llm_config.spk_embed_dim),
                &device,
            )?;
            let bias = Tensor::zeros((llm_config.llm_input_size,), DType::F32, &device)?;
            Linear::new(weight, Some(bias))
        };

        Ok(Self {
            llm,
            llm_embedding,
            speech_embedding,
            llm_decoder,
            spk_embed_affine_layer,
            config: llm_config,
            device,
            sos: 0,
            task_id: 1,
            eos_token: 4096, // speech_token_size
        })
    }

    /// Get SOS embedding
    fn get_sos_emb(&self) -> Result<Tensor> {
        let idx = Tensor::new(&[self.sos as u32], &self.device)?;
        self.llm_embedding.forward(&idx)?.unsqueeze(0)
    }

    /// Get task_id embedding
    fn get_task_id_emb(&self) -> Result<Tensor> {
        let idx = Tensor::new(&[self.task_id as u32], &self.device)?;
        self.llm_embedding.forward(&idx)?.unsqueeze(0)
    }

    /// Embed speech tokens
    fn embed_speech_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.speech_embedding.forward(tokens)
    }

    /// Top-k sampling
    fn sample_top_k(&self, logits: &Tensor, k: usize, ignore_eos: bool) -> Result<u32> {
        let logits = logits.squeeze(0)?; // [vocab_size]
        let vocab_size = logits.dim(0)?;

        // Get logits as vec
        let logits_vec: Vec<f32> = logits.to_vec1()?;

        // Find top-k indices and values
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Filter out EOS if ignoring
        let candidates: Vec<(usize, f32)> = if ignore_eos {
            indexed.into_iter()
                .filter(|(idx, _)| *idx < self.config.speech_token_size)
                .take(k)
                .collect()
        } else {
            indexed.into_iter().take(k).collect()
        };

        if candidates.is_empty() {
            return Ok(self.eos_token as u32);
        }

        // Softmax over top-k
        let max_logit = candidates.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = candidates.iter().map(|(_, v)| (v - max_logit).exp()).sum();
        let probs: Vec<f32> = candidates.iter().map(|(_, v)| (v - max_logit).exp() / exp_sum).collect();

        // Sample from categorical distribution
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return Ok(candidates[i].0 as u32);
            }
        }

        // Fallback to first candidate
        Ok(candidates[0].0 as u32)
    }

    /// Generate speech tokens autoregressively
    ///
    /// # Arguments
    /// * `text_embeds` - Text embeddings [1, text_len, hidden_size]
    /// * `prompt_speech_tokens` - Prompt speech tokens [1, prompt_len] (optional)
    /// * `speaker_embedding` - Speaker embedding [1, spk_embed_dim] (optional)
    /// * `sampling_k` - Top-k sampling parameter
    /// * `min_len` - Minimum output length
    /// * `max_len` - Maximum output length
    ///
    /// # Returns
    /// Vector of generated speech token IDs
    pub fn generate(
        &mut self,
        text_embeds: &Tensor,
        prompt_speech_tokens: Option<&Tensor>,
        speaker_embedding: Option<&Tensor>,
        sampling_k: usize,
        min_len: usize,
        max_len: usize,
    ) -> Result<Vec<u32>> {
        // Clear KV cache for fresh generation
        self.llm.clear_kv_cache();

        // Build initial input: [sos, text, task_id, prompt_speech]
        let sos_emb = self.get_sos_emb()?;
        let task_id_emb = self.get_task_id_emb()?;

        let mut parts = vec![sos_emb];
        parts.push(text_embeds.clone());
        parts.push(task_id_emb);

        if let Some(prompt_tokens) = prompt_speech_tokens {
            if prompt_tokens.dim(1)? > 0 {
                let prompt_emb = self.embed_speech_tokens(prompt_tokens)?;
                parts.push(prompt_emb);
            }
        }

        // Concatenate all parts into initial input
        let mut lm_input = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        let mut out_tokens: Vec<u32> = Vec::new();
        let mut seqlen_offset = 0;

        for i in 0..max_len {
            // Forward pass through LLM
            let y_pred = self.llm.forward_embeds(&lm_input, seqlen_offset)?;

            // Get logits from last position
            let logits = self.llm_decoder.forward(&y_pred.i((.., y_pred.dim(1)? - 1..y_pred.dim(1)?, ..))?)?;
            let logp = candle_nn::ops::log_softmax(&logits.squeeze(1)?, 1)?;

            // Sample next token
            let ignore_eos = i < min_len;
            let top_id = self.sample_top_k(&logp.i(0)?, sampling_k, ignore_eos)?;

            // Check for EOS
            if top_id as usize >= self.config.speech_token_size {
                break;
            }

            out_tokens.push(top_id);

            // Prepare next input (just the new token embedding)
            seqlen_offset += lm_input.dim(1)?;
            let token_tensor = Tensor::new(&[top_id], &self.device)?;
            lm_input = self.embed_speech_tokens(&token_tensor.unsqueeze(0)?)?;
        }

        Ok(out_tokens)
    }
}
