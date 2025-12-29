//! CosyVoice LLM module for speech token generation.
//!
//! This module wraps the Qwen2 model and adds speech token generation capabilities.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use rand::Rng;

use crate::qwen::{Config as QwenConfig, ModelForCausalLM};

/// Configuration for CosyVoice LLM
#[derive(Debug, Clone)]
pub struct CosyVoiceLLMConfig {
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,  // Will be overridden by actual weight size
    pub spk_embed_dim: usize,
}

impl Default for CosyVoiceLLMConfig {
    fn default() -> Self {
        Self {
            llm_input_size: 896,
            llm_output_size: 896,
            speech_token_size: 6758,  // Fun-CosyVoice3-0.5B default
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
    /// Actual speech token vocabulary size (inferred from weights)
    speech_token_size: usize,
    /// Configuration
    config: CosyVoiceLLMConfig,
    /// Device
    device: Device,
    /// Special token IDs
    sos: usize,
    task_id: usize,
}

impl CosyVoiceLLM {
    /// Create a new CosyVoice LLM from safetensors weights
    ///
    /// # Arguments
    /// * `qwen_config` - Qwen2 model configuration
    /// * `llm_config` - CosyVoice LLM configuration
    /// * `vb` - VarBuilder at the ROOT level (no prefix) for llm.safetensors
    pub fn new(
        qwen_config: &QwenConfig,
        llm_config: CosyVoiceLLMConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Load core Qwen2 model - it's under "llm.model" prefix
        eprintln!("Loading Qwen2 model from llm.model prefix...");
        let llm = ModelForCausalLM::new(qwen_config, vb.pp("llm.model"))?;
        eprintln!("Qwen2 model loaded successfully");

        // Load LLM embedding (SOS=0, task_id=1)
        // This is NOT in the safetensors file, so we create it randomly
        eprintln!("Creating llm_embedding (2 tokens)...");
        let llm_embedding = create_random_embedding(2, llm_config.llm_input_size, &device)?;

        // Load speech embedding using the config's speech_token_size
        // The actual size in Fun-CosyVoice3-0.5B is 6761 (6758 + 3 special tokens)
        eprintln!("Loading speech_embedding from top-level...");
        let speech_vocab_size = llm_config.speech_token_size + 3;
        let speech_emb_weight = vb.pp("speech_embedding").get(
            (speech_vocab_size, llm_config.llm_input_size),
            "weight",
        )?;
        eprintln!("speech_embedding loaded successfully");
        let speech_embedding = Embedding::new(speech_emb_weight, llm_config.llm_input_size);

        // Load decoder head (weight is transposed: [vocab_size, hidden_size])
        eprintln!("Loading llm_decoder from top-level...");
        let decoder_weight = vb.pp("llm_decoder").get(
            (speech_vocab_size, llm_config.llm_output_size),
            "weight",
        )?;
        let llm_decoder = Linear::new(decoder_weight, None);
        eprintln!("llm_decoder loaded successfully");

        Ok(Self {
            llm,
            llm_embedding,
            speech_embedding,
            llm_decoder,
            speech_token_size: llm_config.speech_token_size,
            config: llm_config,
            device,
            sos: 0,
            task_id: 1,
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
        let _vocab_size = logits.dim(0)?;

        // Get logits as vec
        let logits_vec: Vec<f32> = logits.to_vec1()?;

        // Find top-k indices and values
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Filter out EOS if ignoring
        let candidates: Vec<(usize, f32)> = if ignore_eos {
            indexed.into_iter()
                .filter(|(idx, _)| *idx < self.speech_token_size)
                .take(k)
                .collect()
        } else {
            indexed.into_iter().take(k).collect()
        };

        if candidates.is_empty() {
            return Ok(self.speech_token_size as u32); // EOS token
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
    /// * `_speaker_embedding` - Speaker embedding [1, spk_embed_dim] (optional, unused for now)
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
        _speaker_embedding: Option<&Tensor>,
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
            if top_id as usize >= self.speech_token_size {
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

/// Create a random embedding layer (for weights not in safetensors)
fn create_random_embedding(vocab_size: usize, hidden_size: usize, device: &Device) -> Result<Embedding> {
    let weight = Tensor::randn(0.0f32, 0.02, (vocab_size, hidden_size), device)?;
    Ok(Embedding::new(weight, hidden_size))
}
