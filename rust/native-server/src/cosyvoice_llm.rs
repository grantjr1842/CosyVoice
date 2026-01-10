//! CosyVoice LLM module for speech token generation.
//!
//! This module wraps the Qwen2 model and adds speech token generation capabilities.

use candle_core::{quantized::gguf_file, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use rand::Rng;
use std::cmp::Ordering;
use std::path::Path;
use tracing::{debug, info};

use crate::qwen::{Config as QwenConfig, ModelForCausalLM};
use crate::quantized_qwen::ModelForCausalLM as QuantizedModelForCausalLM;

/// Configuration for CosyVoice LLM
#[derive(Debug, Clone)]
pub struct CosyVoiceLLMConfig {
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,   // Base speech token vocab size
    pub speech_extra_tokens: usize, // Special/stop tokens appended to speech vocab
    pub sampling_vocab_size: usize, // Valid output range (flow vocab size)
    pub spk_embed_dim: usize,
    pub sampling_top_p: f32,
    pub ras_window_size: usize,
    pub ras_tau_r: f32,
    pub stop_token_count: usize,
}

impl Default for CosyVoiceLLMConfig {
    fn default() -> Self {
        Self {
            llm_input_size: 896,
            llm_output_size: 896,
            speech_token_size: 6561,  // Fun-CosyVoice3-0.5B speech token size
            speech_extra_tokens: 25,  // CosyVoice3 special tokens (speech + 3)
            sampling_vocab_size: 6561, // Fun-CosyVoice3-0.5B Flow vocab size
            spk_embed_dim: 192,
            sampling_top_p: 0.8,
            ras_window_size: 10,
            ras_tau_r: 0.1,
            stop_token_count: 200,
        }
    }
}

enum QwenModel {
    Standard(ModelForCausalLM),
    Quantized(QuantizedModelForCausalLM),
}

impl QwenModel {
    fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.base_model.forward_embeds(inputs_embeds, seqlen_offset, attn_mask),
            Self::Quantized(m) => m.base_model.forward_embeds(inputs_embeds, seqlen_offset, attn_mask),
        }
    }

    fn clear_kv_cache(&mut self) {
         match self {
            Self::Standard(m) => m.clear_kv_cache(),
            Self::Quantized(m) => m.clear_kv_cache(),
        }
    }

    fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
         match self {
            Self::Standard(m) => m.base_model.embed_tokens.forward(tokens),
            Self::Quantized(m) => m.base_model.embed_tokens.forward(tokens),
        }
    }
}


/// CosyVoice LLM for speech token generation
pub struct CosyVoiceLLM {
    /// Core Qwen2 model
    llm: QwenModel,
    /// LLM embedding for SOS and task_id tokens
    llm_embedding: Option<Embedding>,
    /// Speech token embedding
    speech_embedding: Embedding,
    /// Speaker embedding projection
    spk_embed_affine_layer: Linear,
    /// Decoder head for speech tokens
    llm_decoder: Linear,
    /// Decoder vocabulary size (including special tokens)
    _speech_vocab_size: usize,
    /// Valid sampling range limit
    _sampling_vocab_size: usize,
    /// Configuration
    config: CosyVoiceLLMConfig,
    /// Stop token IDs used to truncate generation
    stop_token_ids: Vec<usize>,
    /// Device
    device: Device,
    /// Whether special tokens come from speech embedding
    use_speech_special_tokens: bool,
    /// Special token IDs
    sos: usize,
    task_id: usize,
}

impl CosyVoiceLLM {
    /// Create a new CosyVoice LLM from safetensors weights
    pub fn new(
        qwen_config: &QwenConfig,
        llm_config: CosyVoiceLLMConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Load core Qwen2 model - it's under "llm.model" prefix
        info!("Loading Qwen2 model from llm.model prefix...");
        let llm = ModelForCausalLM::new(qwen_config, vb.pp("llm.model"))?;
        info!("Qwen2 model loaded successfully");

        Self::build(QwenModel::Standard(llm), llm_config, vb, device)
    }

    pub fn from_gguf<P: AsRef<Path>>(
        gguf_path: P,
        llm_config: CosyVoiceLLMConfig,
        vb_extras: VarBuilder, // For speech_embedding and llm_decoder
    ) -> Result<Self> {
        let device = vb_extras.device().clone();

        info!("Loading Quantized Qwen2 from {:?}...", gguf_path.as_ref());
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let llm = QuantizedModelForCausalLM::from_gguf(&content, &mut file, &device)?;
        info!("Quantized Qwen2 loaded successfully");

        Self::build(QwenModel::Quantized(llm), llm_config, vb_extras, device)
    }

    fn build(
        llm: QwenModel,
        llm_config: CosyVoiceLLMConfig,
        vb: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        // Load speech embedding using weights to infer vocab size.
        info!("Loading speech_embedding from top-level...");
        let target_vocab_size = llm_config.speech_token_size + llm_config.stop_token_count;
        let speech_emb_weight = vb.pp("speech_embedding").get((target_vocab_size, llm_config.llm_input_size), "weight")?.to_dtype(vb.dtype())?;
        let (speech_vocab_size, speech_emb_dim) = speech_emb_weight.dims2()?;
        if speech_emb_dim != llm_config.llm_input_size {
            return Err(candle_core::Error::msg(format!(
                "speech_embedding dim mismatch: expected {}, got {}",
                llm_config.llm_input_size, speech_emb_dim
            )));
        }
        if speech_vocab_size < llm_config.sampling_vocab_size {
            return Err(candle_core::Error::msg(format!(
                "speech_embedding vocab too small: {} < sampling_vocab_size {}",
                speech_vocab_size, llm_config.sampling_vocab_size
            )));
        }
        info!("speech_embedding loaded successfully");
        let speech_embedding = Embedding::new(speech_emb_weight, llm_config.llm_input_size);

        // Load decoder head (weight is transposed: [vocab_size, hidden_size])
        info!("Loading llm_decoder from top-level...");
        let decoder_weight = vb.pp("llm_decoder").get((speech_vocab_size, llm_config.llm_output_size), "weight")?.to_dtype(vb.dtype())?;
        let (decoder_vocab_size, decoder_hidden) = decoder_weight.dims2()?;
        if decoder_hidden != llm_config.llm_output_size {
            return Err(candle_core::Error::msg(format!(
                "llm_decoder dim mismatch: expected {}, got {}",
                llm_config.llm_output_size, decoder_hidden
            )));
        }
        if decoder_vocab_size != speech_vocab_size {
            return Err(candle_core::Error::msg(format!(
                "llm_decoder vocab mismatch: expected {}, got {}",
                speech_vocab_size, decoder_vocab_size
            )));
        }
        if decoder_vocab_size != speech_vocab_size {
            return Err(candle_core::Error::msg(format!(
                "llm_decoder vocab mismatch: expected {}, got {}",
                speech_vocab_size, decoder_vocab_size
            )));
        }
        let llm_decoder = Linear::new(decoder_weight, None);
        info!("llm_decoder loaded successfully");

        let _extra_tokens = speech_vocab_size - llm_config.sampling_vocab_size;
        let mut stop_token_ids = Vec::new();
        for idx in llm_config.sampling_vocab_size..speech_vocab_size {
            if stop_token_ids.len() >= llm_config.stop_token_count {
                break;
            }
            stop_token_ids.push(idx);
        }

        // Match Python Qwen2LM logic: Always use llm_embedding for SOS/TaskID if it exists.
        // For CosyVoice3, llm_embedding does not exist, and special tokens are in speech_embedding.

        let (llm_embedding, use_speech_special_tokens, sos, task_id) = if vb.pp("llm_embedding").contains_tensor("weight") {
            let llm_emb_weight = vb.pp("llm_embedding").get((2, llm_config.llm_input_size), "weight")?.to_dtype(vb.dtype())?;
            let (_llm_vocab_size, llm_emb_dim) = llm_emb_weight.dims2()?;
            if llm_emb_dim != llm_config.llm_input_size {
                return Err(candle_core::Error::msg(format!(
                    "llm_embedding dim mismatch: expected {}, got {}",
                    llm_config.llm_input_size, llm_emb_dim
                )));
            }
             (Some(Embedding::new(llm_emb_weight, llm_config.llm_input_size)), false, 0, 1)
        } else {
             // CosyVoice3 logic: indices 6561 (SOS) and 6563 (TaskID) in speech_embedding
             (None, true, llm_config.sampling_vocab_size, llm_config.sampling_vocab_size + 2)
        };

        // Load spk_embed_affine_layer (try both 'llm.spk...' and 'spk...')
        // The patched file has 'llm.spk_embed_affine_layer', but logically it should be top-level.
        let spk_prefix = if vb.pp("llm.spk_embed_affine_layer").contains_tensor("weight") {
            "llm.spk_embed_affine_layer"
        } else {
            "spk_embed_affine_layer"
        };
        info!("Loading spk_embed_affine_layer from {}...", spk_prefix);
        let spk_affine_weight = vb.pp(spk_prefix).get((llm_config.llm_input_size, llm_config.spk_embed_dim), "weight")?.to_dtype(vb.dtype())?;
        let spk_affine_bias = vb.pp(spk_prefix).get(llm_config.llm_input_size, "bias")?.to_dtype(vb.dtype())?;
        let spk_embed_affine_layer = Linear::new(spk_affine_weight, Some(spk_affine_bias));

        Ok(Self {
            llm,
            llm_embedding,
            speech_embedding,
            spk_embed_affine_layer,
            llm_decoder,
            _speech_vocab_size: speech_vocab_size,
            _sampling_vocab_size: llm_config.sampling_vocab_size,
            config: llm_config,
            device,
            use_speech_special_tokens,
            sos,
            task_id,
            stop_token_ids,
        })
    }

    /// Get SOS embedding
    fn get_sos_emb(&self) -> Result<Tensor> {
        let idx = Tensor::new(&[self.sos as u32], &self.device)?;
        if self.use_speech_special_tokens {
            let emb = self.speech_embedding.forward(&idx)?.unsqueeze(0)?;
            println!("LLM: Using speech special token sos={} (norm: {:?})", self.sos, emb.to_dtype(candle_core::DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?);
            Ok(emb)
        } else {
            let emb = self.llm_embedding
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("llm_embedding missing"))?
                .forward(&idx)?
                .unsqueeze(0)?;
            println!("LLM: Using llm_embedding sos={} (norm: {:?})", self.sos, emb.sqr()?.sum_all()?.to_scalar::<f32>()?);
            Ok(emb)
        }
    }

    /// Get task_id embedding
    fn get_task_id_emb(&self) -> Result<Tensor> {
        let idx = Tensor::new(&[self.task_id as u32], &self.device)?;
        if self.use_speech_special_tokens {
            let emb = self.speech_embedding.forward(&idx)?.unsqueeze(0)?;
            println!("LLM: Using speech special token task_id={} (norm: {:?})", self.task_id, emb.to_dtype(candle_core::DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?);
            Ok(emb)
        } else {
            let emb = self.llm_embedding
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("llm_embedding missing"))?
                .forward(&idx)?
                .unsqueeze(0)?;
            println!("LLM: Using llm_embedding task_id={} (norm: {:?})", self.task_id, emb.sqr()?.sum_all()?.to_scalar::<f32>()?);
            Ok(emb)
        }
    }

    /// Embed text tokens using the underlying LLM's embedding layer
    pub fn embed_text_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.llm.embed_tokens(tokens)
    }

    /// Embed speech tokens
    fn embed_speech_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        self.speech_embedding.forward(tokens)
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        let idx = token_id as usize;
        self.stop_token_ids.contains(&idx)
    }

    pub fn stop_token_ids(&self) -> &[usize] {
        &self.stop_token_ids
    }

    fn sample_from_weighted(&self, weights: &[(usize, f32)]) -> Option<usize> {
        let total: f32 = weights.iter().map(|(_, w)| *w).sum();
        if total <= 0.0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let mut r = rng.gen::<f32>() * total;
        for (idx, w) in weights {
            r -= *w;
            if r <= 0.0 {
                return Some(*idx);
            }
        }
        weights.last().map(|(idx, _)| *idx)
    }

    fn sample_from_probs(&self, probs: &[f32]) -> Option<usize> {
        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut rng = rand::thread_rng();
        let mut r = rng.gen::<f32>() * total;
        for (idx, p) in probs.iter().enumerate() {
            r -= *p;
            if r <= 0.0 {
                return Some(idx);
            }
        }
        if probs.is_empty() {
            None
        } else {
            Some(probs.len() - 1)
        }
    }

    fn sample_nucleus(&self, probs: &[f32], top_p: f32, top_k: usize) -> Option<usize> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut candidates: Vec<(usize, f32)> = Vec::new();
        let mut cum_prob = 0.0;
        let k = top_k.max(1);
        for (idx, prob) in indexed {
            if cum_prob < top_p && candidates.len() < k {
                cum_prob += prob;
                candidates.push((idx, prob));
            } else {
                break;
            }
        }

        if candidates.is_empty() {
            return None;
        }

        self.sample_from_weighted(&candidates)
    }

    fn sample_ras(
        &self,
        logp: &Tensor,
        decoded_tokens: &[u32],
        sampling_k: usize,
        ignore_stop: bool,
    ) -> Result<u32> {
        let logp = logp.to_dtype(candle_core::DType::F32)?.squeeze(0)?;
        let logp_vec: Vec<f32> = logp.to_vec1()?;
        let mut probs: Vec<f32> = logp_vec.iter().map(|v| v.exp()).collect();
        if ignore_stop {
            let stop_start = self.config.sampling_vocab_size.min(probs.len());
            for p in probs.iter_mut().skip(stop_start) {
                *p = 0.0;
            }
        }

        let top_id = self
            .sample_nucleus(&probs, self.config.sampling_top_p, sampling_k)
            .unwrap_or(0);

        if self.config.ras_window_size > 0 && !decoded_tokens.is_empty() {
            let recent = decoded_tokens
                .iter()
                .rev()
                .take(self.config.ras_window_size);
            let rep_num = recent.filter(|&&t| t as usize == top_id).count() as f32;
            let threshold = self.config.ras_window_size as f32 * self.config.ras_tau_r;
            if rep_num >= threshold {
                if let Some(random_id) = self.sample_from_probs(&probs) {
                    return Ok(random_id as u32);
                }
            }
        }

        Ok(top_id as u32)
    }

    fn sample_ids(
        &self,
        logp: &Tensor,
        decoded_tokens: &[u32],
        sampling_k: usize,
        ignore_stop: bool,
    ) -> Result<u32> {
        let max_trials = 100;
        let mut trials = 0;
        loop {
            let top_id = self.sample_ras(logp, decoded_tokens, sampling_k, ignore_stop)?;
            if !ignore_stop || !self.is_stop_token(top_id) {
                return Ok(top_id);
            }
            trials += 1;
            if trials > max_trials {
                return Err(candle_core::Error::msg(
                    "sampling reaches max_trials and still gets stop tokens",
                ));
            }
        }
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
        speaker_embedding: Option<&Tensor>,
        sampling_k: usize,
        min_len: usize,
        max_len: usize,
    ) -> Result<Vec<u32>> {
        // Clear KV cache for fresh generation
        self.llm.clear_kv_cache();

        // Build initial input: [sos, spk_emb, text, task_id, prompt_speech]
        let sos_emb = self.get_sos_emb()?;
        let task_id_emb = self.get_task_id_emb()?;

        // Ensure all parts have same dtype as llm_decoder
        let dtype = self.llm_decoder.weight().dtype();
        let sos_emb = sos_emb.to_dtype(dtype)?;
        let text_embeds = text_embeds.to_dtype(dtype)?;
        let task_id_emb = task_id_emb.to_dtype(dtype)?;

        // Speaker embedding projection
        // NOTE: CosyVoice3 (Qwen2LM) does NOT use speaker embedding in the LLM.
        // We keep the argument for compatibility but do not use it in `parts`.
        let _spk_emb = if let Some(emb) = speaker_embedding {
            // F.normalize(embedding, dim=1) + affine + unsqueeze
            let emb = crate::cosyvoice_flow::l2_normalize(emb)?.to_dtype(dtype)?;
            let proj = self.spk_embed_affine_layer.forward(&emb)?;
            proj.unsqueeze(1)?
        } else {
             Tensor::zeros((1, 0, self.config.llm_input_size), dtype, &self.device)?
        };

        let mut parts = vec![sos_emb];
        // parts.push(spk_emb); // DISABLED for CosyVoice3 parity
        parts.push(text_embeds.clone());
        parts.push(task_id_emb);

        if let Some(prompt_tokens) = prompt_speech_tokens {
            if prompt_tokens.dim(1)? > 0 {
                let prompt_emb = self.embed_speech_tokens(prompt_tokens)?.to_dtype(dtype)?;
                parts.push(prompt_emb);
            }
        }

        // Concatenate all parts into initial input
        let mut lm_input = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        let mut out_tokens: Vec<u32> = Vec::new();
        let mut seqlen_offset = 0;

        for i in 0..max_len {
            // Generate causal mask for current input
            let seq_len = lm_input.dim(1)?;
            let mask = if seq_len > 1 {
                let mask: Vec<_> = (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
                    .collect();
                Some(Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), &self.device)?.to_dtype(lm_input.dtype())?)
            } else {
                None
            };

            // Forward pass through LLM
            let y_pred = self
                .llm
                .forward_embeds(&lm_input, seqlen_offset, mask.as_ref())?;

            // Get logits from last position
            let logits = self.llm_decoder.forward(&y_pred.i((
                ..,
                y_pred.dim(1)? - 1..y_pred.dim(1)?,
                ..,
            ))?)?;
            let logp = candle_nn::ops::log_softmax(&logits.squeeze(1)?.to_dtype(candle_core::DType::F32)?, 1)?;

            // Sample next token
            let ignore_stop = i < min_len;
            let top_id = self.sample_ids(&logp, &out_tokens, sampling_k, ignore_stop)?;

            if self.is_stop_token(top_id) {
                debug!(
                    "LLM: stop token reached at step {} (token id {})",
                    i, top_id
                );
                break;
            }

            out_tokens.push(top_id);
            if i < 20 {
                debug!("Gen token {}: {}", i, top_id);
            }

            // Prepare next input (just the new token embedding)
            seqlen_offset += lm_input.dim(1)?;
            let token_tensor = Tensor::new(&[top_id], &self.device)?;
            lm_input = self.embed_speech_tokens(&token_tensor.unsqueeze(0)?)?;
        }

        Ok(out_tokens)
    }
    pub fn debug_forward_one(
        &mut self,
        text_embeds: &Tensor,
        prompt_speech_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.llm.clear_kv_cache();
        let sos_emb = self.get_sos_emb()?;
        let task_id_emb = self.get_task_id_emb()?;
        let dtype = self.llm_decoder.weight().dtype();
        let sos_emb = sos_emb.to_dtype(dtype)?;
        let text_embeds = text_embeds.to_dtype(dtype)?;
        let task_id_emb = task_id_emb.to_dtype(dtype)?;

        let mut parts = vec![sos_emb, text_embeds.clone(), task_id_emb];
        if let Some(prompt_tokens) = prompt_speech_tokens {
            if prompt_tokens.dim(1)? > 0 {
                let prompt_emb = self.embed_speech_tokens(prompt_tokens)?.to_dtype(dtype)?;
                parts.push(prompt_emb);
            }
        }
        let lm_input = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;

        // Generate causal mask
        let seq_len = lm_input.dim(1)?;
        let mask = if seq_len > 1 {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
                .collect();
            Some(Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), &self.device)?.to_dtype(dtype)?)
        } else {
            None
        };

        // Forward
        let y_pred = self.llm.forward_embeds(&lm_input, 0, mask.as_ref())?;
        let logits = self.llm_decoder.forward(&y_pred.i((.., y_pred.dim(1)? - 1..y_pred.dim(1)?, ..))?)?;
        Ok(logits)
    }
}
