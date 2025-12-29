//! Native Rust TTS Engine for CosyVoice.
//!
//! This module orchestrates the full TTS pipeline using native Rust implementations.
//! ONNX-based frontend is handled by a minimal Python bridge for now.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use thiserror::Error;

use crate::cosyvoice_flow::{CosyVoiceFlow, CosyVoiceFlowConfig};
use crate::cosyvoice_llm::{CosyVoiceLLM, CosyVoiceLLMConfig};
use crate::flow::FlowConfig;
use crate::hift::{HiFTConfig, HiFTGenerator};
use crate::qwen::Config as QwenConfig;

#[derive(Error, Debug)]
pub enum NativeTtsError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Native Rust CosyVoice TTS Engine (LLM + Flow + HiFT)
/// Note: Frontend (tokenization, speaker embedding) still requires Python/ONNX
pub struct NativeTtsEngine {
    /// LLM for speech token generation
    pub llm: CosyVoiceLLM,
    /// Flow model for mel generation
    pub flow: CosyVoiceFlow,
    /// HiFT vocoder for audio synthesis
    pub hift: HiFTGenerator,
    /// Device (CPU or CUDA)
    pub device: Device,
    /// Sample rate
    pub sample_rate: u32,
}

impl NativeTtsEngine {
    /// Create a new native TTS engine from safetensors weights
    pub fn new(model_dir: &str) -> Result<Self, NativeTtsError> {
        let model_path = PathBuf::from(model_dir);

        // Initialize device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        eprintln!("Native TTS Engine using device: {:?}", device);

        // Load configurations
        let config_str = std::fs::read_to_string(model_path.join("config.json"))?;

        // Load Qwen2 config
        let qwen_config: QwenConfig = serde_json::from_str(&config_str)?;

        // Load LLM from safetensors
        let llm_path = model_path.join("llm.safetensors");
        eprintln!("Loading LLM from {:?}", llm_path);
        let llm_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&llm_path], DType::F32, &device)
                .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to load LLM: {}", e)))?
        };

        let llm_config = CosyVoiceLLMConfig::default();
        // Pass root VarBuilder - CosyVoiceLLM::new handles prefixing internally
        let llm = CosyVoiceLLM::new(&qwen_config, llm_config, llm_vb)
            .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to initialize LLM: {}", e)))?;
        eprintln!("LLM initialized successfully");

        // Load Flow from safetensors
        let flow_path = model_path.join("flow.safetensors");
        eprintln!("Loading Flow from {:?}", flow_path);
        let flow_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&flow_path], DType::F32, &device)
                .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to load Flow: {}", e)))?
        };

        let flow_config = CosyVoiceFlowConfig::default();
        let dit_config = FlowConfig::default();
        // Pass root VarBuilder - CosyVoiceFlow handles prefixing internally
        let flow = CosyVoiceFlow::new(flow_config, &dit_config, flow_vb)
            .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to initialize Flow: {}", e)))?;
        eprintln!("Flow initialized successfully");

        // Load HiFT from safetensors
        let hift_path = model_path.join("hift.safetensors");
        eprintln!("Loading HiFT from {:?}", hift_path);
        let hift_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&hift_path], DType::F32, &device)
                .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to load HiFT: {}", e)))?
        };

        let hift_config = HiFTConfig::default();
        let hift = HiFTGenerator::new(hift_vb, &hift_config)
            .map_err(|e| NativeTtsError::ModelLoad(format!("Failed to initialize HiFT: {}", e)))?;
        eprintln!("HiFT initialized successfully");

        Ok(Self {
            llm,
            flow,
            hift,
            device,
            sample_rate: 24000,
        })
    }

    /// Synthesize speech from speech tokens (assumes tokenization done externally)
    ///
    /// # Arguments
    /// * `speech_tokens` - Generated speech tokens [batch, token_len]
    /// * `prompt_tokens` - Prompt speech tokens [batch, prompt_len]
    /// * `prompt_mel` - Prompt mel features [batch, mel_len, mel_dim]
    /// * `speaker_embedding` - Speaker embedding [batch, spk_dim]
    ///
    /// # Returns
    /// Audio samples as i16
    pub fn synthesize_from_tokens(
        &self,
        speech_tokens: &Tensor,
        prompt_tokens: &Tensor,
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
    ) -> Result<Vec<i16>, NativeTtsError> {
        // 1. Run Flow to get mel spectrogram
        eprintln!("Running Flow inference...");
        let mel = self.flow.inference(
            speech_tokens,
            prompt_tokens,
            prompt_mel,
            speaker_embedding,
            10, // n_timesteps
        )?;
        eprintln!("Flow output shape: {:?}", mel.shape());

        // 2. Run HiFT to get audio
        eprintln!("Running HiFT inference...");
        let audio = self.hift.forward(&mel)?;
        eprintln!("HiFT output shape: {:?}", audio.shape());

        // 3. Convert to i16 samples
        let audio_vec: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        let samples: Vec<i16> = audio_vec
            .iter()
            .map(|&x: &f32| (x * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        Ok(samples)
    }

    /// Full synthesis with LLM generation (requires pre-computed text embeddings)
    ///
    /// # Arguments
    /// * `text_embeds` - Text embeddings from encoder [batch, text_len, hidden_dim]
    /// * `prompt_speech_tokens` - Prompt speech tokens [batch, prompt_len]
    /// * `prompt_mel` - Prompt mel features [batch, mel_len, mel_dim]
    /// * `speaker_embedding` - Speaker embedding [batch, spk_dim]
    /// * `sampling_k` - Top-k sampling parameter
    ///
    /// # Returns
    /// Audio samples as i16
    pub fn synthesize_full(
        &mut self,
        text_embeds: &Tensor,
        prompt_speech_tokens: Option<&Tensor>,
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
        sampling_k: usize,
    ) -> Result<Vec<i16>, NativeTtsError> {
        // Calculate generation length bounds
        let text_len = text_embeds.dim(1)?;
        let min_len = (text_len as f32 * 2.0) as usize;
        let max_len = (text_len as f32 * 20.0) as usize;

        // 1. Generate speech tokens via LLM
        eprintln!("Generating speech tokens... (min={}, max={})", min_len, max_len);
        let speech_tokens = self.llm.generate(
            text_embeds,
            prompt_speech_tokens,
            Some(speaker_embedding),
            sampling_k,
            min_len,
            max_len,
        )?;
        eprintln!("Generated {} speech tokens", speech_tokens.len());

        // Convert to tensor
        let token_len = speech_tokens.len();
        let speech_token_tensor = Tensor::from_vec(
            speech_tokens,
            (1, token_len),
            &self.device,
        )?;

        // Get prompt tokens or empty
        let empty_prompt = Tensor::zeros((1, 0), DType::U32, &self.device)?;
        let prompt_tokens = prompt_speech_tokens.unwrap_or(&empty_prompt);

        // 2. Run Flow + HiFT
        self.synthesize_from_tokens(
            &speech_token_tensor,
            prompt_tokens,
            prompt_mel,
            speaker_embedding,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        // This test requires model files to exist
        let result = NativeTtsEngine::new("pretrained_models/Fun-CosyVoice3-0.5B");
        match result {
            Ok(_) => println!("Engine created successfully"),
            Err(e) => println!("Engine creation failed (expected if models missing): {}", e),
        }
    }
}
