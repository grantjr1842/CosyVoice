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
use crate::onnx_frontend::OnnxFrontend;
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
    /// ONNX Frontend for tokenization and embedding
    pub frontend: Option<OnnxFrontend>,
    /// Device (CPU or CUDA)
    pub device: Device,
    /// Sample rate
    pub sample_rate: u32,
}

const MIN_TOKEN_TEXT_RATIO: f32 = 2.0;
const MAX_TOKEN_TEXT_RATIO: f32 = 20.0;

impl NativeTtsEngine {
    /// Create a new native TTS engine from safetensors weights
    pub fn new(model_dir: &str, device: Option<Device>) -> Result<Self, NativeTtsError> {
        let model_path = PathBuf::from(model_dir);

        // Initialize device
        let device = device.unwrap_or_else(|| Device::cuda_if_available(0).unwrap_or(Device::Cpu));
        eprintln!("Native TTS Engine using device: {:?}", device);

        // Load configurations
        let config_str = std::fs::read_to_string(model_path.join("config.json"))?;

        // Load Qwen2 config
        let qwen_config: QwenConfig = serde_json::from_str(&config_str)?;

        // Load LLM from safetensors (prefer RL weights when available)
        let use_rl = std::env::var("COSYVOICE_USE_RL")
            .map(|v| v != "0")
            .unwrap_or(true);
        let llm_path = if use_rl {
            let rl_path = model_path.join("llm.rl.safetensors");
            if rl_path.exists() {
                rl_path
            } else {
                let rl_pt = model_path.join("llm.rl.pt");
                if rl_pt.exists() {
                    eprintln!(
                        "RL checkpoint found at {:?} but no safetensors. Run: pixi run python tools/convert_llm_pt_to_safetensors.py {:?}",
                        rl_pt, rl_pt
                    );
                }
                model_path.join("llm.safetensors")
            }
        } else {
            model_path.join("llm.safetensors")
        };
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

        // Initialize ONNX Frontend
        eprintln!("Initializing ONNX Frontend...");
        // Use device for frontend if CUDA, otherwise CPU
        // Note: ORT CUDA provider registration inside OnnxFrontend handles the actual GPU usage.
        // We pass the device mainly for tensor placement if needed.
        let frontend = match OnnxFrontend::new(model_dir, device.clone()) {
            Ok(fe) => {
                eprintln!("ONNX Frontend initialized successfully");
                Some(fe)
            },
            Err(e) => {
                eprintln!("WARNING: Failed to initialize ONNX frontend: {}. Continuing without frontend.", e);
                None
            }
        };

        Ok(Self {
            llm,
            flow,
            hift,
            frontend,
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
    /// * `flow_noise` - Optional pre-generated noise for parity testing (use `None` for random)
    ///
    /// # Returns
    /// Audio samples as i16
    pub fn synthesize_from_tokens(
        &self,
        speech_tokens: &Tensor,
        prompt_tokens: &Tensor,
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
        flow_noise: Option<&Tensor>,
    ) -> Result<Vec<i16>, NativeTtsError> {
        // Debug input tensors
        eprintln!("\n=== SYNTHESIS DEBUG ===");
        eprintln!("Input speech_tokens shape: {:?}", speech_tokens.shape());
        eprintln!("Input prompt_tokens shape: {:?}", prompt_tokens.shape());
        eprintln!("Input prompt_mel shape: {:?}", prompt_mel.shape());
        eprintln!("Input speaker_embedding shape: {:?}", speaker_embedding.shape());

        // Print tensor statistics helper
        fn print_tensor_stats(name: &str, t: &Tensor) {
            if let Ok(flat) = t.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = vec.iter().sum();
                    let mean = sum / vec.len() as f32;
                    let variance: f32 = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;
                    let std = variance.sqrt();
                    eprintln!("  {} stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}, len={}",
                             name, min, max, mean, std, vec.len());
                    if vec.len() >= 10 {
                        eprintln!("    first_5: {:?}", &vec[..5]);
                        eprintln!("    last_5:  {:?}", &vec[vec.len()-5..]);
                    }
                }
            }
        }

        print_tensor_stats("prompt_mel", prompt_mel);
        print_tensor_stats("speaker_embedding", speaker_embedding);

        // 1. Run Flow to get mel spectrogram
        eprintln!("\n--- Running Flow inference... ---");
        let mel = self.flow.inference(
            speech_tokens,
            prompt_tokens,
            prompt_mel,
            speaker_embedding,
            10, // n_timesteps
            flow_noise,
        )?;
        eprintln!("Flow output shape: {:?}", mel.shape());
        print_tensor_stats("flow_output_mel", &mel);

        // 2. Run HiFT to get audio
        eprintln!("\n--- Running HiFT inference... ---");
        let audio = self.hift.forward(&mel)?;
        eprintln!("HiFT output shape: {:?}", audio.shape());
        print_tensor_stats("hift_output_audio", &audio);

        // 3. Convert to i16 samples
        let audio_vec: Vec<f32> = audio.flatten_all()?.to_vec1()?;

        // Check for issues BEFORE conversion
        let min_val = audio_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = audio_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_val: f32 = audio_vec.iter().sum::<f32>() / audio_vec.len() as f32;

        if min_val < -1.0 || max_val > 1.0 {
            eprintln!("⚠️  WARNING: Audio values out of [-1, 1] range! min={}, max={}", min_val, max_val);
        }
        if mean_val.abs() > 0.1 {
            eprintln!("⚠️  WARNING: Large DC offset detected! mean={}. Applying DC removal.", mean_val);
        }

        // DC Removal & Conversion
        let samples: Vec<i16> = audio_vec
            .iter()
            .map(|&x: &f32| ((x - mean_val) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        eprintln!("=== END SYNTHESIS DEBUG ===\n");

        Ok(samples)
    }

    /// Synthesize audio directly from a mel spectrogram (bypassing LLM and Flow)
    pub fn synthesize_from_mel(
        &self,
        mel: &Tensor,
    ) -> Result<Vec<i16>, NativeTtsError> {
        eprintln!("\n=== SYNTHESIS FROM MEL DEBUG ===");
        eprintln!("Input mel shape: {:?}", mel.shape());

        // Print tensor statistics helper (duplicated logic for now)
        fn print_tensor_stats(name: &str, t: &Tensor) {
            if let Ok(flat) = t.flatten_all() {
                if let Ok(vec) = flat.to_vec1::<f32>() {
                    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = vec.iter().sum();
                    let mean = sum / vec.len() as f32;
                    let variance: f32 = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;
                    let std = variance.sqrt();
                    eprintln!("  {} stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}, len={}",
                             name, min, max, mean, std, vec.len());
                }
            }
        }

        print_tensor_stats("input_mel", mel);

        // Run HiFT
        eprintln!("\n--- Running HiFT inference (Direct Mel)... ---");
        let audio = self.hift.forward(mel)?;
        eprintln!("HiFT output shape: {:?}", audio.shape());
        print_tensor_stats("hift_output_audio", &audio);

        let audio_vec: Vec<f32> = audio.flatten_all()?.to_vec1()?;

        // Check for issues
        let min_val = audio_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = audio_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_val: f32 = audio_vec.iter().sum::<f32>() / audio_vec.len() as f32;

        if min_val < -1.0 || max_val > 1.0 {
            eprintln!("⚠️  WARNING: Audio values out of [-1, 1] range! min={}, max={}", min_val, max_val);
        }
        if mean_val.abs() > 0.1 {
            eprintln!("⚠️  WARNING: Large DC offset detected! mean={}. Applying DC removal.", mean_val);
        }

        let samples: Vec<i16> = audio_vec
            .iter()
            .map(|&x: &f32| ((x - mean_val) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        eprintln!("=== END SYNTHESIS FROM MEL DEBUG ===\n");
        Ok(samples)
    }

    /// Full synthesis with LLM generation (requires pre-computed text embeddings)
    ///
    /// # Arguments
    /// * `text_embeds` - Text embeddings from encoder [batch, text_len, hidden_dim]
    /// * `prompt_text_len` - Prompt text token length included in `text_embeds`
    /// * `prompt_speech_tokens` - Prompt speech tokens [batch, prompt_len]
    /// * `prompt_mel` - Prompt mel features [batch, mel_len, mel_dim]
    /// * `speaker_embedding` - Speaker embedding [batch, spk_dim]
    /// * `sampling_k` - Top-k sampling parameter
    ///
    /// # Returns
    /// Audio samples as i16
    pub fn synthesize_full_with_prompt_len(
        &mut self,
        text_embeds: &Tensor,
        prompt_text_len: usize,
        prompt_speech_tokens: Option<&Tensor>,
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
        sampling_k: usize,
    ) -> Result<Vec<i16>, NativeTtsError> {
        // Calculate generation length bounds
        let text_len = text_embeds.dim(1)?;
        let effective_len = text_len.saturating_sub(prompt_text_len);
        let min_len = (effective_len as f32 * MIN_TOKEN_TEXT_RATIO) as usize;
        let max_len = (effective_len as f32 * MAX_TOKEN_TEXT_RATIO) as usize;

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
            None, // Use random noise in production
        )
    }

    /// Full synthesis with LLM generation (assumes no prompt text tokens are present)
    pub fn synthesize_full(
        &mut self,
        text_embeds: &Tensor,
        prompt_speech_tokens: Option<&Tensor>,
        prompt_mel: &Tensor,
        speaker_embedding: &Tensor,
        sampling_k: usize,
    ) -> Result<Vec<i16>, NativeTtsError> {
        self.synthesize_full_with_prompt_len(
            text_embeds,
            0,
            prompt_speech_tokens,
            prompt_mel,
            speaker_embedding,
            sampling_k,
        )
    }

    /// Process prompt audio tensors to get speech tokens and speaker embedding
    pub fn process_prompt_tensors(
        &mut self,
        prompt_speech_16k: &Tensor,   // [1, 128, T] for Tokenizer
        prompt_fbank: &Tensor, // [1, T, 80] for Speaker Embedding
    ) -> Result<(Tensor, Tensor), NativeTtsError> {
        // 1. Check frontend availability
        let frontend = self.frontend.as_mut().ok_or_else(|| {
            NativeTtsError::InferenceError("Frontend not initialized (ONNX error previously logged)".to_string())
        })?;

        // 2. Speech tokens
        let mel_len = prompt_speech_16k.dim(2)? as i32;
        let speech_tokens = frontend.tokenize_speech(prompt_speech_16k, mel_len)
            .map_err(|e| NativeTtsError::InferenceError(format!("Frontend error (tokenize): {}", e)))?;

        // 3. Speaker embedding
        let spk_emb = frontend.extract_speaker_embedding(prompt_fbank)
            .map_err(|e| NativeTtsError::InferenceError(format!("Frontend error (embedding): {}", e)))?;

        Ok((speech_tokens, spk_emb))
    }

    /// Synthesize speech in instruct/zero-shot mode
    ///
    /// # Arguments
    /// * `text_tokens` - Input text tokens [1, text_len] (including prompt text if any)
    /// * `prompt_text_len` - Prompt text token length included in `text_tokens`
    /// * `prompt_speech_16k` - Prompt mel features for tokenizer [1, 128, mel_len]
    /// * `prompt_speech_24k` - Prompt mel features for Flow [1, 80, mel_len]
    /// * `prompt_fbank` - Prompt fbank features for embedding [1, fbank_len, 80]
    /// * `sampling_k` - Top-k sampling
    ///
    /// # Returns
    /// Audio samples as i16
    pub fn synthesize_instruct_with_prompt_len(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_len: usize,
        prompt_speech_16k: &Tensor,
        prompt_speech_24k: &Tensor,
        prompt_fbank: &Tensor,
        sampling_k: usize,
    ) -> Result<Vec<i16>, NativeTtsError> {
        // 1. Process prompt audio
        let (prompt_speech_tokens, spk_emb) = self.process_prompt_tensors(prompt_speech_16k, prompt_fbank)?;

        // 2. Embed text tokens
        let text_embeds = self.llm.embed_text_tokens(text_tokens)?;

        // 3. Full Synthesis pipeline
        self.synthesize_full_with_prompt_len(
            &text_embeds,
            prompt_text_len,
            Some(&prompt_speech_tokens),
            prompt_speech_24k, // Flow uses 24k mel (80 dim)
            &spk_emb,
            sampling_k
        )
    }

    /// Synthesize speech in instruct/zero-shot mode (assumes no prompt text tokens)
    pub fn synthesize_instruct(
        &mut self,
        text_tokens: &Tensor,
        prompt_speech_16k: &Tensor,
        prompt_speech_24k: &Tensor,
        prompt_fbank: &Tensor,
        sampling_k: usize,
    ) -> Result<Vec<i16>, NativeTtsError> {
        self.synthesize_instruct_with_prompt_len(
            text_tokens,
            0,
            prompt_speech_16k,
            prompt_speech_24k,
            prompt_fbank,
            sampling_k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        // This test requires model files to exist
        let result = NativeTtsEngine::new("pretrained_models/Fun-CosyVoice3-0.5B", None);
        match result {
            Ok(_) => println!("Engine created successfully"),
            Err(e) => println!("Engine creation failed (expected if models missing): {}", e),
        }
    }
}
