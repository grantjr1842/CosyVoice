//! ONNX Frontend module for CosyVoice using native ort crate.
//!
//! This module handles:
//! - Speech tokenization (mel → speech tokens)
//! - Speaker embedding extraction (audio → embedding)
//!
//! Uses the ONNX Runtime via the `ort` crate.

use candle_core::{Device, Tensor};
// Import ExecutionProvider trait
#[cfg(feature = "cuda")]
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider,
};
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::env;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FrontendError {
    #[error("Ort error: {0}")]
    OrtError(String),
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    #[error("Execution Provider error: {0}")]
    ProviderError(String),
}

/// ONNX-based frontend for speech tokenization and speaker extraction
pub struct OnnxFrontend {
    speech_tokenizer: Session,
    campplus: Session,
    device: Device,
}

impl OnnxFrontend {
    /// Create a new ONNX frontend
    pub fn new(model_dir: &str, device: Device) -> Result<Self, FrontendError> {
        println!("OnnxFrontend::new called");
        let model_path = PathBuf::from(model_dir);

        // Initialize ORT
        println!("Initializing ORT...");
        let _ = ort::init().with_name("cosyvoice").commit();
        println!("ORT initialized.");

        let speech_tokenizer_path = model_path.join("speech_tokenizer_v3.onnx");
        let campplus_path = model_path.join("campplus.onnx");

        // Load models into memory
        println!("Reading model files...");
        let speech_tokenizer_bytes = std::fs::read(&speech_tokenizer_path).map_err(|e| {
            FrontendError::ModelLoad(format!("Failed to read speech tokenizer: {}", e))
        })?;
        let campplus_bytes = std::fs::read(&campplus_path)
            .map_err(|e| FrontendError::ModelLoad(format!("Failed to read campplus: {}", e)))?;
        println!("Model files read.");

        // Initialize ORT sessions
        eprintln!("Creating speech_tokenizer session (builder init)...");
        let intra_threads = env::var("COSYVOICE_ORT_INTRA_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let inter_threads = env::var("COSYVOICE_ORT_INTER_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);

        let builder = Session::builder()
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_inter_threads(inter_threads)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Register Execution Providers (TensorRT -> CUDA -> CPU)
        #[cfg(feature = "cuda")]
        let builder = builder
            .with_execution_providers(vec![
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e: ort::Error| FrontendError::OrtError(e.to_string()))?;

        eprintln!("Speech Tokenizer bytes: {}", speech_tokenizer_bytes.len());
        eprintln!("Threads set. Committing from memory...");
        let speech_tokenizer = builder
            .commit_from_memory(&speech_tokenizer_bytes)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;
        eprintln!("Speech tokenizer session created.");

        eprintln!("Creating campplus session...");
        let builder = Session::builder()
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?
            .with_inter_threads(inter_threads)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Register Execution Providers for Campplus
        #[cfg(feature = "cuda")]
        let builder = builder
            .with_execution_providers(vec![
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e: ort::Error| FrontendError::OrtError(e.to_string()))?;

        eprintln!("Campplus bytes: {}", campplus_bytes.len());
        eprintln!("Committing campplus from memory...");
        let campplus = builder
            .commit_from_memory(&campplus_bytes)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;
        eprintln!("Campplus session created.");

        Ok(Self {
            speech_tokenizer,
            campplus,
            device,
        })
    }

    /// Extract speaker embedding from audio fbank features
    ///
    /// # Arguments
    /// * `fbank` - Fbank features [batch, frames, 80]
    ///
    /// # Returns
    /// Speaker embedding tensor [batch, embed_dim]
    pub fn extract_speaker_embedding(&mut self, fbank: &Tensor) -> Result<Tensor, FrontendError> {
        // Convert Candle tensor to Vec and shape
        let fbank_vec: Vec<f32> = fbank.flatten_all()?.to_vec1()?;
        let (b, t, d) = fbank.dims3()?;

        // Create Ort Value from (shape, data)
        let input_val = Value::from_array(([b, t, d], fbank_vec))
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Run inference
        let inputs = inputs![
            "input" => input_val
        ];
        let outputs = self
            .campplus
            .run(inputs)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Output is (shape, data_slice)
        let (shape, data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        let out_dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let out_data: Vec<f32> = data.to_vec();

        Tensor::from_vec(out_data, out_dims, &self.device).map_err(FrontendError::CandleError)
    }

    /// Tokenize speech from mel spectrogram
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, 128, frames]
    /// * `mel_length` - Length of mel spectrogram
    ///
    /// # Returns
    /// Speech tokens tensor [batch, tokens]
    pub fn tokenize_speech(
        &mut self,
        mel: &Tensor,
        mel_length: i32,
    ) -> Result<Tensor, FrontendError> {
        // Mel input
        let mel_vec: Vec<f32> = mel.flatten_all()?.to_vec1()?;
        let (b, c, t) = mel.dims3()?;
        let mel_val = Value::from_array(([b, c, t], mel_vec))
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Length input
        let len_vec = vec![mel_length];
        let len_val = Value::from_array(([1], len_vec))
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        // Inputs
        let inputs = inputs![
            "feats" => mel_val,
            "feats_length" => len_val
        ];
        let outputs = self
            .speech_tokenizer
            .run(inputs)
            .map_err(|e| FrontendError::OrtError(e.to_string()))?;

        let (out_dims, out_data): (Vec<usize>, Vec<u32>) =
            if let Ok((shape, data)) = outputs["indices"].try_extract_tensor::<i64>() {
                let dims = shape.iter().map(|&x| x as usize).collect();
                let values = data.iter().map(|&x| x as u32).collect();
                (dims, values)
            } else if let Ok((shape, data)) = outputs["indices"].try_extract_tensor::<i32>() {
                let dims = shape.iter().map(|&x| x as usize).collect();
                let values = data.iter().map(|&x| x as u32).collect();
                (dims, values)
            } else {
                return Err(FrontendError::OrtError(
                    "Speech tokenizer output type is neither i64 nor i32".to_string(),
                ));
            };

        Tensor::from_vec(out_data, out_dims, &self.device).map_err(FrontendError::CandleError)
    }
}
