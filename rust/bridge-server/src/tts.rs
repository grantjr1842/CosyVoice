//! CosyVoice TTS Engine wrapper using PyO3.
//!
//! This module uses the CosyVoice3 Python model for TTS synthesis.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TtsError {
    #[error("TTS initialization failed: {0}")]
    InitError(String),
    #[error("Synthesis failed: {0}")]
    SynthesisError(String),
    #[error("Python error: {0}")]
    PythonError(String),
}

impl From<PyErr> for TtsError {
    fn from(err: PyErr) -> Self {
        TtsError::PythonError(err.to_string())
    }
}

/// TTS Engine using CosyVoice3 Python model via PyO3.
pub struct TtsEngine {
    model: Arc<Mutex<PyObject>>,
    sample_rate: u32,
}

// SAFETY: We manage GIL and use Arc/Mutex
unsafe impl Send for TtsEngine {}
unsafe impl Sync for TtsEngine {}

impl TtsEngine {
    /// Create a new TTS engine by initializing the CosyVoice3 model.
    pub fn new(model_dir: &str) -> Result<Self, TtsError> {
        Python::with_gil(|py| {
            // Add project root and third_party paths to Python path
            let sys = py.import("sys")?;
            let path_attr = sys.getattr("path")?;
            let path: Bound<'_, PyList> = path_attr
                .downcast_into()
                .map_err(|e| TtsError::InitError(format!("Failed to get sys.path: {}", e)))?;

            // Matcha-TTS path no longer needed - using cosyvoice.compat.matcha_compat
            path.insert(0, "/home/grant/github/CosyVoice-1")?;

            // Import and initialize CosyVoice3
            let cosyvoice_module = py.import("cosyvoice.cli.cosyvoice")?;
            let cosyvoice_class = cosyvoice_module.getattr("CosyVoice3")?;

            // Initialize model with model_dir
            let model = cosyvoice_class.call1((model_dir,))?;

            let sample_rate: u32 = model.getattr("sample_rate")?.extract()?;

            Ok(Self {
                model: Arc::new(Mutex::new(model.unbind())),
                sample_rate,
            })
        })
    }

    /// Synthesize speech from text using zero-shot voice cloning.
    pub fn synthesize_zero_shot(
        &self,
        text: &str,
        prompt_audio_path: &str,
        prompt_text: &str,
        _speed: f32,
    ) -> Result<(Vec<i16>, u32), TtsError> {
        Python::with_gil(|py| {
            let model = self
                .model
                .lock()
                .map_err(|e| TtsError::SynthesisError(e.to_string()))?;
            let model = model.bind(py);

            // Call inference_zero_shot (returns a generator)
            let generator = model.call_method1(
                "inference_zero_shot",
                (text, prompt_text, prompt_audio_path),
            )?;

            let mut all_samples: Vec<i16> = Vec::new();

            // Iterate over the generator
            for result in generator.try_iter()? {
                let output = result?;
                let speech_tensor = output.get_item("tts_speech")?;

                // Convert tensor to samples: speech is [1, samples] shape
                let samples_list = speech_tensor
                    .call_method0("squeeze")?
                    .call_method0("cpu")?
                    .call_method0("numpy")?
                    .call_method0("tolist")?;

                let samples_f32: Vec<f32> = samples_list.extract()?;

                // Convert f32 to i16
                all_samples.extend(
                    samples_f32
                        .iter()
                        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16),
                );
            }

            Ok((all_samples, self.sample_rate))
        })
    }

    /// Synthesize speech using instruction-based synthesis.
    pub fn synthesize_instruct(
        &self,
        text: &str,
        instruct_text: &str,
        prompt_audio_path: &str,
        _speed: f32,
    ) -> Result<(Vec<i16>, u32), TtsError> {
        Python::with_gil(|py| {
            let model = self
                .model
                .lock()
                .map_err(|e| TtsError::SynthesisError(e.to_string()))?;
            let model = model.bind(py);

            // Call inference_instruct2
            let generator = model.call_method1(
                "inference_instruct2",
                (text, instruct_text, prompt_audio_path),
            )?;

            let mut all_samples: Vec<i16> = Vec::new();

            // Iterate over the generator
            for result in generator.try_iter()? {
                let output = result?;
                let speech_tensor = output.get_item("tts_speech")?;

                // Convert tensor to samples
                let samples_list = speech_tensor
                    .call_method0("squeeze")?
                    .call_method0("cpu")?
                    .call_method0("numpy")?
                    .call_method0("tolist")?;

                let samples_f32: Vec<f32> = samples_list.extract()?;

                all_samples.extend(
                    samples_f32
                        .iter()
                        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16),
                );
            }

            Ok((all_samples, self.sample_rate))
        })
    }

    /// Get available speakers (for SFT mode).
    pub fn list_speakers(&self) -> Result<Vec<String>, TtsError> {
        Python::with_gil(|py| {
            let model = self
                .model
                .lock()
                .map_err(|e| TtsError::SynthesisError(e.to_string()))?;
            let model = model.bind(py);

            let speakers_list = model.call_method0("list_available_spks")?;
            let speakers: Vec<String> = speakers_list.extract()?;
            Ok(speakers)
        })
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
