//! CosyVoice TTS Engine wrapper using PyO3.
//!
//! This module bridges to the Python CosyVoice implementation for TTS synthesis.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;
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

/// TTS Engine using CosyVoice Python backend via PyO3.
pub struct TtsEngine {
    cosyvoice: Mutex<PyObject>,
    sample_rate: u32,
}

// SAFETY: PyO3 handles GIL properly, we wrap in Mutex
unsafe impl Send for TtsEngine {}
unsafe impl Sync for TtsEngine {}

impl TtsEngine {
    /// Create a new TTS engine by initializing CosyVoice Python model.
    pub fn new(model_dir: &str) -> Result<Self, TtsError> {
        Python::with_gil(|py| {
            // Add project root and third_party paths to Python path
            let sys = py.import("sys")?;
            let path_attr = sys.getattr("path")?;
            let path: Bound<'_, pyo3::types::PyList> = path_attr.downcast_into()
                .map_err(|e| TtsError::InitError(format!("Failed to get sys.path: {}", e)))?;

            // Add paths in reverse order (last added = first searched)
            path.insert(0, "/home/grant/github/CosyVoice-1/third_party/Matcha-TTS")?;
            path.insert(0, "/home/grant/github/CosyVoice-1")?;

            // Import CosyVoice
            let cosyvoice_module = py.import("cosyvoice.cli.cosyvoice")?;
            let cosyvoice_class = cosyvoice_module.getattr("CosyVoice3")?;

            // Initialize with model directory
            let kwargs = PyDict::new(py);
            kwargs.set_item("model_dir", model_dir)?;

            let cosyvoice = cosyvoice_class.call((), Some(&kwargs))?;

            Ok(Self {
                cosyvoice: Mutex::new(cosyvoice.into()),
                sample_rate: 24000, // CosyVoice uses 24kHz
            })
        })
    }

    /// Synthesize speech from text using zero-shot voice cloning.
    pub fn synthesize_zero_shot(
        &self,
        text: &str,
        prompt_audio_path: &str,
        prompt_text: &str,
        speed: f32,
    ) -> Result<(Vec<i16>, u32), TtsError> {
        let cosyvoice = self.cosyvoice.lock()
            .map_err(|e| TtsError::SynthesisError(format!("Lock error: {}", e)))?;

        Python::with_gil(|py| {
            let cv = cosyvoice.bind(py);

            // Call inference_zero_shot method
            let kwargs = PyDict::new(py);
            kwargs.set_item("tts_text", text)?;
            kwargs.set_item("prompt_text", prompt_text)?;
            kwargs.set_item("prompt_wav", prompt_audio_path)?;
            kwargs.set_item("speed", speed)?;
            kwargs.set_item("stream", false)?;

            let result = cv.call_method("inference_zero_shot", (), Some(&kwargs))?;

            // Result is a generator, get the first (and only) result
            let iter = result.try_iter()?;
            let mut samples: Vec<i16> = Vec::new();

            for item in iter {
                let audio_dict = item?;
                // Use getattr to access dict-like object
                let tts_speech = audio_dict.get_item("tts_speech")?;

                // Convert tensor to numpy array to Vec<f32>
                let numpy_array = tts_speech.call_method0("cpu")?.call_method0("numpy")?;
                let flat = numpy_array.call_method0("flatten")?;

                // Get as Python list and convert
                let py_samples: Vec<f32> = flat.extract()?;
                samples.extend(py_samples.iter().map(|s: &f32| (*s * 32767.0).clamp(-32768.0, 32767.0) as i16));
            }

            Ok((samples, self.sample_rate))
        })
    }

    /// Synthesize speech using instruction-based synthesis.
    /// Uses inference_instruct2 with a default prompt audio.
    pub fn synthesize_instruct(
        &self,
        text: &str,
        instruct_text: &str,
        prompt_audio_path: &str,
        speed: f32,
    ) -> Result<(Vec<i16>, u32), TtsError> {
        let cosyvoice = self.cosyvoice.lock()
            .map_err(|e| TtsError::SynthesisError(format!("Lock error: {}", e)))?;

        Python::with_gil(|py| {
            let cv = cosyvoice.bind(py);

            let kwargs = PyDict::new(py);
            kwargs.set_item("tts_text", text)?;
            kwargs.set_item("instruct_text", instruct_text)?;
            kwargs.set_item("prompt_wav", prompt_audio_path)?;
            kwargs.set_item("speed", speed)?;
            kwargs.set_item("stream", false)?;

            let result = cv.call_method("inference_instruct2", (), Some(&kwargs))?;

            let iter = result.try_iter()?;
            let mut samples: Vec<i16> = Vec::new();

            for item in iter {
                let audio_dict = item?;
                let tts_speech = audio_dict.get_item("tts_speech")?;

                let numpy_array = tts_speech.call_method0("cpu")?.call_method0("numpy")?;
                let flat = numpy_array.call_method0("flatten")?;
                let py_samples: Vec<f32> = flat.extract()?;
                samples.extend(py_samples.iter().map(|s: &f32| (*s * 32767.0).clamp(-32768.0, 32767.0) as i16));
            }

            Ok((samples, self.sample_rate))
        })
    }

    /// Get available speakers (for SFT mode).
    pub fn list_speakers(&self) -> Result<Vec<String>, TtsError> {
        let cosyvoice = self.cosyvoice.lock()
            .map_err(|e| TtsError::SynthesisError(format!("Lock error: {}", e)))?;

        Python::with_gil(|py| {
            let cv = cosyvoice.bind(py);
            let speakers = cv.getattr("list_available_spks")?.call0()?;
            let speaker_list: Vec<String> = speakers.extract()?;
            Ok(speaker_list)
        })
    }

    /// Get sample rate.
    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
