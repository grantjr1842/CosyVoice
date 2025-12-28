//! Shared types for CosyVoice TTS server and client.

use serde::{Deserialize, Serialize};

/// Request to synthesize speech from text (zero-shot voice cloning).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeRequest {
    /// The text to synthesize.
    pub text: String,
    /// Path to reference audio file for voice cloning.
    #[serde(default)]
    pub prompt_audio: Option<String>,
    /// Transcription of the reference audio.
    #[serde(default)]
    pub prompt_text: Option<String>,
    /// Speaker ID for SFT mode (if not using zero-shot).
    #[serde(default)]
    pub speaker: Option<String>,
    /// Speech speed multiplier (0.5 to 2.0).
    #[serde(default = "default_speed")]
    pub speed: f32,
}

fn default_speed() -> f32 {
    1.0
}

/// Response metadata for synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeResponse {
    /// Duration of the generated audio in seconds.
    pub duration_secs: f32,
    /// Sample rate of the audio.
    pub sample_rate: u32,
    /// Number of audio samples.
    pub num_samples: usize,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speakers: Option<Vec<String>>,
}

/// Error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

/// Server configuration constants.
pub mod config {
    /// Default server port.
    pub const DEFAULT_PORT: u16 = 3000;
    /// Default sample rate for CosyVoice.
    pub const SAMPLE_RATE: u32 = 24000;
    /// Server version.
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");
    /// Default model directory.
    pub const DEFAULT_MODEL_DIR: &str = "pretrained_models/CosyVoice2-0.5B";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_request_defaults() {
        let json = r#"{"text": "Hello"}"#;
        let req: SynthesizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.speed, 1.0);
        assert!(req.prompt_audio.is_none());
    }
}
