//! Test binary for verifying native TTS component weight loading.

use cosyvoice_rust_backend::native_tts::NativeTtsEngine;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set ORT path if not set
    /*
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        let path = std::path::Path::new("../../.pixi/envs/default/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.19.2");
        if path.exists() {
             // Convert to absolute path because ORT might need it or CWD context matters
             if let Ok(abs_path) = std::fs::canonicalize(path) {
                 println!("Setting ORT_DYLIB_PATH to {:?}", abs_path);
                 std::env::set_var("ORT_DYLIB_PATH", abs_path);
             }
        } else {
             println!("Warning: ORT lib not found at expected path: {:?}", path);
        }
    }
    */

    println!("=== Testing Native TTS Engine End-to-End ===\n");

    let model_dir = std::env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| "../../pretrained_models/Fun-CosyVoice3-0.5B".to_string());

    println!("Model directory: {}", model_dir);
    println!("Current Directory: {:?}", std::env::current_dir().unwrap());

    // Load test artifacts
    let artifact_path_str = "../../tests/test_artifacts.safetensors";
    let artifact_path = Path::new(artifact_path_str);
    if !artifact_path.exists() {
        eprintln!("Error: {} not found. Run 'pixi run python tests/generate_test_artifacts.py' first.", artifact_path_str);
        std::process::exit(1);
    }

    println!("Candle CUDA available: {}", candle_core::utils::cuda_is_available());
    let device = if candle_core::utils::cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to create CUDA device 0: {}", e);
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    };
    println!("Selected device: {:?}", device);

    let cpu = Device::Cpu;
    let artifacts = safe_load_artifacts(artifact_path.to_str().unwrap(), &cpu)?;

    // Unpack artifacts and move to device
    let flow_feat_24k = artifacts.get("flow_feat_24k").expect("Missing flow_feat_24k").to_device(&device)?;
    let text_ids = artifacts.get("text_ids").expect("Missing text_ids").to_dtype(DType::U32)?.to_device(&device)?;

    // New artifacts for bypassing frontend
    let speech_tokens = artifacts.get("speech_tokens").expect("Missing speech_tokens in artifacts. Did you upgrade generate_test_artifacts.py?").to_dtype(DType::U32)?.to_device(&device)?;
    let speaker_embedding = artifacts.get("speaker_embedding").expect("Missing speaker_embedding in artifacts.").to_device(&device)?;

    // Initialize Engine
    let mut engine = NativeTtsEngine::new(&model_dir, Some(device.clone()))?;
    println!("\n✅ Engine loaded successfully!");

    // Run synthesis
    println!("Synthesizing (Direct/Bypassing Frontend)...");

    // 1. Embed text tokens
    println!("Embedding text tokens...");
    let text_embeds = engine.llm.embed_text_tokens(&text_ids)?;

    // 2. Full Synthesis
    // Force sampling_k=1 for determinism if possible, or 25 as config
    let audio_samples = engine.synthesize_full(
        &text_embeds,
        Some(&speech_tokens),
        &flow_feat_24k,
        &speaker_embedding,
        25
    )?;

    println!("Synthesis complete! Generated {} samples.", audio_samples.len());

    // Save to WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("native_output.wav", spec)?;
    for sample in audio_samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    println!("Saved audio to native_output.wav");

    Ok(())
}

fn safe_load_artifacts(path: &str, device: &Device) -> Result<std::collections::HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    use candle_core::safetensors::load;
    let tensors = load(path, device)?;
    Ok(tensors)
}
