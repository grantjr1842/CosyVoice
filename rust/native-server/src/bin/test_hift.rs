use candle_core::{Device, Tensor};
use cosyvoice_native_server::tts::NativeTtsEngine;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HiFT Parity Test ===");

    // Setup paths
    // Use absolute path to ensure model finding
    let repo_root = PathBuf::from("/home/grant/github/CosyVoice-1");
    let model_dir = repo_root.join("pretrained_models/Fun-CosyVoice3-0.5B");

    // Initialize Device
    // Try CUDA if available, else CPU
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // Load Engine
    println!("Loading NativeTtsEngine...");
    let engine = NativeTtsEngine::new(model_dir.to_str().unwrap(), Some(device.clone()))?;
    println!("Engine loaded.");

    // Load Debug Artifacts
    let artifact_path = "debug_artifacts.safetensors";
    if !Path::new(artifact_path).exists() {
        return Err(format!("Artifact file not found: {}", artifact_path).into());
    }

    println!("Loading artifacts from: {}", artifact_path);
    // Load to CPU first
    let tensors = candle_core::safetensors::load(artifact_path, &Device::Cpu)?;

    // Get flow output (Mel)
    let mel = tensors.get("python_flow_output")
        .ok_or("python_flow_output not found in artifacts")?;

    println!("Loaded Mel shape: {:?}", mel.shape());

    // Move to device
    let mel_dev = mel.to_device(&device)?;

    // Run Synthesis (HiFT only)
    // Load python_f0 if available
    if let Some(py_f0) = tensors.get("python_f0") {
         eprintln!("Loaded python_f0 shape: {:?}", py_f0.shape());
         if let Ok(flat) = py_f0.flatten_all() {
            if let Ok(vec) = flat.to_vec1::<f32>() {
                 let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                 let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                 let sum: f32 = vec.iter().sum();
                 let mean = sum / vec.len() as f32;
                 eprintln!("    python_f0 stats: min={:.6} Hz, max={:.6} Hz, mean={:.6} Hz", min, max, mean);
            }
         }
    }

    println!("\nRunning synthesize_from_mel...");
    let audio_samples = engine.synthesize_from_mel(&mel_dev)?;

    println!("Generated {} samples.", audio_samples.len());

    // Save WAV
    let output_path = repo_root.join("output/test_hift_output.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&output_path, spec)?;
    for sample in audio_samples.iter() {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    println!("Saved output to: {:?}", output_path);

    // Analyze output (simple check)
    let max_amp = audio_samples.iter().map(|s| s.abs()).max().unwrap_or(0);
    println!("Max Amplitude (i16): {}", max_amp);
    if max_amp > 32700 {
        println!("WARNING: Output likely clipped!");
    } else {
        println!("Output seems within range.");
    }

    Ok(())
}
