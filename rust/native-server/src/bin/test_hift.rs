use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use cosyvoice_native_server::tts::NativeTtsEngine;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "pretrained_models/Fun-CosyVoice3-0.5B")]
    model_dir: String,

    #[arg(long, default_value = "debug_artifacts.safetensors")]
    artifacts_path: String,
}

fn main() -> Result<()> {
    println!("=== HiFT Parity Test ===");
    let args = Args::parse();

    // Setup paths
    let model_dir = PathBuf::from(&args.model_dir);

    // Initialize Device
    let device = Device::new_cuda(0).context("CUDA device required")?;
    println!("Using device: {:?}", device);

    // Load Engine
    println!("Loading NativeTtsEngine...");
    let engine = NativeTtsEngine::new(model_dir.to_str().unwrap(), Some(device.clone()))?;
    println!("Engine loaded.");

    // Load Debug Artifacts
    let artifact_path = PathBuf::from(&args.artifacts_path);
    if !artifact_path.exists() {
        return Err(anyhow::anyhow!(
            "Artifact file not found: {}",
            artifact_path.display()
        ));
    }

    println!("Loading artifacts from: {:?}", artifact_path);
    // Load to CPU first
    let tensors = candle_core::safetensors::load(&artifact_path, &Device::Cpu)?;
    println!("Loaded {} tensors. Keys: {:?}", tensors.len(), tensors.keys().collect::<Vec<_>>());

    // Get flow output (Mel)
    let mel = tensors
        .get("python_flow_output")
        .context("python_flow_output not found in artifacts")?;

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
                eprintln!(
                    "    python_f0 stats: min={:.6} Hz, max={:.6} Hz, mean={:.6} Hz",
                    min, max, mean
                );
            }
        }
    }

    if let Some(py_source) = tensors.get("python_hift_source") {
        eprintln!("Loaded python_hift_source shape: {:?}, dtype: {:?}", py_source.shape(), py_source.dtype());
        let py_source = py_source.to_dtype(candle_core::DType::F32).unwrap_or(py_source.clone());
        if let Ok(flat) = py_source.flatten_all() {
             if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    python_hift_source stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }
    }

    if let Some(py_conv_pre) = tensors.get("python_conv_pre") {
        eprintln!("Loaded python_conv_pre shape: {:?}, dtype: {:?}", py_conv_pre.shape(), py_conv_pre.dtype());
        let py_conv_pre = py_conv_pre.to_dtype(candle_core::DType::F32).unwrap_or(py_conv_pre.clone());
        if let Ok(flat) = py_conv_pre.flatten_all() {
             if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    python_conv_pre stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }
    }

    if let Some(py_ups0) = tensors.get("python_ups_0") {
        eprintln!("Loaded python_ups_0 shape: {:?}, dtype: {:?}", py_ups0.shape(), py_ups0.dtype());
        let py_ups0 = py_ups0.to_dtype(candle_core::DType::F32).unwrap_or(py_ups0.clone());
        if let Ok(flat) = py_ups0.flatten_all() {
             if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    python_ups_0 stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }
    }

    if let Some(py_l0) = tensors.get("python_loop_0_output") {
        eprintln!("Loaded python_loop_0_output shape: {:?}, dtype: {:?}", py_l0.shape(), py_l0.dtype());
        let py_l0 = py_l0.to_dtype(candle_core::DType::F32).unwrap_or(py_l0.clone());
        if let Ok(flat) = py_l0.flatten_all() {
             if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    python_loop_0_output stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }
    }

    if let Some(py_ups1) = tensors.get("python_ups_1") {
        eprintln!("Loaded python_ups_1 shape: {:?}, dtype: {:?}", py_ups1.shape(), py_ups1.dtype());
        let py_ups1 = py_ups1.to_dtype(candle_core::DType::F32).unwrap_or(py_ups1.clone());
        if let Ok(flat) = py_ups1.flatten_all() {
             if let Ok(vec) = flat.to_vec1::<f32>() {
                let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = vec.iter().sum();
                let mean = sum / vec.len() as f32;
                eprintln!(
                    "    python_ups_1 stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
            }
        }
    }

    println!("\nRunning synthesize_from_mel...");
    let audio_samples = engine.synthesize_from_mel(&mel_dev)?;

    println!("Generated {} samples.", audio_samples.len());

    // Save WAV
    let output_dir = artifact_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let output_path = output_dir.join("test_hift_output.wav");
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
