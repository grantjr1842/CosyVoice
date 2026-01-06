use anyhow::{Result, Context};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use cosyvoice_native_server::hift::{HiFTGenerator, HiFTConfig};
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
    let args = Args::parse();
    println!("=== HiFT Standalone Parity Test ===");

    // Choose device
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    // For stability, use F32
    let dtype = DType::F32;

    // Load HiFT Model
    println!("Loading HiFT model from {}...", args.model_dir);
    let hift_path = PathBuf::from(&args.model_dir).join("hift.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[hift_path], dtype, &device)? };

    let config = HiFTConfig::default();
    let hift = HiFTGenerator::new(vb, &config)?;
    println!("HiFT loaded.");

    // Load Artifacts
    println!("Loading artifacts from {}...", args.artifacts_path);
    let artifacts = candle_core::safetensors::load(&args.artifacts_path, &Device::Cpu)?;

    let mel = artifacts.get("python_flow_output").context("python_flow_output not found")?;
    let expected_audio = artifacts.get("python_audio_output").context("python_audio_output not found")?;

    println!("  mel (input): {:?}", mel.shape());
    println!("  expected_audio: {:?}", expected_audio.shape());

    // Move mel to device and convert to dtype
    let mel = mel.to_dtype(dtype)?.to_device(&device)?;

    // Run HiFT
    println!("\nRunning HiFT inference...");
    let generated_audio = hift.forward(&mel)?;

    println!("  generated_audio shape: {:?}", generated_audio.shape());

    // Move to CPU for comparison
    let generated_cpu = generated_audio.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let expected_f32 = expected_audio.to_dtype(DType::F32)?;

    // Flatten for comparison
    let gen_vec = generated_cpu.flatten_all()?.to_vec1::<f32>()?;
    let exp_vec = expected_f32.flatten_all()?.to_vec1::<f32>()?;

    println!("  generated samples: {}", gen_vec.len());
    println!("  expected samples: {}", exp_vec.len());

    // Print stats
    let gen_min = gen_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let gen_max = gen_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gen_mean: f32 = gen_vec.iter().sum::<f32>() / gen_vec.len() as f32;

    let exp_min = exp_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let exp_max = exp_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_mean: f32 = exp_vec.iter().sum::<f32>() / exp_vec.len() as f32;

    println!("\n  Generated audio stats: min={:.6}, max={:.6}, mean={:.6}", gen_min, gen_max, gen_mean);
    println!("  Expected audio stats:  min={:.6}, max={:.6}, mean={:.6}", exp_min, exp_max, exp_mean);

    // Compare (truncate to min length)
    let compare_len = std::cmp::min(gen_vec.len(), exp_vec.len());

    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    for i in 0..compare_len {
        let diff = (gen_vec[i] - exp_vec[i]).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
    }
    let mean_diff = total_diff / compare_len as f32;

    println!("\nParity Results:");
    println!("  Max Diff: {:.6}", max_diff);
    println!("  Mean Diff: {:.6}", mean_diff);

    // Pass/Fail threshold
    if max_diff > 0.1 {
        println!("FAIL: Max diff {:.6} > 0.1", max_diff);

        // Save debug tensors
        let debug_save = std::collections::HashMap::from([
            ("generated".to_string(), generated_cpu),
            ("expected".to_string(), expected_f32),
        ]);
        candle_core::safetensors::save(&debug_save, "hift_failure_debug.safetensors")?;
        println!("Saved failure debug tensors to hift_failure_debug.safetensors");
    } else {
        println!("SUCCESS: HiFT output matches Python reference");
    }

    Ok(())
}
