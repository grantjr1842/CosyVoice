use anyhow::{Result, Context};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Module};
use clap::Parser;
use cosyvoice_native_server::hift::{HiFTGenerator, HiFTConfig};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "pretrained_models/Fun-CosyVoice3-0.5B")]
    model_dir: String,

    #[arg(long, default_value = "hift_decode_stages.safetensors")]
    artifacts_path: String,

    #[arg(long, default_value = "rust/debug_artifacts.safetensors.bak")]
    input_artifacts_path: String,
}

fn print_stats(name: &str, t: &Tensor) -> Result<()> {
    let vec = t.flatten_all()?.to_vec1::<f32>()?;
    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
    println!("  {}: shape={:?}, min={:.4}, max={:.4}, mean={:.6}", name, t.shape(), min, max, mean);
    Ok(())
}

fn compare_tensors(name: &str, rust: &Tensor, python: &Tensor) -> Result<(f32, f32)> {
    let rust_vec = rust.flatten_all()?.to_vec1::<f32>()?;
    let py_vec = python.flatten_all()?.to_vec1::<f32>()?;

    let compare_len = std::cmp::min(rust_vec.len(), py_vec.len());
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    for i in 0..compare_len {
        let diff = (rust_vec[i] - py_vec[i]).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
    }
    let mean_diff = total_diff / compare_len as f32;

    println!("  {}: max_diff={:.4}, mean_diff={:.6}", name, max_diff, mean_diff);
    Ok((max_diff, mean_diff))
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== HiFT Decode Stage Comparison ===\n");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    let dtype = DType::F32;

    // Load HiFT Model
    println!("Loading HiFT model from {}...", args.model_dir);
    let hift_path = PathBuf::from(&args.model_dir).join("hift.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[hift_path], dtype, &device)? };

    let config = HiFTConfig::default();
    let hift = HiFTGenerator::new(vb, &config)?;
    println!("HiFT loaded.\n");

    // Load Python stage intermediates
    println!("Loading Python intermediates from {}...", args.artifacts_path);
    let py_stages = candle_core::safetensors::load(&args.artifacts_path, &Device::Cpu)?;

    // Load input artifacts (mel, source)
    println!("Loading inputs from {}...", args.input_artifacts_path);
    let inputs = candle_core::safetensors::load(&args.input_artifacts_path, &Device::Cpu)?;

    let mel = inputs.get("python_flow_output").context("python_flow_output not found")?;
    let source = inputs.get("python_hift_source").context("python_hift_source not found")?;

    let mel = mel.to_dtype(dtype)?.to_device(&device)?;
    let source = source.to_dtype(dtype)?.to_device(&device)?;

    println!("\nInput shapes:");
    println!("  mel: {:?}", mel.shape());
    println!("  source: {:?}", source.shape());

    // Access internal conv_pre via direct access
    // Since HiFTGenerator doesn't expose internals, we'll run decode and compare final output
    // Then add getter methods in a follow-up if needed

    println!("\n=== Running Rust decode ===");
    let rust_audio = hift.decode_with_source(&mel, &source)?;
    let rust_audio_cpu = rust_audio.to_device(&Device::Cpu)?;

    print_stats("rust_audio", &rust_audio_cpu)?;

    // Compare with Python audio
    if let Some(py_audio) = py_stages.get("python_decode_audio") {
        let py_audio = py_audio.to_dtype(DType::F32)?;
        print_stats("python_audio", &py_audio)?;
        compare_tensors("audio", &rust_audio_cpu, &py_audio)?;
    }

    // Compare with Python convpost output
    if let Some(py_conv_post) = py_stages.get("conv_post_output") {
        let py_conv_post = py_conv_post.to_dtype(DType::F32)?;
        println!("\nPython conv_post stats:");
        print_stats("py_conv_post", &py_conv_post)?;
    }

    println!("\n=== Comparison Complete ===");
    println!("\nNote: To debug internal stages, need to expose intermediate tensors from HiFTGenerator.");

    Ok(())
}
