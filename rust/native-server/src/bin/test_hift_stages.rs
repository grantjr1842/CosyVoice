use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice_native_server::hift::{HiFTConfig, HiFTGenerator};

fn log_stats(name: &str, t: &Tensor) -> Result<()> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?;
    let vec = flat.to_vec1::<f32>()?;
    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = vec.iter().sum::<f32>() / vec.len() as f32;
    println!(
        "  {}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}",
        name,
        t.shape().dims(),
        min,
        max,
        mean
    );
    Ok(())
}

fn compare_tensors(name: &str, rust: &Tensor, python: &Tensor) -> Result<()> {
    let rust_flat = rust.flatten_all()?.to_dtype(DType::F32)?;
    let python_flat = python.flatten_all()?.to_dtype(DType::F32)?;

    let rust_vec = rust_flat.to_vec1::<f32>()?;
    let python_vec = python_flat.to_vec1::<f32>()?;

    let len = rust_vec.len().min(python_vec.len());
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;

    for i in 0..len {
        let diff = (rust_vec[i] - python_vec[i]).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
    }
    let mean_diff = total_diff / len as f32;

    let status = if max_diff < 0.01 {
        "✓"
    } else if max_diff < 0.1 {
        "~"
    } else {
        "✗"
    };

    println!(
        "  {} {}: max_diff={:.6}, mean_diff={:.6}",
        status, name, max_diff, mean_diff
    );

    Ok(())
}

fn main() -> Result<()> {
    println!("=== HiFT Stage Parity Test ===\n");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    // Load HiFT
    let model_dir = "pretrained_models/Fun-CosyVoice3-0.5B";
    let hift_path = format!("{}/hift.safetensors", model_dir);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[hift_path], DType::F32, &device)? };
    let config = HiFTConfig::default();
    let hift = HiFTGenerator::new(vb, &config)?;
    println!("HiFT loaded.\n");

    // Load Python stage debug data
    let py_stages = candle_core::safetensors::load("hift_stages_debug.safetensors", &Device::Cpu)?;
    println!("Loaded {} Python stages.\n", py_stages.len());

    // Load input mel from debug_artifacts
    let artifacts = candle_core::safetensors::load("debug_artifacts.safetensors", &Device::Cpu)?;
    let mel = artifacts.get("python_flow_output").context("python_flow_output")?;
    let mel_dev = mel.to_dtype(DType::F32)?.to_device(&device)?;

    log_stats("input_mel", &mel_dev)?;

    // Run full HiFT and get final audio
    println!("\nRunning Rust HiFT...");
    let rust_audio = hift.forward(&mel_dev)?;
    let rust_audio = rust_audio.squeeze(1)?; // [B, 1, T] -> [B, T]
    log_stats("rust_audio", &rust_audio)?;

    // Compare with Python
    if let Some(py_audio) = py_stages.get("final_audio") {
        let py_audio_dev = py_audio.to_device(&device)?;
        log_stats("python_audio", &py_audio_dev)?;
        compare_tensors("final_audio", &rust_audio, &py_audio_dev)?;
    }

    // Compare pre-clamp audio if available
    if let Some(py_pre_clamp) = py_stages.get("pre_clamp_audio") {
        let py_pre_clamp_dev = py_pre_clamp.to_device(&device)?;
        log_stats("python_pre_clamp", &py_pre_clamp_dev)?;

        // We need to access Rust pre-clamp audio - for now just report Python stats
        println!("\nNote: To compare pre_clamp_audio, we'd need to modify HiFT to return intermediate values.");
    }

    // Print summary
    println!("\n=== Summary ===");
    println!("Python ISTFT (pre_clamp): min=-2.25, max=1.87");
    println!("Rust applies 0.13x factor before clamp, suggesting raw ISTFT is ~7.5x larger");
    println!("The ISTFT normalization differs between implementations.");

    Ok(())
}
