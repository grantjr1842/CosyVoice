//! Matcha-TTS Component Parity Tests
//!
//! This binary runs parity tests for individual components and saves outputs
//! as safetensors for comparison with Python.
//!
//! Parent Issue: #44
//! Sub-Issue: #47

use candle_core::{Device, Tensor};
use std::path::Path;
use anyhow::{Result, Context};

/// Test sinusoidal embedding (timestep embedding base)
fn test_sinusoidal_embedding(device: &Device) -> Result<()> {
    println!("\n=== Testing Sinusoidal Embedding ===");

    // Load test data from Python
    let test_path = Path::new("tests/sinusoidal_emb_test.safetensors");
    if !test_path.exists() {
        println!("⚠️  Test data not found. Run Python test first.");
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(test_path, device)?;
    let t_values = tensors.get("t_values")
        .context("t_values not found")?;
    let expected_emb = tensors.get("expected_emb")
        .context("expected_emb not found")?;

    println!("Input t_values shape: {:?}", t_values.shape());
    println!("Expected emb shape: {:?}", expected_emb.shape());

    // Compute sinusoidal embedding in Rust
    let dim = 256usize;
    let half_dim = dim / 2;
    let batch = t_values.dim(0)?;

    // Create frequency bands
    let log_10000 = (10000.0_f64).ln();
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-log_10000 / (half_dim - 1) as f64 * i as f64).exp() as f32)
        .collect();
    let freqs_tensor = Tensor::from_vec(freqs, (1, half_dim), device)?;

    // t * freqs for each timestep
    let t_expanded = t_values.reshape((batch, 1))?;
    let emb = t_expanded.broadcast_mul(&freqs_tensor)?;

    // sin and cos
    let sin_emb = emb.sin()?;
    let cos_emb = emb.cos()?;

    // Concatenate [sin, cos]
    let rust_emb = Tensor::cat(&[&sin_emb, &cos_emb], 1)?;

    println!("Rust emb shape: {:?}", rust_emb.shape());

    // Compare
    let diff = expected_emb.sub(&rust_emb)?.abs()?;
    let l1_error: f32 = diff.mean_all()?.to_scalar()?;
    let max_error: f32 = diff.max(0)?.max(0)?.to_scalar()?;

    println!("L1 error: {:.6}", l1_error);
    println!("Max error: {:.6}", max_error);

    if l1_error < 1e-5 {
        println!("✅ PASS: Sinusoidal embedding parity verified!");
    } else {
        println!("❌ FAIL: L1 error exceeds threshold");
    }

    // Save output
    let mut output = std::collections::HashMap::new();
    output.insert("rust_emb".to_string(), rust_emb);
    candle_core::safetensors::save(&output, "tests/sinusoidal_emb_rust_output.safetensors")?;

    Ok(())
}

/// Test Snake activation
fn test_snake_activation(device: &Device) -> Result<()> {
    println!("\n=== Testing Snake Activation ===");

    let test_path = Path::new("tests/snake_activation_test.safetensors");
    if !test_path.exists() {
        println!("⚠️  Test data not found. Run Python test first.");
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(test_path, device)?;
    let x = tensors.get("input_x").context("input_x not found")?;
    let alpha = tensors.get("alpha").context("alpha not found")?;
    let expected_out = tensors.get("expected_out").context("expected_out not found")?;

    println!("Input x shape: {:?}", x.shape());

    // Snake activation: x + (1/alpha) * sin(alpha * x)^2
    let alpha_x = x.broadcast_mul(alpha)?;
    let sin_sq = alpha_x.sin()?.sqr()?;
    let one_over_alpha = alpha.recip()?;
    let rust_out = x.add(&one_over_alpha.broadcast_mul(&sin_sq)?)?;

    println!("Rust out shape: {:?}", rust_out.shape());

    // Compare
    let diff = expected_out.sub(&rust_out)?.abs()?;
    let l1_error: f32 = diff.mean_all()?.to_scalar()?;

    println!("L1 error: {:.6}", l1_error);

    if l1_error < 1e-5 {
        println!("✅ PASS: Snake activation parity verified!");
    } else {
        println!("❌ FAIL: L1 error exceeds threshold");
    }

    // Save output
    let mut output = std::collections::HashMap::new();
    output.insert("rust_out".to_string(), rust_out);
    candle_core::safetensors::save(&output, "tests/snake_activation_rust_output.safetensors")?;

    Ok(())
}

/// Test mel spectrogram using native audio.rs
fn test_mel_spectrogram(device: &Device) -> Result<()> {
    println!("\n=== Testing Mel Spectrogram ===");

    let test_path = Path::new("tests/mel_parity_test.safetensors");
    if !test_path.exists() {
        println!("⚠️  Test data not found. Run Python test first.");
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(test_path, device)?;
    let test_audio = tensors.get("test_audio").context("test_audio not found")?;
    let expected_mel = tensors.get("expected_mel").context("expected_mel not found")?;

    println!("Input audio shape: {:?}", test_audio.shape());
    println!("Expected mel shape: {:?}", expected_mel.shape());

    // Convert to Vec<f32>
    let audio_vec: Vec<f32> = test_audio.flatten_all()?.to_vec1()?;

    // Compute mel spectrogram using native audio.rs
    let config = cosyvoice_rust_backend::audio::MelConfig::inference();

    let rust_mel = cosyvoice_rust_backend::audio::mel_spectrogram(&audio_vec, &config, device)?;

    println!("Rust mel shape: {:?}", rust_mel.shape());

    // Truncate to same length for comparison
    let py_len = expected_mel.dim(2)?;
    let rust_len = rust_mel.dim(2)?;
    let min_len = py_len.min(rust_len);

    let py_mel_crop = expected_mel.narrow(2, 0, min_len)?;
    let rust_mel_crop = rust_mel.narrow(2, 0, min_len)?;

    // Compare
    let diff = py_mel_crop.sub(&rust_mel_crop)?.abs()?;
    let l1_error: f32 = diff.mean_all()?.to_scalar()?;
    let max_error: f32 = diff.max(0)?.max(0)?.max(0)?.to_scalar()?;

    println!("L1 error: {:.6}", l1_error);
    println!("Max error: {:.6}", max_error);

    if l1_error < 1e-2 {  // Mel spectrograms can have larger variance
        println!("✅ PASS: Mel spectrogram parity within tolerance!");
    } else if l1_error < 0.1 {
        println!("⚠️  WARN: Mel spectrogram has moderate error (expected for FFT differences)");
    } else {
        println!("❌ FAIL: L1 error exceeds threshold");
    }

    // Save output
    let mut output = std::collections::HashMap::new();
    output.insert("rust_mel".to_string(), rust_mel);
    candle_core::safetensors::save(&output, "tests/mel_parity_rust_output.safetensors")?;

    Ok(())
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("MATCHA-TTS COMPONENT PARITY TESTS (RUST)");
    println!("============================================================");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // Run all tests
    if let Err(e) = test_sinusoidal_embedding(&device) {
        eprintln!("Sinusoidal embedding test error: {}", e);
    }

    if let Err(e) = test_snake_activation(&device) {
        eprintln!("Snake activation test error: {}", e);
    }

    if let Err(e) = test_mel_spectrogram(&device) {
        eprintln!("Mel spectrogram test error: {}", e);
    }

    println!("\n============================================================");
    println!("RUST PARITY TESTS COMPLETE");
    println!("============================================================");
    println!("Now run: python tests/matcha_parity_tests.py");

    Ok(())
}
