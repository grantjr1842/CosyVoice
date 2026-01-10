use anyhow::{Result, Context};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== Rust Weight Loading Verification ===\n");

    let device = Device::Cpu;  // Use CPU for deterministic comparison
    let dtype = DType::F32;

    let hift_path = PathBuf::from("../../pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[hift_path], dtype, &device)? };

    // conv_pre - test weight_norm reconstruction
    let vb_conv_pre = vb.pp("conv_pre");

    println!("conv_pre weight loading test:");

    // Load raw parametrizations
    let g = vb_conv_pre.get((512, 1, 1), "parametrizations.weight.original0")?;
    let v = vb_conv_pre.get((512, 80, 5), "parametrizations.weight.original1")?;

    let g_vec = g.flatten_all()?.to_vec1::<f32>()?;
    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;

    println!("  g: min={:.4}, max={:.4}",
        g_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        g_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("  v: min={:.4}, max={:.4}",
        v_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        v_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Reconstruct weight_norm: weight = g * v / ||v||
    // ||v|| is norm over dims (1, 2) for each output channel
    let norm_v = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let scale = g.broadcast_div(&norm_v)?;
    let weight = v.broadcast_mul(&scale)?;

    let weight_vec = weight.flatten_all()?.to_vec1::<f32>()?;
    let w_min = weight_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let w_max = weight_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w_mean: f32 = weight_vec.iter().sum::<f32>() / weight_vec.len() as f32;

    println!("  Reconstructed weight: min={:.4}, max={:.4}, mean={:.6}", w_min, w_max, w_mean);
    println!("  (Python reference:    min=-0.5683, max=0.7964, mean=0.000073)");

    // Check if they match
    let py_min = -0.5683;
    let py_max = 0.7964;
    let py_mean = 0.000073;

    let err_min = (w_min - py_min).abs();
    let err_max = (w_max - py_max).abs();
    let err_mean = (w_mean - py_mean).abs();

    println!("\n  Errors: min={:.6}, max={:.6}, mean={:.8}", err_min, err_max, err_mean);

    if err_min < 0.001 && err_max < 0.001 && err_mean < 0.0001 {
        println!("  ✓ Weight reconstruction MATCHES Python!");
    } else {
        println!("  ✗ Weight reconstruction DIFFERS from Python!");
    }

    // Check bias loading
    let bias = vb_conv_pre.get(512, "bias")?;
    let bias_vec = bias.flatten_all()?.to_vec1::<f32>()?;
    let b_min = bias_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let b_max = bias_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let b_mean: f32 = bias_vec.iter().sum::<f32>() / bias_vec.len() as f32;

    println!("\n  conv_pre bias: min={:.4}, max={:.4}, mean={:.6}", b_min, b_max, b_mean);
    println!("  (Python reference: min=-0.8973, max=0.9964, mean=0.122229)");

    Ok(())
}
