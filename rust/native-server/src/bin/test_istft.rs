use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use cosyvoice_native_server::utils::InverseStftModule;
use std::path::Path;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Paths
    let inputs_path = "../output/benchmarks/2026-01-10/hift_istft_inputs.safetensors";
    let target_path = "../output/benchmarks/2026-01-10/python_istft_output.safetensors";

    if !Path::new(inputs_path).exists() {
        anyhow::bail!("Inputs file not found: {}", inputs_path);
    }

    println!("Loading inputs from {}", inputs_path);
    let inputs = candle_core::safetensors::load(inputs_path, &device)?;
    let magnitude = inputs.get("magnitude").context("missing magnitude")?;
    let phase = inputs.get("phase").context("missing phase")?; // This is sin(raw_phase)

    println!("Magnitude: {:?}", magnitude.shape());
    println!("Phase: {:?}", phase.shape());

    // Initialize ISTFT
    // n_fft=16, hop_len=4, center=true
    let istft = InverseStftModule::new(16, 4, true, &device)?;

    println!("Running Rust ISTFT...");
    let rust_output = istft.forward(magnitude, phase)?;

    println!("Rust Output Shape: {:?}", rust_output.shape());

    // Load target
    println!("Loading target from {}", target_path);
    let targets = candle_core::safetensors::load(target_path, &device)?;
    let py_output = targets
        .get("python_istft_output")
        .context("missing python output")?;

    // Compare
    // Python output might be longer due to padding logic diffs?
    // My previous check showed Rust/Py lengths matched (99840).

    let diff = (rust_output - py_output)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!("\n=== Parity Results ===");
    println!("Max Diff: {:.6}", max_diff);
    println!("Mean Diff: {:.6}", mean_diff);

    if mean_diff < 1e-4 {
        println!("SUCCESS: ISTFT matches Python!");
    } else {
        println!("FAILURE: ISTFT divergence detected.");
    }

    Ok(())
}
