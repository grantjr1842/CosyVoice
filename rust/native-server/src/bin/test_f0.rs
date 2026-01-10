use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice_native_server::hift::{HiFTConfig, HiFTGenerator};
use std::fs;

fn main() -> Result<()> {
    let device = Device::Cpu; // Force CPU for F0 predictor parity

    // 1. Load Config (Use default matching cosyvoice3.yaml)
    println!("Using default HiFTConfig (CosyVoice3)...");
    let hift_config = HiFTConfig::default();

    // 2. Load Weights
    println!("Loading weights...");
    let model_path = "../pretrained_models/Fun-CosyVoice3-0.5B/hift.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    // 3. Build Generator
    println!("Building generator...");
    let generator = HiFTGenerator::new(vb, &hift_config)?;

    // 4. Load Inputs (Mel)
    println!("Loading inputs...");
    let inputs_path = "../output/benchmarks/2026-01-10/hift_components.safetensors";
    let inputs = candle_core::safetensors::load(inputs_path, &device)?;
    let mel = inputs.get("mel").context("missing mel")?; // [1, 80, T]

    println!("Mel shape: {:?}", mel.shape());

    // 5. Test F0 Predictor
    println!("Running F0 Predictor...");
    let f0 = generator.f0_predictor.forward(mel)?;
    println!("Rust F0 shape: {:?}", f0.shape());

    // Compare with Python F0
    let py_f0 = inputs.get("f0").context("missing py_f0")?;
    let py_f0 = py_f0.unsqueeze(1)?;
    println!("Py F0 shape: {:?}", py_f0.shape());

    let diff = (&f0 - py_f0)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;
    println!("F0 Max Diff: {:.6}", max_diff);
    println!("F0 Mean Diff: {:.6}", mean_diff);

    if mean_diff > 1.0 {
        println!("FAILURE: F0 Predictor divergence!");
    } else {
        println!("SUCCESS: F0 Predictor matches!");
    }

    // 6. Run Stages to verify Source and Layers
    println!("\nRunning forward_with_stages...");
    // Mel needs to be loaded again or cloned? inputs["mel"] is available.
    // Move to CPU just in case, though we used CPU device.
    let (audio, stages) = generator.forward_with_stages(&mel)?;

    if let Some(py_audio) = inputs.get("audio") {
        println!("Python Audio found. Comparing...");
        let py_audio = py_audio.squeeze(1)?; // Py [1, T] vs Rust [1, 1, T] or similar?
                                             // Rust audio is [1, 1, T] ?
                                             // HiFTGenerator forward returns [1, 1, T].
        println!(
            "Rust Audio: {:?}, Py Audio: {:?}",
            audio.shape(),
            py_audio.shape()
        );

        let diff = (&audio - &py_audio)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;
        println!(
            "Audio Max Diff: {:.6}, Mean Diff: {:.6}",
            max_diff, mean_diff
        );
    }

    // Check Source
    if let Some(source) = stages.get("source") {
        println!("Source tensor found. Shape: {:?}", source.shape());
        let s_mean = source.mean_all()?.to_scalar::<f32>()?;
        let s_max = source.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("Source Mean: {:.6}, Max Abs: {:.6}", s_mean, s_max);

        // Compare with python source if available?
        if let Some(py_source) = inputs.get("source") {
            println!("Python output source shape: {:?}", py_source.shape());
            // Source is random, so direct comparison fails. But stats should be similar.
            let py_s_mean = py_source.mean_all()?.to_scalar::<f32>()?;
            let py_s_max = py_source.abs()?.max_all()?.to_scalar::<f32>()?;
            println!("Py Source Mean: {:.6}, Max Abs: {:.6}", py_s_mean, py_s_max);
        }
    } else {
        println!("Source tensor NOT found in stages.");
    }

    // Check f0_upsampled
    if let Some(f0_up) = stages.get("f0_upsampled") {
        println!("F0 Upsampled Shape: {:?}", f0_up.shape());
    }

    Ok(())
}
