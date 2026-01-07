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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== HiFT Stage Parity Test ===\n");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    // Load HiFT
    let model_dir = "../../pretrained_models/Fun-CosyVoice3-0.5B";
    let hift_path = format!("{}/hift.safetensors", model_dir);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[hift_path], DType::F32, &device)? };
    let config = HiFTConfig::default();
    let hift = HiFTGenerator::new(vb, &config)?;
    println!("HiFT loaded.\n");

    // Load Python stage debug data
    let py_stages = candle_core::safetensors::load("../../hift_stages_debug.safetensors", &Device::Cpu)?;
    println!("Loaded {} Python stages.\n", py_stages.len());

    // Load input mel from debug_artifacts
    let artifacts = candle_core::safetensors::load("../../debug_artifacts.safetensors", &Device::Cpu)?;
    let mel = artifacts.get("python_flow_output").context("python_flow_output")?;
    let mel_dev = mel.to_dtype(DType::F32)?.to_device(&device)?;

    log_stats("input_mel", &mel_dev)?;

    // Run full HiFT and get stages
    println!("\nRunning Rust HiFT with stages...");
    let (rust_audio, rust_stages) = hift.forward_with_stages(&mel_dev)?;

    if let Some(s_stft) = rust_stages.get("s_stft") {
        let s_stft_cpu = s_stft.to_device(&Device::Cpu)?;
        for frame_idx in [8, 100, 112, 1000, 24960] {
            let real = s_stft_cpu.narrow(2, frame_idx, 1)?.narrow(1, 0, 9)?.flatten_all()?.to_vec1::<f32>()?;
            println!("Rust stats for s_stft (Frame {}):", frame_idx);
            println!("  Real: {:?}", real);
        }
    }
    if let Some(si_rust) = rust_stages.get("si_0") {
        if let Some(si_py) = py_stages.get("si_0") {
            let rust_cpu = si_rust.to_device(&Device::Cpu)?;
            let py_cpu = si_py.to_device(&Device::Cpu)?;
            let rust_val = rust_cpu.narrow(1, 255, 1)?.narrow(2, 320, 10)?.flatten_all()?.to_vec1::<f32>()?;
            let py_val = py_cpu.narrow(1, 255, 1)?.narrow(2, 320, 10)?.flatten_all()?.to_vec1::<f32>()?;
            println!("si_0 Channel 255 (near frame 323):");
            println!("  Rust:   {:?}", rust_val);
            println!("  Python: {:?}", py_val);
        }
    }

    // Check filters
    // Analysis STFT is in HiFTGenerator. I'll need to expose it or print it during construction.
    // Wait, I can just re-create a StftModule here with n_fft=16.
    let _debug_stft = cosyvoice_native_server::utils::StftModule::new(16, 4, true, &device)?;
    // No way to get filters from StftModule directly as they are private.
    // I'll add a temporary debug print in utils.rs.

    // List of stages to compare
    let stage_names = vec![
        "input_mel",
        "f0",
        "f0_upsampled",
        "source",
        "s_stft",
        "conv_pre",
        "source_down_out_0",
        "si_res_0_c1_0",
        "si_res_0_c2_0",
        "si_res_0_c1_1",
        "si_res_0_c2_1",
        "si_res_0_c1_2",
        "si_res_0_c2_2",
        "si_0",
        "after_fusion_0",
        "after_resblocks_0",
        "source_down_out_1",
        "si_1",
        "after_fusion_1",
        "after_resblocks_1",
        "source_down_out_2",
        "si_2",
        "after_fusion_2",
        "after_resblocks_2",
        "conv_post",
        "magnitude",
        "phase",
        "pre_clamp_audio",
    ];

    if let (Some(rust), Some(py)) = (rust_stages.get("source_down_out_0"), py_stages.get("source_down_out_0")) {
        let (_b, c, l) = rust.dims3()?;
        let rust_vec = rust.flatten_all()?.to_vec1::<f32>()?;
        let py_vec = py.flatten_all()?.to_vec1::<f32>()?;
        let mut max_diff = 0.0f32;
        let mut max_idx = (0, 0);
        for ch in 0..c {
            for f in 0..l {
                let idx = ch * l + f;
                let diff = (rust_vec[idx] - py_vec[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_idx = (ch, f);
                }
            }
        }
        println!("--- source_down_out_0 max diff localization ---");
        println!("  max_diff={} at channel {}, frame {}", max_diff, max_idx.0, max_idx.1);
        let ch = max_idx.0;
        let f = max_idx.1;
        println!("  values at ch {}, frame range {}-{}:", ch, f.saturating_sub(2), (f + 2).min(l - 1));
        for i in f.saturating_sub(2)..=(f+2).min(l-1) {
            let idx = ch * l + i;
            println!("    frame {}: rust={:.6}, py={:.6}, diff={:.6}", i, rust_vec[idx], py_vec[idx], (rust_vec[idx] - py_vec[idx]).abs());
        }
    }
    if let (Some(rust), Some(py)) = (rust_stages.get("s_stft"), py_stages.get("s_stft")) {
        let (_b, c, l) = rust.dims3()?;
        let rust_vec = rust.flatten_all()?.to_vec1::<f32>()?;
        let py_vec = py.flatten_all()?.to_vec1::<f32>()?;
        let f = 6735;
        println!("--- s_stft check at frame {} ---", f);
        for ch in 0..c.min(2) {
            let idx = ch * l + f;
            println!("    ch {}, frame {}: rust={:.6}, py={:.6}, diff={:.6}", ch, f, rust_vec[idx], py_vec[idx], (rust_vec[idx] - py_vec[idx]).abs());
        }
    }

    println!("\nComparing Intermediate Stages:");
    for name in stage_names {
        if let Some(py_tensor) = py_stages.get(name) {
            if let Some(rust_tensor) = rust_stages.get(name) {

                if name == "si_0" {
                    let rust_cpu = rust_tensor.to_device(&Device::Cpu)?;
                    let py_cpu = py_tensor.to_device(&Device::Cpu)?;
                    let (_, ch_dim, frames) = rust_cpu.dims3()?;
                    let rust_vec = rust_cpu.flatten_all()?.to_vec1::<f32>()?;
                    let py_vec = py_cpu.flatten_all()?.to_vec1::<f32>()?;

                    let mut max_d = 0.0f32;
                    let mut max_ch = 0;
                    let mut max_f = 0;
                    for c in 0..ch_dim {
                        for f in 0..frames {
                            let idx = c * frames + f;
                            let diff = (rust_vec[idx] - py_vec[idx]).abs();
                            if diff > max_d {
                                max_d = diff;
                                max_ch = c;
                                max_f = f;
                            }
                        }
                    }
                    println!("  [si_0] Max diff {} at channel {}, frame {}", max_d, max_ch, max_f);
                }

                if name == "s_stft" {
                    let rust_cpu = rust_tensor.to_device(&Device::Cpu)?;
                    let py_cpu = py_tensor.to_device(&Device::Cpu)?;
                    let rust_vec = rust_cpu.flatten_all()?.to_vec1::<f32>()?;
                    let py_vec = py_cpu.flatten_all()?.to_vec1::<f32>()?;

                    let mut total_ratio = 0.0;
                    let mut count = 0;
                    for i in 0..rust_vec.len() {
                        if py_vec[i].abs() > 1e-4 {
                            total_ratio += rust_vec[i] / py_vec[i];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        println!("  [s_stft] Average ratio (Rust / Py): {}", total_ratio / count as f32);
                    }
                }

                if name == "source" {
                    let rust_cpu = rust_tensor.to_device(&Device::Cpu)?;
                    let rust_vec = rust_cpu.flatten_all()?.to_vec1::<f32>()?;
                    println!("Rust source samples (32-48): {:?}", &rust_vec[32..48]);
                    println!("Rust source samples (440-456): {:?}", &rust_vec[440..456]);
                }

                println!("--- {} ---", name);
                log_stats("rust", rust_tensor)?;
                log_stats("python", py_tensor)?;
                compare_tensors(name, rust_tensor, py_tensor)?;
            } else {
                println!("  ? {}: Missing in Rust", name);
            }
        } else {
            println!("  ? {}: Missing in Python", name);
        }
    }

    // Final Audio comparison (already squeezed)
    if let Some(py_audio) = py_stages.get("final_audio") {
        let py_audio_dev = py_audio.to_device(&device)?;
        println!("");
        compare_tensors("final_audio", &rust_audio, &py_audio_dev)?;
    }

    // Print summary
    println!("\n=== Summary ===");
    println!("Python ISTFT (pre_clamp): min=-2.25, max=1.87");
    println!("Rust applies 0.13x factor before clamp, suggesting raw ISTFT is ~7.5x larger");
    println!("The ISTFT normalization differs between implementations.");

    Ok(())
}
