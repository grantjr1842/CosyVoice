use anyhow::{Result, Context};
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use clap::Parser;
use cosyvoice_native_server::cosyvoice_flow::{CosyVoiceFlow, CosyVoiceFlowConfig};
use cosyvoice_native_server::flow::FlowConfig;
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

    // Choose device
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);
    // Force F32 for parity verification to avoid F16 instability
    let dtype = DType::F32; // if device.is_cuda() { DType::F16 } else { DType::F32 };

    // Load Flow Model manually to avoid LLM issues
    println!("Loading Flow model from {}...", args.model_dir);
    let flow_path = PathBuf::from(&args.model_dir).join("flow.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[flow_path], dtype, &device)? };

    let flow_config = CosyVoiceFlowConfig::default();
    let dit_config = FlowConfig::default();

    let flow = CosyVoiceFlow::new(flow_config, &dit_config, vb)?;

    // Load Artifacts
    println!("Loading artifacts from {}...", args.artifacts_path);
    // Artifacts are typically F32 (or I32). Load to CPU first to avoid missing kernels for casting.
    let artifacts = candle_core::safetensors::load(&args.artifacts_path, &Device::Cpu)?;

    let token = artifacts.get("token").context("token not found")?;
    let prompt_token = artifacts.get("prompt_token").context("prompt_token not found")?;
    let prompt_feat = artifacts.get("prompt_feat").context("prompt_feat not found")?;
    let embedding = artifacts.get("embedding").context("embedding not found")?;
    let expected_mel = artifacts.get("python_flow_output").context("python_flow_output not found")?;
    let rand_noise = artifacts.get("rand_noise").context("rand_noise not found")?;

    println!("  token: {:?}", token.shape());
    println!("  prompt_token: {:?}", prompt_token.shape());
    println!("  prompt_feat: {:?}", prompt_feat.shape());
    println!("  embedding: {:?}", embedding.shape());
    println!("  rand_noise: {:?}", rand_noise.shape());

    // Cast inputs to dtype if needed (embedding, prompt_feat, rand_noise are float)
    // token outputs are int/long usually?
    // token: Long/Int? check python.
    // Actually Rust `CosyVoiceFlow::inference` expects Tensor.
    // `input_embedding` expects Int/Long indices.
    // `token` and `prompt_token` should be DType::U32 or I64?
    // Artifact loading loads as is. Python saved Long as I64 usually.
    // Candle Embedding layer expects U32.
    // We should cast tokens.

    // Check loaded dtype
    println!("  token dtype: {:?}", token.dtype());

    println!("Casting token to U32 (via F32 workaround)...");
    let token = if token.dtype() != DType::U32 {
        token.to_dtype(DType::F32)?.to_dtype(DType::U32)?
    } else {
        token.clone()
    };
    println!("Casting prompt_token to U32 (via F32 workaround)...");
    let prompt_token = if prompt_token.dtype() != DType::U32 {
        prompt_token.to_dtype(DType::F32)?.to_dtype(DType::U32)?
    } else {
        prompt_token.clone()
    };

    // Feature tensors to model dtype
    // prompt_feat from python is [B, T, D], inference expects [B, D, T]
    // Feature tensors to model dtype (on CPU)
    println!("Processing prompt_feat (cast + transpose on CPU)...");
    let prompt_feat = prompt_feat.to_dtype(dtype)?.transpose(1, 2)?;
    println!("Processing embedding...");
    let embedding = embedding.to_dtype(dtype)?;

    println!("Moving inputs to device...");
    let token = token.to_device(&device)?;
    let prompt_token = prompt_token.to_device(&device)?;
    let prompt_feat = prompt_feat.to_device(&device)?;
    let embedding = embedding.to_device(&device)?;

    println!("Running Flow Inference...");

    let rand_noise_cast = rand_noise.to_dtype(dtype)?.to_device(&device)?;

    let generated_mel = flow.inference(
        &token,
        &prompt_token,
        &prompt_feat,
        &embedding,
        1, // n_timesteps - Python uses 1 in CausalMaskedDiffWithXvec.inference()
        Some(&rand_noise_cast)
    )?;

    println!("  generated_mel: {:?}", generated_mel.shape());
    println!("  expected_mel: {:?}", expected_mel.shape());

    // Compare
    // Cast generated back to F32 for comparison with expected (which is likely F32 from python cpu save)
    // Move generated to CPU first
    let generated_cpu = generated_mel.to_device(&Device::Cpu)?;
    let generated_f32 = generated_cpu.to_dtype(DType::F32)?;
    let expected_f32 = expected_mel.to_dtype(DType::F32)?; // Should ideally already be F32

    // Check for NaNs via vector
    let generated_vec = generated_f32.flatten_all()?.to_vec1::<f32>()?;
    let nan_count = generated_vec.iter().filter(|x| x.is_nan()).count();

    if nan_count > 0 {
        println!("FAIL: Output contains {} NaNs", nan_count);
        // Save failure
        let debug_save = std::collections::HashMap::from([
            ("generated".to_string(), generated_f32),
            ("expected".to_string(), expected_f32),
        ]);
        candle_core::safetensors::save(&debug_save, "flow_nan_debug.safetensors")?;
        return Ok(());
    }

    let diff = (generated_f32 - expected_f32)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!("Max Diff: {:.6}", max_diff);
    println!("Mean Diff: {:.6}", mean_diff);

    if max_diff > 1e-2 { // F32 can have ~0.01 diff across different implementations
        println!("FAIL: Max diff {:.6} > 1e-2", max_diff);
        let debug_save = std::collections::HashMap::from([
            ("diff".to_string(), diff),
            ("generated".to_string(), generated_mel.to_dtype(DType::F32)?), // save as F32
            ("expected".to_string(), expected_mel.clone()),
        ]);
        candle_core::safetensors::save(&debug_save, "flow_failure_debug.safetensors")?;
        println!("Saved failure debug tensors to flow_failure_debug.safetensors");
    } else {
        println!("SUCCESS: Flow output matches Python reference");
    }

    Ok(())
}
