use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice_native_server::cosyvoice_llm::{CosyVoiceLLM, CosyVoiceLLMConfig};
use cosyvoice_native_server::qwen::Config as QwenConfig;
use std::fs;
use std::path::Path;

fn print_stats(name: &str, t: &Tensor) -> Result<()> {
    let t = t.to_dtype(DType::F32)?;
    let vec = t.flatten_all()?.to_vec1::<f32>()?;
    let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
    println!(
        "  {}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}",
        name,
        t.shape(),
        min,
        max,
        mean
    );
    if min.is_nan() || max.is_nan() || mean.is_nan() {
        println!("  WARNING: {} contains NaN!", name);
    }
    Ok(())
}

fn compare_tensors(name: &str, rust: &Tensor, python: &Tensor) -> Result<()> {
    let rust = rust.to_dtype(DType::F32)?;
    let python = python.to_dtype(DType::F32)?;

    // Ensure shapes match
    if rust.shape() != python.shape() {
        println!(
            "  {}: Shape mismatch! Rust {:?} vs Python {:?}",
            name,
            rust.shape(),
            python.shape()
        );
    }

    let diff = (rust - python)?.abs()?;
    let max_diff = diff
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .cloned()
        .fold(0.0, f32::max);
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!(
        "  {}: max_diff={:.6}, mean_diff={:.6}",
        name, max_diff, mean_diff
    );

    if max_diff > 1e-4 {
        println!("  WARNING: Large divergence in {}!", name);
    } else {
        println!("  OK: {} matches.", name);
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let model_dir = repo_root.join("pretrained_models/Fun-CosyVoice3-0.5B");
    let debug_data_path = repo_root.join("debug_llm_data.safetensors");

    println!("Loading debug data from {:?}", debug_data_path);
    let tensors = candle_core::safetensors::load(&debug_data_path, &Device::Cpu)?;

    // Load inputs
    let text_embeds = tensors
        .get("text_embeds")
        .context("text_embeds missing")?
        .to_device(&device)?
        .to_dtype(DType::F16)?;
    let prompt_speech_tokens = tensors
        .get("prompt_speech_tokens")
        .context("prompt_speech_tokens missing")?
        .to_device(&device)?
        .to_dtype(DType::U32)?;
    let expected_logits = tensors
        .get("expected_logits")
        .context("expected_logits missing")?
        .to_device(&device)?
        .to_dtype(DType::F16)?;

    // Print stats
    print_stats("input_text_embeds", &text_embeds)?;
    print_stats("input_prompt_tokens", &prompt_speech_tokens)?;

    // Load Model
    println!("Loading CosyVoice LLM...");
    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path)?;
    // The config.json might contain "llm_config" or be the Qwen config itself?
    // Usually config.json in the model dir is the Qwen config.
    let qwen_config: QwenConfig = serde_json::from_str(&config_str)?;

    let llm_config = CosyVoiceLLMConfig::default();

    let llm_path = model_dir.join("llm.safetensors"); // Or just passing the directory? VarBuilder needs a file or list of files.
                                                      // CosyVoiceLLM::new expects VarBuilder.
                                                      // Note: The python script uses "llm.safetensors" usually if separated, or it's inside the main weights.
                                                      // In Fun-CosyVoice3-0.5B, there is likely a `llm.safetensors` or similar.
                                                      // Let's assume `llm.safetensors` exists or we load the whole folder.
                                                      // The previous test `test_decode_stages.rs` loaded `hift.safetensors` directly.
                                                      // `test_flow_parity.rs` loaded `flow.safetensors`.
                                                      // Let's check if `llm.safetensors` exists. If not, maybe it's `model.safetensors`.
                                                      // The Python model loading loaded `pretrained_models/Fun-CosyVoice3-0.5B`.

    let model_file = if model_dir.join("llm.rl.safetensors").exists() {
        println!("Detected RL-tuned model, loading llm.rl.safetensors");
        model_dir.join("llm.rl.safetensors")
    } else {
        println!("Loading standard llm.safetensors");
        model_dir.join("llm.safetensors")
    };
    // If not, we fall back to checking directory. We can use find_by_name tool to check beforehand, but let's assume standard naming or use `model.safetensors` if that's standard for Qwen.
    // Actually, Qwen usually has `model.safetensors`.

    // Let's check file existence in a real scenario, but for now I'll point to `llm.safetensors` or `model.safetensors`.
    // `test_llm_parity.rs` will fail if file missing.

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, &device)? };

    let mut cosy_llm = CosyVoiceLLM::new(&qwen_config, llm_config, vb)?;

    // Run Debug Forward
    // Debug: Compare components
    if let Some(t) = tensors.get("sos_emb") {
        print_stats("py_sos_emb", &t.to_dtype(DType::F32)?)?;
    }
    if let Some(t) = tensors.get("task_id_emb") {
        print_stats("py_task_id_emb", &t.to_dtype(DType::F32)?)?;
    }
    if let Some(t) = tensors.get("prompt_speech_token_emb") {
        print_stats("py_prompt_emb", &t.to_dtype(DType::F32)?)?;
    }

    // Run Debug Forward
    println!("Running debug_forward_one...");
    let (rust_logits, rust_lm_input) =
        cosy_llm.debug_forward_one(&text_embeds, Some(&prompt_speech_tokens))?;

    // Verify lm_input
    // Check if lm_input exists in tensors
    if let Some(py_lm_input) = tensors.get("lm_input") {
        let py_lm_input = py_lm_input.to_device(&device)?.to_dtype(DType::F32)?;
        let rust_lm_input_f32 = rust_lm_input.to_dtype(DType::F32)?;
        print_stats("rust_lm_input", &rust_lm_input_f32)?;
        print_stats("py_lm_input", &py_lm_input)?;
        compare_tensors("LLM LM Input", &rust_lm_input_f32, &py_lm_input)?;
    } else {
        println!("WARNING: 'lm_input' not found in safetensors. Skipping input verification.");
    }

    let rust_logits = rust_logits.to_dtype(DType::F32)?;
    let expected_logits = expected_logits.to_dtype(DType::F32)?; // Already loaded, but ensure F32 for stats

    // Compare
    print_stats("rust_logits", &rust_logits)?;
    print_stats("py_logits", &expected_logits)?;

    compare_tensors("LLM Logits", &rust_logits, &expected_logits)?;

    Ok(())
}
