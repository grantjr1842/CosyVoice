use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice_native_server::cosyvoice_llm::{CosyVoiceLLM, CosyVoiceLLMConfig};
use cosyvoice_native_server::qwen::Config as QwenConfig;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct PyOutput {
    text_len: usize,
    min_len: usize,
    max_len: usize,
    token_len: usize,
}

fn run_python(
    text: &str,
    model_dir: &str,
    tokenizer_path: &str,
    use_rl: bool,
    sampling_k: usize,
    min_ratio: f32,
    max_ratio: f32,
) -> Result<PyOutput> {
    let mut child = Command::new("pixi")
        .args(["run", "python", "tools/llm_length_parity.py"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn pixi python")?;

    let payload = serde_json::json!({
        "text": text,
        "model_dir": model_dir,
        "tokenizer_path": tokenizer_path,
        "sampling_k": sampling_k,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "use_rl": use_rl,
        "greedy": true,
    });

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(payload.to_string().as_bytes())
            .context("Failed to write payload to python stdin")?;
    }

    let output = child
        .wait_with_output()
        .context("Failed to read python output")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("Python parity script failed: {stderr}"));
    }

    serde_json::from_slice(&output.stdout).context("Failed to parse python output")
}

fn load_llm(model_dir: &Path, device: &Device, use_rl: bool) -> Result<CosyVoiceLLM> {
    let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
    let qwen_config: QwenConfig = serde_json::from_str(&config_str)?;

    let llm_path = if use_rl {
        let rl_path = model_dir.join("llm.rl.safetensors");
        if rl_path.exists() {
            rl_path
        } else {
            model_dir.join("llm.safetensors")
        }
    } else {
        model_dir.join("llm.safetensors")
    };

    let llm_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&llm_path], DType::F32, device)
            .map_err(|e| anyhow!("Failed to load LLM: {e}"))?
    };

    let llm_config = CosyVoiceLLMConfig {
        sampling_top_p: 1.0,
        ras_window_size: 0,
        ras_tau_r: 0.0,
        ..CosyVoiceLLMConfig::default()
    };

    CosyVoiceLLM::new(&qwen_config, llm_config, llm_vb)
        .map_err(|e| anyhow!("Failed to initialize LLM: {e}"))
}

#[test]
fn llm_length_parity() -> Result<()> {
    if std::env::var("COSYVOICE_PY_PARITY").is_err() {
        eprintln!("Skipping parity test (set COSYVOICE_PY_PARITY=1 to enable).");
        return Ok(());
    }

    let model_dir = std::env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| "pretrained_models/Fun-CosyVoice3-0.5B".to_string());
    let model_path = PathBuf::from(&model_dir);
    if !model_path.exists() {
        eprintln!(
            "Skipping parity test (model_dir not found at {}).",
            model_dir
        );
        return Ok(());
    }

    let tokenizer_path = std::env::var("COSYVOICE_TOKENIZER_PATH").unwrap_or_else(|_| {
        model_path
            .join("tokenizer.json")
            .to_string_lossy()
            .to_string()
    });
    if !Path::new(&tokenizer_path).exists() {
        eprintln!(
            "Skipping parity test (tokenizer not found at {}).",
            tokenizer_path
        );
        return Ok(());
    }

    let text = "Hello! This is a quick parity check.";
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenizer encode failed: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let min_ratio = 2.0_f32;
    let max_ratio = 20.0_f32;
    let min_len = (token_ids.len() as f32 * min_ratio) as usize;
    let max_len = (token_ids.len() as f32 * max_ratio) as usize;

    let use_rl = std::env::var("COSYVOICE_USE_RL")
        .map(|v| v != "0")
        .unwrap_or(true);

    let device = Device::Cpu;
    let mut llm = load_llm(&model_path, &device, use_rl)?;
    let text_tensor = Tensor::from_vec(token_ids, (1, encoding.get_ids().len()), &device)?;
    let text_embeds = llm.embed_text_tokens(&text_tensor)?;

    let rust_tokens = llm.generate(&text_embeds, None, None, 1, min_len, max_len)?;
    let rust_len = rust_tokens.len();

    let py = run_python(
        text,
        &model_dir,
        &tokenizer_path,
        use_rl,
        1,
        min_ratio,
        max_ratio,
    )?;

    assert_eq!(py.text_len, encoding.get_ids().len());
    assert_eq!(py.min_len, min_len);
    assert_eq!(py.max_len, max_len);
    assert_eq!(py.token_len, rust_len);

    Ok(())
}
