use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use cosyvoice_native_server::text_frontend::text_normalize_english;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct PyOutput {
    segments: Vec<String>,
    token_lengths: Vec<usize>,
}

fn run_python(text: &str, tokenizer_path: &str, split: bool) -> Result<PyOutput> {
    let mut child = Command::new("pixi")
        .args(["run", "python", "../../tools/text_normalize_parity.py"])
        .env("PYTHONPATH", "../../")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn pixi python")?;

    let payload = serde_json::json!({
        "text": text,
        "tokenizer_path": tokenizer_path,
        "split": split,
        "text_frontend": true,
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

fn rust_token_lengths(tokenizer: &Tokenizer, segments: &[String]) -> Result<Vec<usize>> {
    segments
        .iter()
        .map(|segment| {
            let encoding = tokenizer
                .encode(segment.as_str(), true)
                .map_err(|e| anyhow!("Tokenizer encode failed: {e}"))?;
            Ok(encoding.get_ids().len())
        })
        .collect()
}

fn assert_parity(
    text: &str,
    tokenizer: &Tokenizer,
    tokenizer_path: &str,
    split: bool,
) -> Result<()> {
    let rust_segments = text_normalize_english(text, tokenizer, split, true)?;
    let py_output = run_python(text, tokenizer_path, split)?;

    assert_eq!(rust_segments, py_output.segments);

    let rust_lengths = rust_token_lengths(tokenizer, &rust_segments)?;
    assert_eq!(rust_lengths, py_output.token_lengths);

    Ok(())
}

#[test]
fn english_text_normalize_parity() -> Result<()> {
    if std::env::var("COSYVOICE_PY_PARITY").is_err() {
        eprintln!("Skipping parity test (set COSYVOICE_PY_PARITY=1 to enable).");
        return Ok(());
    }

    let tokenizer_path = std::env::var("COSYVOICE_TOKENIZER_PATH")
        .unwrap_or_else(|_| "pretrained_models/Fun-CosyVoice3-0.5B/tokenizer.json".to_string());
    if !Path::new(&tokenizer_path).exists() {
        eprintln!(
            "Skipping parity test (tokenizer not found at {}).",
            tokenizer_path
        );
        return Ok(());
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?;

    let text = "Hello! I have 2 dogs. This is a longer sentence to exercise splitting. ".repeat(12);
    assert_parity(&text, &tokenizer, &tokenizer_path, true)?;

    let prompt_text = "Please speak in English.<|endofprompt|>Testing special tokens.";
    assert_parity(prompt_text, &tokenizer, &tokenizer_path, false)?;

    Ok(())
}
