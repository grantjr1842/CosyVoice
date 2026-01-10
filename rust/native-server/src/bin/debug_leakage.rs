
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use cosyvoice_native_server::audio;
use cosyvoice_native_server::tts::NativeTtsEngine;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use cosyvoice_native_server::text_frontend::text_normalize_english;

const DEFAULT_MODEL_SUBDIR: &str = "pretrained_models/Fun-CosyVoice3-0.5B";
const DEFAULT_PROMPT_TEXT: &str = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that.";
const PROMPT_PREFIX: &str = "Please speak in English.<|endofprompt|>";

#[derive(Parser, Debug)]
#[command(author, version, about = "Debug prompt leakage with optional artifacts override", long_about = None)]
struct Args {
    /// Path to frontend_artifacts.safetensors
    #[arg(long)]
    artifacts_path: Option<String>,
}

fn encode_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer.encode(text, true).map_err(|e| anyhow!("{}", e))?;
    Ok(encoding.get_ids().to_vec())
}

fn latest_benchmark_dir(root: &Path) -> Result<Option<PathBuf>> {
    let bench_dir = root.join("output").join("benchmarks");
    if !bench_dir.exists() {
        return Ok(None);
    }
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&bench_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let meta = entry.metadata()?;
        let modified = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        match &newest {
            Some((best_time, _)) if *best_time >= modified => {}
            _ => newest = Some((modified, path)),
        }
    }
    Ok(newest.map(|(_, p)| p))
}

fn main() -> Result<()> {
    println!("=== Debug Leaked Prompt / Fast Speech ===");
    let args = Args::parse();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let model_dir = repo_root.join(DEFAULT_MODEL_SUBDIR);
    let tokenizer_path = model_dir.join("tokenizer.json");

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{}", e))?;
    let device = Device::new_cuda(0).expect("CUDA device required");

    // Init Engine
    let mut engine = NativeTtsEngine::new(model_dir.to_str().unwrap(), Some(device.clone()))?;

    // Load Parity Artifacts
    let artifacts_path = if let Some(path) = args.artifacts_path {
        PathBuf::from(path)
    } else if let Some(latest_dir) = latest_benchmark_dir(&repo_root)? {
        latest_dir.join("frontend_artifacts.safetensors")
    } else {
        repo_root.join("frontend_artifacts.safetensors")
    };
    if !artifacts_path.exists() {
        return Err(anyhow!(
            "frontend_artifacts.safetensors not found at {}",
            artifacts_path.display()
        ));
    }
    // Load to CPU first to avoid CUDA loader kernel issues, then move to device.
    let tensors = candle_core::safetensors::load(&artifacts_path, &Device::Cpu)?;

    // Get Prompt Tokens (Python)
    let py_speech_tokens = tensors.get("speech_tokens").ok_or(anyhow!("missing speech_tokens"))?
        .to_dtype(candle_core::DType::U32)?
        .to_device(&device)?;

    // Get Speaker Embedding (Python)
    let py_spk_emb = tensors.get("speaker_embedding").ok_or(anyhow!("missing speaker_embedding"))?
        .to_device(&device)?;

    // Prepare Text (Target)
    let tts_text = "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?";
    let full_prompt_text = format!("{}{}", PROMPT_PREFIX, DEFAULT_PROMPT_TEXT);

    // Normalize Prompt
    let prompt_texts = text_normalize_english(&full_prompt_text, &tokenizer, false, true)?;
    let prompt_text = prompt_texts.first().unwrap();
    let prompt_tokens = encode_tokens(&tokenizer, prompt_text)?;

    // Normalize Target
    let target_segments = text_normalize_english(tts_text, &tokenizer, true, true)?;
    let target_text = target_segments.first().unwrap();
    let target_tokens = encode_tokens(&tokenizer, target_text)?;

    // Embed Text
    let mut text_tokens_vec = Vec::new();
    text_tokens_vec.extend_from_slice(&prompt_tokens);
    text_tokens_vec.extend_from_slice(&target_tokens);

    let text_tensor = Tensor::from_vec(text_tokens_vec.clone(), (1, text_tokens_vec.len()), &device)?;
    let text_embeds = engine.llm.embed_text_tokens(&text_tensor)?;

    // Calc Lens
    let tts_text_len = target_tokens.len();
    let min_len = (tts_text_len as f32 * 2.0) as usize;
    let max_len = (tts_text_len as f32 * 20.0) as usize;

    println!("Prompt Len: {}, Target Len: {}", prompt_tokens.len(), target_tokens.len());
    println!("Min Gen: {}, Max Gen: {}", min_len, max_len);

    // Generate with Python Prompt Inputs
    println!("Generating with PYTHON Prompt Speech Tokens...");
    let gen_tokens_vec = engine.llm.generate(
        &text_embeds,
        Some(&py_speech_tokens),
        Some(&py_spk_emb),
        25,
        min_len,
        max_len
    )?;

    println!("Generated {} tokens using Python Inputs.", gen_tokens_vec.len());
    println!("Tokens: {:?}", gen_tokens_vec.iter().take(20).collect::<Vec<_>>());

    // Generate with Rust Prompt Inputs (Calculate locally)
    println!("\nCalculating Rust Prompt Inputs...");
    let wav_path = repo_root.join("asset/interstellar-tars-01-resemble-denoised.wav");
    let (samples, sr) = audio::load_wav(&wav_path)?;
    let samples_16k = audio::resample_audio_cuda(&samples, sr, 16000, &device)?;

    let mel_16k = audio::whisper_log_mel_spectrogram_cuda(&samples_16k)?;
    let fbank = audio::kaldi_fbank_cuda(&samples_16k, 16000)?;
    let (rust_speech_tokens, rust_spk_emb) = engine.process_prompt_tensors(&mel_16k, &fbank)?;

    println!("Generating with RUST Prompt Inputs (Tokens + Emb)...");
    let gen_tokens_rust_vec = engine.llm.generate(
        &text_embeds,
        Some(&rust_speech_tokens),
        Some(&rust_spk_emb),
        25,
        min_len,
        max_len
    )?;

    println!("Generated {} tokens using Rust Inputs.", gen_tokens_rust_vec.len());
    println!("Tokens: {:?}", gen_tokens_rust_vec.iter().take(20).collect::<Vec<_>>());

    Ok(())
}
