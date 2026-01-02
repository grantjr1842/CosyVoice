//! Native Rust equivalent of example.py (zero-shot voice cloning demo).

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use cosyvoice_native_server::audio::{self, MelConfig};
use cosyvoice_native_server::text_frontend::text_normalize_english;
use cosyvoice_native_server::tts::NativeTtsEngine;
use hound::WavWriter;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const DEFAULT_MODEL_SUBDIR: &str = "pretrained_models/Fun-CosyVoice3-0.5B";
const DEFAULT_PROMPT_WAV_REL: &str = "asset/interstellar-tars-01-resemble-denoised.wav";
const DEFAULT_PROMPT_TEXT: &str = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that.";
const PROMPT_PREFIX: &str = "Please speak in English.<|endofprompt|>";

fn encode_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenizer encode failed: {}", e))?;
    Ok(encoding.get_ids().iter().map(|&id| id as u32).collect())
}

fn save_wav(path: &Path, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for s in samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Native CosyVoice3 Voice Cloning Example ===");

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let model_dir = std::env::var("COSYVOICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join(DEFAULT_MODEL_SUBDIR));
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(anyhow!(
            "Tokenizer not found at {:?}. Run: pixi run python -c \"from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer; import os; md='{}'; tp=os.path.join(md,'CosyVoice-BlankEN'); tok=CosyVoice3Tokenizer(token_path=tp, skip_special_tokens=True); tok.tokenizer.save_pretrained(md)\"",
            tokenizer_path,
            model_dir.to_string_lossy()
        ));
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    let mut engine = NativeTtsEngine::new(model_dir.to_string_lossy().as_ref(), Some(device.clone()))?;
    if engine.frontend.is_none() {
        return Err(anyhow!("ONNX frontend failed to initialize; cannot run native example."));
    }

    let prompt_wav = repo_root.join(DEFAULT_PROMPT_WAV_REL);
    println!("Loading prompt audio: {}", prompt_wav.display());
    let (prompt_samples, prompt_sr) = audio::load_wav(&prompt_wav)
        .with_context(|| format!("Failed to load prompt wav: {}", prompt_wav.display()))?;

    if prompt_sr < 16000 {
        return Err(anyhow!("Prompt sample rate too low: {}", prompt_sr));
    }

    let prompt_16k = audio::resample_audio(&prompt_samples, prompt_sr, 16000)?;
    let prompt_24k = audio::resample_audio(&prompt_samples, prompt_sr, 24000)?;

    let prompt_speech_16k = audio::whisper_log_mel_spectrogram(&prompt_16k, &Device::Cpu)?;
    let prompt_fbank = audio::kaldi_fbank(&prompt_16k, 16000, &Device::Cpu)?;
    let mut prompt_speech_24k = audio::mel_spectrogram(&prompt_24k, &MelConfig::cosyvoice3(), &device)?;

    let (mut prompt_speech_tokens, speaker_embedding) = engine.process_prompt_tensors(&prompt_speech_16k, &prompt_fbank)?;

    // Align prompt mel length with prompt token length (token_mel_ratio = 2)
    let mel_len = prompt_speech_24k.dim(2)?;
    let token_len = prompt_speech_tokens.dim(1)?;
    let aligned_token_len = usize::min(mel_len / 2, token_len);
    if aligned_token_len == 0 {
        return Err(anyhow!("Prompt token length is zero after alignment"));
    }
    prompt_speech_24k = prompt_speech_24k.narrow(2, 0, aligned_token_len * 2)?;
    prompt_speech_tokens = prompt_speech_tokens.narrow(1, 0, aligned_token_len)?;

    let texts = [
        "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    ];

    let output_dir = repo_root.join("output");
    fs::create_dir_all(&output_dir)?;

    let full_prompt_text = format!("{}{}", PROMPT_PREFIX, DEFAULT_PROMPT_TEXT);
    let prompt_texts = text_normalize_english(&full_prompt_text, &tokenizer, false, true)?;
    let prompt_text = prompt_texts
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Prompt text normalization produced no output"))?;
    let prompt_tokens = encode_tokens(&tokenizer, &prompt_text)?;

    for (idx, tts_text) in texts.iter().enumerate() {
        let segments = text_normalize_english(tts_text, &tokenizer, true, true)?;
        if segments.is_empty() {
            println!("\nSkipping empty text segment for input [{}/{}]", idx + 1, texts.len());
            continue;
        }

        for (seg_idx, segment) in segments.iter().enumerate() {
            println!(
                "\nSynthesizing [{}/{}] segment [{}/{}]...",
                idx + 1,
                texts.len(),
                seg_idx + 1,
                segments.len()
            );
            let tts_tokens = encode_tokens(&tokenizer, segment)?;

            let mut text_tokens = Vec::with_capacity(prompt_tokens.len() + tts_tokens.len());
            text_tokens.extend_from_slice(&prompt_tokens);
            text_tokens.extend_from_slice(&tts_tokens);

            let text_tensor = Tensor::from_vec(
                text_tokens,
                (1, prompt_tokens.len() + tts_tokens.len()),
                &device,
            )?;
            let text_embeds = engine.llm.embed_text_tokens(&text_tensor)?;
            let audio_samples = engine.synthesize_full_with_prompt_len(
                &text_embeds,
                prompt_tokens.len(),
                Some(&prompt_speech_tokens),
                &prompt_speech_24k,
                &speaker_embedding,
                25,
            )?;

            let output_path = output_dir.join(format!("native_voice_clone_{}_{}.wav", idx, seg_idx));
            save_wav(&output_path, &audio_samples, engine.sample_rate)?;
            let duration = audio_samples.len() as f32 / engine.sample_rate as f32;
            println!("Saved: {:?} (duration {:.2}s)", output_path, duration);
        }
    }

    println!("\nNative voice cloning complete!");
    Ok(())
}
