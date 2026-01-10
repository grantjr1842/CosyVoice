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

fn encode_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenizer encode failed: {}", e))?;
    Ok(encoding.get_ids().to_vec())
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

    let mut engine =
        NativeTtsEngine::new(model_dir.to_string_lossy().as_ref(), Some(device.clone()))?;
    if engine.frontend.is_none() {
        return Err(anyhow!(
            "ONNX frontend failed to initialize; cannot run native example."
        ));
    }

    let prompt_wav = repo_root.join(DEFAULT_PROMPT_WAV_REL);
    println!("Loading prompt audio: {}", prompt_wav.display());
    let (prompt_samples, prompt_sr) = audio::load_wav(&prompt_wav)
        .with_context(|| format!("Failed to load prompt wav: {}", prompt_wav.display()))?;

    if prompt_sr < 16000 {
        return Err(anyhow!("Prompt sample rate too low: {}", prompt_sr));
    }

    let prompt_16k = audio::resample_audio_cuda(&prompt_samples, prompt_sr, 16000, &device)?;
    let prompt_24k = audio::resample_audio_cuda(&prompt_samples, prompt_sr, 24000, &device)?;

    let prompt_speech_16k = audio::whisper_log_mel_spectrogram_cuda(&prompt_16k)?;
    let prompt_fbank = audio::kaldi_fbank_cuda(&prompt_16k, 16000)?;
    let mut prompt_speech_24k = audio::mel_spectrogram_cuda(&prompt_24k, &MelConfig::cosyvoice3())?;

    let (mut prompt_speech_tokens, mut speaker_embedding) =
        engine.process_prompt_tensors(&prompt_speech_16k, &prompt_fbank)?;

    // Align prompt mel length with prompt token length (token_mel_ratio = 2)
    let mel_len = prompt_speech_24k.dim(2)?;
    let token_len = prompt_speech_tokens.dim(1)?;
    let aligned_token_len = usize::min(mel_len / 2, token_len);
    println!(
        "DEBUG: mel_len={}, token_len={}, aligned_token_len={}",
        mel_len, token_len, aligned_token_len
    );
    if aligned_token_len == 0 {
        return Err(anyhow!("Prompt token length is zero after alignment"));
    }
    prompt_speech_24k = prompt_speech_24k.narrow(2, 0, aligned_token_len * 2)?;
    prompt_speech_tokens = prompt_speech_tokens.narrow(1, 0, aligned_token_len)?;

    let use_debug_artifacts =
        std::env::var("COSYVOICE_USE_DEBUG_ARTIFACTS").map(|v| v != "0").unwrap_or(false);
    let force_tts_tokens =
        std::env::var("COSYVOICE_FORCE_TTS_TOKENS").map(|v| v != "0").unwrap_or(false);
    let debug_artifacts_path = Path::new("debug_artifacts.safetensors");
    let debug_artifacts = if use_debug_artifacts && debug_artifacts_path.exists() {
        println!("\n*** LOADING DEBUG ARTIFACTS FROM PYTHON ***");
        Some(candle_core::safetensors::load(
            debug_artifacts_path,
            &Device::Cpu,
        )?)
    } else {
        None
    };

    if let Some(tensors) = debug_artifacts.as_ref() {
        if let Some(py_st) = tensors.get("python_speech_tokens") {
            println!(
                "Replacing prompt_speech_tokens: {:?} -> {:?}",
                prompt_speech_tokens.shape(),
                py_st.shape()
            );
            // Cast on CPU then move to device
            prompt_speech_tokens = py_st
                .to_dtype(candle_core::DType::U32)?
                .to_device(&device)?;
        }
        if let Some(py_se) = tensors.get("python_spk_emb") {
            println!(
                "Replacing speaker_embedding: {:?} -> {:?}",
                speaker_embedding.shape(),
                py_se.shape()
            );
            speaker_embedding = py_se.to_device(&device)?;
        }
        if let Some(py_mel) = tensors.get("python_mel_24k") {
            println!(
                "Replacing prompt_speech_24k: {:?} -> {:?}",
                prompt_speech_24k.shape(),
                py_mel.shape()
            );
            prompt_speech_24k = py_mel.to_device(&device)?;
        }
    }

    let texts = [
        "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    ];

    let output_dir = repo_root.join("output");
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {:?}", output_dir);

    // Process prompt text separately (Python-style: split=false)
    let prompt_texts = text_normalize_english(DEFAULT_PROMPT_TEXT, &tokenizer, false, true)?;
    let prompt_text = prompt_texts
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Prompt text normalization produced no output"))?;
    let prompt_tokens = encode_tokens(&tokenizer, &prompt_text)?;

    for (idx, tts_text) in texts.iter().enumerate() {
        let segments = text_normalize_english(tts_text, &tokenizer, true, true)?;
        if segments.is_empty() {
            println!(
                "\nSkipping empty text segment for input [{}/{}]",
                idx + 1,
                texts.len()
            );
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

            // Create the model input like Python does:
            // - prompt_text goes to the prompt field
            // - text goes to the text field
            // They are NOT concatenated for the LLM

            println!(
                "DEBUG: Prompt tokens: {:?}",
                prompt_tokens.iter().take(20).collect::<Vec<_>>()
            );
            println!(
                "DEBUG: TTS tokens: {:?}",
                tts_tokens.iter().take(20).collect::<Vec<_>>()
            );

            // For the LLM, we need to combine prompt + text like Python's zero_shot mode
            let mut text_tokens = Vec::with_capacity(prompt_tokens.len() + tts_tokens.len());
            text_tokens.extend_from_slice(&prompt_tokens);
            text_tokens.extend_from_slice(&tts_tokens);

            let text_tensor =
                Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &device)?;
            println!(
                "DEBUG: Input text tokens (first 40): {:?}",
                text_tokens.iter().take(40).collect::<Vec<_>>()
            );
            let text_embeds = engine.llm.embed_text_tokens(&text_tensor)?;

            // Optional: force using precomputed speech tokens from debug artifacts.
            let forced_speech_tokens = if force_tts_tokens {
                if let Some(tensors) = debug_artifacts.as_ref() {
                    if let Some(py_st) = tensors.get("speech_tokens") {
                        println!("=== DEBUG: USING FORCED PYTHON SPEECH TOKENS FROM ARTIFACT ===");
                        Some(
                            py_st
                                .to_dtype(candle_core::DType::U32)?
                                .to_device(&device)?,
                        )
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let (speech_tokens, audio_samples) = if let Some(speech_tokens) = forced_speech_tokens {
                // Skip LLM, run Flow + HiFT directly
                println!(
                    "Skipping LLM generation, using {} forced tokens.",
                    speech_tokens.dim(1)?
                );

                let prompt_mel = &prompt_speech_24k;
                let flow_embed = &speaker_embedding;

                let audio = engine.synthesize_flow_hift(
                    &speech_tokens,
                    &prompt_speech_tokens,
                    prompt_mel,
                    flow_embed,
                )?;
                (speech_tokens, audio)
            } else {
                let tts_text_len = tts_tokens.len();
                let min_len = (tts_text_len as f32 * 2.0) as usize;
                let max_len = (tts_text_len as f32 * 20.0) as usize;

                let speech_tokens_vec = engine.llm.generate(
                    &text_embeds,
                    Some(&prompt_speech_tokens),
                    Some(&speaker_embedding),
                    25,
                    min_len,
                    max_len,
                )?;
                // Convert Vec<u32> to Tensor [1, N]
                let speech_tokens = Tensor::from_vec(
                    speech_tokens_vec.clone(),
                    (1, speech_tokens_vec.len()),
                    &device,
                )?;
                println!(
                    "Generated {} speech tokens: {:?}",
                    speech_tokens.dim(1)?,
                    speech_tokens_vec.iter().take(20).collect::<Vec<_>>()
                );

                let audio = engine.synthesize_from_tokens(
                    &speech_tokens,
                    &prompt_speech_tokens,
                    &prompt_speech_24k,
                    &speaker_embedding,
                    None,
                )?;
                (speech_tokens, audio)
            };
            println!(
                "DEBUG: Generated audio duration: {:.2}s",
                audio_samples.len() as f32 / engine.sample_rate as f32
            );

            let output_path =
                output_dir.join(format!("native_voice_clone_{}_{}.wav", idx, seg_idx));
            save_wav(&output_path, &audio_samples, engine.sample_rate)?;
            let duration = audio_samples.len() as f32 / engine.sample_rate as f32;
            println!("Saved: {:?} (duration {:.2}s)", output_path, duration);
        }
    }

    println!("\nNative voice cloning complete!");
    Ok(())
}
