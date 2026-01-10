
use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use cosyvoice_native_server::audio;
use cosyvoice_native_server::onnx_frontend::OnnxFrontend;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "pretrained_models/Fun-CosyVoice3-0.5B")]
    model_dir: String,

    #[arg(long, default_value = "frontend_artifacts.safetensors")]
    artifacts_path: String,

    #[arg(long, default_value = "asset/interstellar-tars-01-resemble-denoised.wav")]
    prompt_wav: String,
}

fn main() -> Result<()> {
    println!("=== Checking ONNX Frontend Parity ===");

    let args = Args::parse();
    let model_dir = PathBuf::from(&args.model_dir);

    // Initialize Rust Frontend
    let device = Device::new_cuda(0).context("CUDA device required")?;
    let mut frontend = OnnxFrontend::new(model_dir.to_str().unwrap(), device.clone())?;

    // Load Python Artifacts
    let artifacts_path = PathBuf::from(&args.artifacts_path);
    if !artifacts_path.exists() {
        println!(
            "Error: {} not found. Run dump_frontend.py first.",
            artifacts_path.display()
        );
        return Ok(());
    }
    let tensors = candle_core::safetensors::load(&artifacts_path, &Device::Cpu)?;
    let py_speech_tokens = tensors.get("speech_tokens").context("Missing speech_tokens")?;
    let py_spk_emb = tensors.get("speaker_embedding").context("Missing speaker_embedding")?;
    // let py_speech_feat = tensors.get("speech_feat").context("Missing speech_feat")?;

    // Load Audio
    let wav_path = PathBuf::from(&args.prompt_wav);
    let (samples, sr) = audio::load_wav(&wav_path)?;
    println!("Loaded wav: {} samples, sr {}", samples.len(), sr);

    // Resample to 16k for tokenizer (CUDA)
    let samples_16k = audio::resample_audio_cuda(&samples, sr, 16000, &device)?;

    // Log mel for tokenizer
    // Note: Python uses whisper.log_mel_spectrogram(speech, n_mels=128)
    // Rust uses audio::whisper_log_mel_spectrogram
    let mel_16k = audio::whisper_log_mel_spectrogram_cuda(&samples_16k)?;

    // Tokenize
    // Rust returns [1, 1, 128] ? No, [1, 128, T]
    let mel_len = mel_16k.dim(2)?;
    println!("Mel 16k shape: {:?}", mel_16k.shape());

    if let Some(py_mel) = tensors.get("whisper_mel") {
        println!("Comparing Mel Specs...");
        let py_mel = py_mel.to_device(&device)?;
        // Shape check
        println!("Py Mel Shape: {:?}", py_mel.shape());
        // L1 Error
        // mel_16k is [1, 128, T]?? No.
        // Rust audio::whisper_log_mel_spectrogram returns [1, 80, T]? No, whisper uses 80?
        // Wait, Python dump says n_mels=128.
        // Rust utils::whisper_log_mel_spectrogram implementation?
        // Let's check `utils.rs` or `audio.rs`.
        // Assuming parity was intended.

        let py_mel = py_mel.to_dtype(candle_core::DType::F32)?;
        let t1 = mel_16k.flatten_all()?;
        let t2 = py_mel.flatten_all()?;
        let len = usize::min(t1.elem_count(), t2.elem_count());
        let t1 = t1.narrow(0, 0, len)?;
        let t2 = t2.narrow(0, 0, len)?;
        let diff_tensor = (t1 - t2)?;
        let diff = diff_tensor.abs()?.sum_all()?.to_scalar::<f32>()?;
        let count = mel_16k.elem_count();
        println!("Mel L1 Error: {:.6} (avg {:.6})", diff, diff / count as f32);
    }

    let rust_tokens = frontend.tokenize_speech(&mel_16k, mel_len as i32)?;
    println!("Rust Tokens Shape: {:?}", rust_tokens.shape());

    // Compare Tokens
    let rust_vec = rust_tokens.flatten_all()?.to_vec1::<u32>()?;
    let py_vec = py_speech_tokens.flatten_all()?.to_dtype(candle_core::DType::U32)?.to_vec1::<u32>()?;

    let len = usize::min(rust_vec.len(), py_vec.len());
    let mut matches = 0;
    for i in 0..len {
        if rust_vec[i] == py_vec[i] {
            matches += 1;
        }
    }
    let acc = matches as f32 / len as f32;
    println!("Token Accuracy: {:.2}% ({}/{})", acc * 100.0, matches, len);

    if acc < 0.99 {
        println!("Mismatch Details (Context 20):");
        for i in 0..usize::min(20, len) {
            println!("  [{}] Rust: {}, Py: {}", i, rust_vec[i], py_vec[i]);
        }
    }

    // Compare Speaker Embedding
    let fbank = audio::kaldi_fbank_cuda(&samples_16k, 16000)?;
    let rust_emb = frontend.extract_speaker_embedding(&fbank)?;

    // CosSim
    let py_spk_emb = py_spk_emb.to_device(&rust_emb.device())?;
    let sim = (rust_emb.flatten_all()? * py_spk_emb.flatten_all()?)?.sum_all()?.to_scalar::<f32>()?;
    // Note: embeddings might need normalization before dot product for cosine sim if not already normalized.
    // Assuming both are raw or both normalized.
    // Actually, usually we normalize.
    let rust_norm = rust_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let py_norm = py_spk_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let cosine_sim = sim / (rust_norm * py_norm);

    println!("Speaker Embedding Cosine Similarity: {:.6}", cosine_sim);
    println!("Norms: Rust={:.6}, Py={:.6}", rust_norm, py_norm);

    Ok(())
}
