use clap::Parser;
use cosyvoice_native_server::tts::NativeTtsEngine;
use cosyvoice_native_server::text_frontend::text_normalize_english;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "../../pretrained_models/Fun-CosyVoice3-0.5B")]
    model_dir: String,

    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog. Performance optimization is key.")]
    text: String,

    /// Path to a text file containing the input text. Overrides --text if provided.
    #[arg(long)]
    text_file: Option<String>,

    #[arg(long, default_value = "../../asset/interstellar-tars-01-resemble-denoised.wav")]
    prompt_wav: String,

    #[arg(long, default_value = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that.")]
    prompt_text: String,

    #[arg(long, default_value_t = 3)]
    iterations: usize,

    /// Path to GGUF model file. If not provided, will check for llm.gguf or default to FP16.
    #[arg(long)]
    gguf: Option<String>,

    /// Force FP16 execution even if GGUF is present (sets COSYVOICE_LLM_GGUF=OFF)
    #[arg(long, default_value_t = false)]
    fp16: bool,

    /// Disable adding the language token <|en|>
    #[arg(long, default_value_t = false)]
    no_lang_token: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();

    // If text_file is provided, read it and override args.text
    if let Some(path) = &args.text_file {
        println!("Reading input text from file: {}", path);
        args.text = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read text file '{}': {}", path, e))?;
        // Trim whitespace just in case
        args.text = args.text.trim().to_string();
    }

    println!("Initializing Native TTS Engine from {}...", args.model_dir);

    // Set environment variables to control GGUF loading
    if args.fp16 {
        println!("Forcing FP16 mode (disabling GGUF)");
        std::env::set_var("COSYVOICE_LLM_GGUF", "OFF");
    } else if let Some(ref g) = args.gguf {
        println!("Using explicit GGUF model: {}", g);
        std::env::set_var("COSYVOICE_LLM_GGUF", g);
    }

    let start_load = Instant::now();
    // NativeTtsEngine::new handles device selection (CUDA if avail) and auto-detects GGUF
    let mut engine = NativeTtsEngine::new(&args.model_dir, None)?;
    let load_time = start_load.elapsed();
    println!("Model loaded in {:.2?}", load_time);

    // Load Tokenizer
    let tokenizer_path = std::path::Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Normalize and tokenize text
    let add_lang_token = !args.no_lang_token;
    println!("Text normalization: add_lang_token={}", add_lang_token);
    let normalized_segments = text_normalize_english(&args.text, &tokenizer, false, true, add_lang_token)?;

    // For benchmarking, we join all segments (though split=false should return 1)
    if normalized_segments.is_empty() {
        return Err("Normalized text is empty".into());
    }
    let full_text = normalized_segments.join("");
    println!("Normalized text: {}", full_text);

    let encoding = tokenizer.encode(full_text, true).map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Tokenized input text ({} chars) into {} tokens.", args.text.len(), token_ids.len());

    // Load prompt audio
    let reader = hound::WavReader::open(&args.prompt_wav)?;
    let _spec = reader.spec();
    let _samples: Vec<f32> = reader.into_samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();

    println!("Preparing inputs...");
    let device = &engine.device;
    let dtype = engine.dtype;

    // Real text tokens [1, text_len]
    let text_tokens = candle_core::Tensor::new(token_ids.as_slice(), device)?.unsqueeze(0)?;

    // Dummy prompt speech 16k [1, 128, 200]
    // Dummy prompt speech 16k [1, 128, 200]
    let prompt_speech_16k = candle_core::Tensor::randn(0f32, 1f32, (1, 128, 200), device)?;

    // Dummy prompt speech 24k [1, 80, 200]
    // Cast to engine dtype (e.g. F16)
    let prompt_speech_24k = candle_core::Tensor::randn(0f32, 1f32, (1, 80, 200), device)?
        .to_dtype(dtype)?;

    // Dummy prompt fbank [1, 200, 80] - Keep as F32 for Frontend compatibility?
    let prompt_fbank = candle_core::Tensor::randn(0f32, 1f32, (1, 200, 80), device)?;

    println!("DEBUG: prompt_speech_16k dtype: {:?}", prompt_speech_16k.dtype());
    println!("DEBUG: prompt_speech_24k dtype: {:?}", prompt_speech_24k.dtype());
    println!("DEBUG: prompt_fbank dtype: {:?}", prompt_fbank.dtype());

    println!("Warming up...");
    // Run one iteration to warm up caches
    let _ = engine.synthesize_instruct(
        &text_tokens,
        &prompt_speech_16k,
        &prompt_speech_24k,
        &prompt_fbank,
        1, // k=1 greedy for deteminism
    ).ok();

    let output_dir = std::path::Path::new("output/benchmark/rust");
    std::fs::create_dir_all(output_dir)?;
    println!("Output directory: {:?}", output_dir);

    println!("Running {} iterations...", args.iterations);
    let mut total_duration = std::time::Duration::new(0, 0);

    for i in 0..args.iterations {
        let start = Instant::now();
        let audio = engine.synthesize_instruct(
            &text_tokens,
            &prompt_speech_16k,
            &prompt_speech_24k,
            &prompt_fbank,
            1,
        )?;
        let duration = start.elapsed();
        total_duration += duration;
        println!("Iteration {}: {:.2?}", i+1, duration);

        // Save audio
        let output_path = output_dir.join(format!("iter_{}.wav", i));
        save_wav(&output_path, &audio, engine.sample_rate)?;
        println!("  Saved: {:?}", output_path);
    }

    let avg_duration = total_duration / args.iterations as u32;
    println!("\nAverage Synthesis Time: {:.2?}", avg_duration);

    Ok(())
}

fn save_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for s in samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}
