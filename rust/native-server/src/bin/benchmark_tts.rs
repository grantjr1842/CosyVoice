use clap::Parser;
use cosyvoice_native_server::tts::NativeTtsEngine;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "pretrained_models/Fun-CosyVoice3-0.5B")]
    model_dir: String,

    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog. Performance optimization is key.")]
    text: String,

    #[arg(long, default_value = "asset/interstellar-tars-01-resemble-denoised.wav")]
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

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

    // Load prompt audio
    let reader = hound::WavReader::open(&args.prompt_wav)?;
    let _spec = reader.spec();
    let _samples: Vec<f32> = reader.into_samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();

    println!("Preparing inputs (using random/dummy data for benchmark consistency)...");
    let device = &engine.device;
    let dtype = engine.dtype;

    // Dummy text tokens [1, 50]
    let text_tokens = candle_core::Tensor::zeros((1, 50), candle_core::DType::U32, device)?;

    // Dummy prompt speech 16k [1, 128, 200]
    // Keep as F32 for Frontend compatibility (ONNX usually expects F32)
    let prompt_speech_16k = candle_core::Tensor::randn(0f32, 1f32, (1, 128, 200), device)?;

    // Dummy prompt speech 24k [1, 80, 200]
    // Must match engine dtype (F16 on CUDA) -> used by Flow/LLM
    let prompt_speech_24k = candle_core::Tensor::randn(0f32, 1f32, (1, 80, 200), device)?
        .to_dtype(dtype)?;

    // Dummy prompt fbank [1, 200, 80] - Keep as F32 for Frontend compatibility
    let prompt_fbank = candle_core::Tensor::randn(0f32, 1f32, (1, 200, 80), device)?;

    // println!("DEBUG: prompt_speech_16k dtype: {:?}", prompt_speech_16k.dtype());
    // println!("DEBUG: prompt_speech_24k dtype: {:?}", prompt_speech_24k.dtype());
    // println!("DEBUG: prompt_fbank dtype: {:?}", prompt_fbank.dtype());

    println!("Warming up...");
    // Run one iteration to warm up caches
    let _ = engine.synthesize_instruct(
        &text_tokens,
        &prompt_speech_16k,
        &prompt_speech_24k,
        &prompt_fbank,
        1, // k=1 greedy for deteminism
    ).ok();

    println!("Running {} iterations...", args.iterations);
    let mut total_duration = std::time::Duration::new(0, 0);

    for i in 0..args.iterations {
        let start = Instant::now();
        let _audio = engine.synthesize_instruct(
            &text_tokens,
            &prompt_speech_16k,
            &prompt_speech_24k,
            &prompt_fbank,
            1,
        )?;
        let duration = start.elapsed();
        total_duration += duration;
        println!("Iteration {}: {:.2?} (Note: Input was dummy, so output is garbage but perf is valid)", i+1, duration);
    }

    let avg_duration = total_duration / args.iterations as u32;
    println!("\nAverage Synthesis Time: {:.2?}", avg_duration);

    Ok(())
}
