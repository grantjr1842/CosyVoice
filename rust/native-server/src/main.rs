//! CosyVoice Native Rust Server
//!
//! High-performance text-to-speech server using native Rust implementation (Candle + Ort).

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use metrics::{counter, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::{
    env,
    net::SocketAddr,
    sync::Arc,
    time::Instant,
    path::PathBuf,
};
use tokio::signal;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use candle_core::{Device, Tensor, DType};

use shared::{config, ErrorResponse, HealthResponse, SynthesizeRequest};

use cosyvoice_native_server::tts::{NativeTtsEngine, NativeTtsError};
use cosyvoice_native_server::audio::{self, MelConfig};

// Use jemalloc for better memory allocation performance
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Application state shared across handlers.
struct AppState {
    // Engine is wrapped in Mutex because OnnxFrontend requires mutable access
    tts: Mutex<NativeTtsEngine>,
    tokenizer: tokenizers::Tokenizer,
    model_dir: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file
    match dotenvy::from_filename(".env") {
        Ok(path) => eprintln!("Loaded environment from: {}", path.display()),
        Err(e) if e.not_found() => eprintln!("No .env file found (this is OK)"),
        Err(e) => eprintln!("Warning: Could not load .env: {}", e),
    }

    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cosyvoice_native_server=info,tower_http=info".into()),
        )
        .init();

    // Initialize Prometheus metrics
    let builder = PrometheusBuilder::new();
    let handle = builder.install_recorder()?;
    info!("Prometheus metrics recorder installed");

    // Get model directory
    let model_dir = env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| config::DEFAULT_MODEL_DIR.to_string());

    // Initialize Native TTS engine
    info!(model_dir = %model_dir, "Initializing Native TTS engine...");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    info!("Using device: {:?}", device);

    let tts = NativeTtsEngine::new(&model_dir, Some(device))?;
    info!("Native TTS engine initialized successfully");

    // Initialize Tokenizer
    let tokenizer_path = PathBuf::from(&model_dir).join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(anyhow::anyhow!("Tokenizer file not found at {:?}", tokenizer_path));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
     info!("Tokenizer initialized successfully");

    let state = Arc::new(AppState {
        tts: Mutex::new(tts),
        tokenizer,
        model_dir,
    });

    // Build router
    let app = Router::new()
        .route("/synthesize", post(synthesize_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(move || std::future::ready(handle.render())))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config::DEFAULT_PORT)); // Consider using a different port or env var
    info!("Starting native server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down gracefully");
    Ok(())
}

/// POST /synthesize - Convert text to speech
async fn synthesize_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SynthesizeRequest>,
) -> Response {
    let start = Instant::now();
    counter!("tts_requests_total").increment(1);

    info!(
        text_len = request.text.len(),
        speaker = ?request.speaker,
        has_prompt = request.prompt_audio.is_some(),
        "Synthesizing speech (Native)"
    );

    // TODO: Implement full synthesis logic
    // This requires:
    // 1. Text -> Tokens (using Tokenizer)
    // 2. Audio -> Mel/Fbank (using audio module)
    // 3. Call tts.synthesize_instruct (or similar)

    // For now, let's verify we can process the inputs

    // 1. Text Tokenization
    // CosyVoice expects [sos, text_tokens, task_id, prompt_tokens]
    // The tokenizer encoding usually gives the text part.
    // Need to handle SOS/task_id separately or check how NativeTtsEngine expects them.
    // NativeTtsEngine::synthesize_instruct takes `text_tokens: &Tensor`.
    // It calls `llm.embed_text_tokens`.
    // The `llm.generate` method inside `synthesize_full` constructs [sos, text, task_id...].
    // So we just need the text tokens here.

    let text_encoding = match state.tokenizer.encode(request.text.clone(), true) {
        Ok(enc) => enc,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Tokenizer error: {}", e)),
    };
    let text_ids: Vec<u32> = text_encoding.get_ids().iter().map(|&id| id).collect();

    // 2. Prompt Audio Processing
    if request.prompt_audio.is_none() {
         return error_response(StatusCode::BAD_REQUEST, "prompt_audio is required".to_string());
    }
    let prompt_path = request.prompt_audio.as_ref().unwrap();

    // Load audio
    // We need to do this potentially blocking IO?
    /*
    let (audio_samples, sample_rate) = match audio::load_wav(prompt_path) {
        Ok(res) => res,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load audio: {}", e)),
    };
    */
    // Since we are inside async handler, we should avoid blocking.
    // But for MVP, let's just do it. Or wrap in spawn_blocking.

    let prompt_path_clone = prompt_path.clone();
    let text_ids_clone = text_ids.clone();
    let state_clone = state.clone();
    let speed = request.speed;

    // Run heavy lifting in spawn_blocking?
    // Problem: `tts` is inside Mutex (async mutex). We can't easily move the locked guard into blocking thread.
    // But `NativeTtsEngine` uses Candle which might be CPU intensive (or GPU).
    // If GPU, it's async-ish (kernels launch), but synchronization happens.
    // Ideally we lock, run, unlock.

    // Let's load audio first (blocking IO)
    let load_audio_result = tokio::task::spawn_blocking(move || {
        audio::load_wav(prompt_path_clone)
    }).await;

    let (audio_samples, sample_rate) = match load_audio_result {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load audio: {}", e)),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Task join error: {}", e)),
    };

    // We have audio samples.
    // Need to compute Mel (24k) and Fbank (16k or whatever OnnxFrontend expects?)
    // OnnxFrontend expects 16k for Campplus/SpeechTokenizer?
    // Let's check `OnnxFrontend` usage in `native_tts.rs`.
    // `process_prompt_tensors(prompt_speech_16k, prompt_fbank)`
    // `prompt_speech_16k`: [1, 128, T] mel? No, "prompt_speech_16k for Tokenizer".
    // `tokenize_speech` takes `mel: &Tensor` [batch, 128, frames].
    // So we need to compute MEL spectrograms.

    // Configs:
    // Flow/HiFT use 24k audio.
    // SpeechTokenizer/Campplus use 16k audio?
    // We need to resample if loaded at 24k (CosyVoice3 default).
    // Or if loaded at 16k.

    // Assuming prompt audio can be anything, strict resampling is needed.
    // For now, assume 24k for Flow.
    // If `sample_rate` != 24000, we need to resample.

    // TODO: Implement Resampling. For now, fail if not 24k or 22050 (if that's what we want).

    // Lock the engine
    let mut tts = state_clone.tts.lock().await;
    let device = tts.device.clone();

    // Convert text tokens to Tensor
    let text_tensor = match Tensor::from_vec(text_ids_clone, (1, text_ids.len()), &device) {
         Ok(t) => t,
         Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)),
    };

    // Calculate Mel/Fbank
    // We need 24k Mel for Flow
    // We need 16k Mel for Tokenizer (SpeechTokenizer) ??
    // We need 16k Fbank for Speaker Embedding (Campplus) ??

    // Let's assume input audio is 24k.
    // We need to resample to 16k for frontend.
    // Since we don't have a resampler in `audio.rs` yet (only upsample 2x),
    // we might need to rely on external tools or add a resampler.

    // For MVP validation, we will skip implementation of full synthesis in `main.rs`
    // and return a 501 Not Implemented with a message.
    // The goal of this task is "Separation". The functional Native Server is a larger task.

    return ((StatusCode::NOT_IMPLEMENTED), "Native synthesis not fully implemented yet (needs resampling/frontend integration in main.rs)").into_response();

    // ... (Remainder of synthesis logic would go here)
}

fn error_response(code: StatusCode, message: String) -> Response {
    (
        code,
        Json(ErrorResponse {
            error: message,
            code: code.as_u16(),
        }),
    )
        .into_response()
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: config::VERSION.to_string(),
        speakers: Some(vec!["Native engine ready".to_string()]),
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down..."),
        _ = terminate => info!("Received SIGTERM, shutting down..."),
    }
}
