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
    path::PathBuf,
    time::Instant,
};
use tokio::signal;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use candle_core::{Device, Tensor};

use shared::{config, ErrorResponse, HealthResponse, SynthesizeRequest};

use cosyvoice_native_server::tts::NativeTtsEngine;
use cosyvoice_native_server::audio::{self, MelConfig};
use cosyvoice_native_server::text_frontend::text_normalize_english;

// Use jemalloc for better memory allocation performance
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Application state shared across handlers.
struct AppState {
    // Engine is wrapped in Mutex because OnnxFrontend requires mutable access
    tts: Mutex<NativeTtsEngine>,
    tokenizer: tokenizers::Tokenizer,
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

    if request.text.trim().is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "text is required".to_string());
    }

    let prompt_audio = match request.prompt_audio.as_deref() {
        Some(path) => path.to_string(),
        None => return error_response(StatusCode::BAD_REQUEST, "prompt_audio is required".to_string()),
    };

    let prompt_text = match request.prompt_text.as_deref() {
        Some(text) if !text.trim().is_empty() => text.to_string(),
        Some(_) => return error_response(StatusCode::BAD_REQUEST, "prompt_text is empty".to_string()),
        None => request
            .speaker
            .clone()
            .unwrap_or_else(|| "Speak naturally in English.".to_string()),
    };

    let prompt_segments = match text_normalize_english(&prompt_text, &state.tokenizer, false, true) {
        Ok(segments) => segments,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt text normalize error: {e}")),
    };
    let prompt_text = match prompt_segments.into_iter().next() {
        Some(text) if !text.trim().is_empty() => text,
        _ => return error_response(StatusCode::BAD_REQUEST, "prompt_text normalized to empty".to_string()),
    };
    let prompt_tokens = match encode_tokens(&state.tokenizer, &prompt_text) {
        Ok(tokens) => tokens,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let prompt_text_len = prompt_tokens.len();

    let mut tts_segments = match text_normalize_english(&request.text, &state.tokenizer, true, true) {
        Ok(segments) => segments,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Text normalize error: {e}")),
    };
    tts_segments.retain(|segment| !segment.trim().is_empty());
    if tts_segments.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "text normalized to empty".to_string());
    }

    let load_audio_result = tokio::task::spawn_blocking(move || audio::load_wav(prompt_audio)).await;
    let (prompt_samples, prompt_sr) = match load_audio_result {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load audio: {e}")),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Task join error: {e}")),
    };

    let prompt_16k = match audio::resample_audio(&prompt_samples, prompt_sr, 16000) {
        Ok(samples) => samples,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Resample to 16k failed: {e}")),
    };
    let prompt_24k = match audio::resample_audio(&prompt_samples, prompt_sr, 24000) {
        Ok(samples) => samples,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Resample to 24k failed: {e}")),
    };

    let prompt_speech_16k = match audio::whisper_log_mel_spectrogram(&prompt_16k, &Device::Cpu) {
        Ok(mel) => mel,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Whisper mel failed: {e}")),
    };
    let prompt_fbank = match audio::kaldi_fbank(&prompt_16k, 16000, &Device::Cpu) {
        Ok(fbank) => fbank,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Fbank failed: {e}")),
    };

    let device = {
        let tts = state.tts.lock().await;
        tts.device.clone()
    };

    let mut prompt_speech_24k = match audio::mel_spectrogram(&prompt_24k, &MelConfig::cosyvoice3(), &device) {
        Ok(mel) => mel,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("24k mel failed: {e}")),
    };

    let mut tts = state.tts.lock().await;
    let (mut prompt_speech_tokens, speaker_embedding) = match tts.process_prompt_tensors(&prompt_speech_16k, &prompt_fbank) {
        Ok(res) => res,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt processing failed: {e}")),
    };

    let mel_len = match prompt_speech_24k.dim(2) {
        Ok(len) => len,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt mel shape error: {e}")),
    };
    let token_len = match prompt_speech_tokens.dim(1) {
        Ok(len) => len,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt token shape error: {e}")),
    };
    let aligned_token_len = usize::min(mel_len / 2, token_len);
    if aligned_token_len == 0 {
        return error_response(StatusCode::BAD_REQUEST, "prompt audio produced no usable tokens".to_string());
    }

    if aligned_token_len != token_len {
        prompt_speech_24k = match prompt_speech_24k.narrow(2, 0, aligned_token_len * 2) {
            Ok(mel) => mel,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt mel align failed: {e}")),
        };
        prompt_speech_tokens = match prompt_speech_tokens.narrow(1, 0, aligned_token_len) {
            Ok(tokens) => tokens,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Prompt token align failed: {e}")),
        };
    }

    let sample_rate = tts.sample_rate;
    let mut all_samples: Vec<i16> = Vec::new();
    for segment in tts_segments {
        let tts_tokens = match encode_tokens(&state.tokenizer, &segment) {
            Ok(tokens) => tokens,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
        };

        let mut text_tokens = Vec::with_capacity(prompt_text_len + tts_tokens.len());
        text_tokens.extend_from_slice(&prompt_tokens);
        text_tokens.extend_from_slice(&tts_tokens);

        let text_tensor = match Tensor::from_vec(
            text_tokens,
            (1, prompt_text_len + tts_tokens.len()),
            &device,
        ) {
            Ok(t) => t,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Text tensor error: {e}")),
        };
        let text_embeds = match tts.llm.embed_text_tokens(&text_tensor) {
            Ok(embeds) => embeds,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Text embed error: {e}")),
        };

        let segment_samples = match tts.synthesize_full_with_prompt_len(
            &text_embeds,
            prompt_text_len,
            Some(&prompt_speech_tokens),
            &prompt_speech_24k,
            &speaker_embedding,
            25,
        ) {
            Ok(samples) => samples,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Synthesis failed: {e}")),
        };

        all_samples.extend_from_slice(&segment_samples);
    }

    let duration = start.elapsed();
    let audio_duration = all_samples.len() as f32 / sample_rate as f32;
    if audio_duration > 0.0 {
        let rtf = duration.as_secs_f32() / audio_duration;
        histogram!("tts_synthesis_duration_seconds").record(duration.as_secs_f64());
        histogram!("tts_rtf").record(rtf as f64);
    }

    let wav_data = encode_wav_i16(&all_samples, sample_rate);
    let mut response = (StatusCode::OK, wav_data).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("audio/wav"),
    );
    if let Ok(duration_header) = header::HeaderValue::try_from(audio_duration.to_string()) {
        response.headers_mut().insert(
            header::HeaderName::from_static("x-audio-duration"),
            duration_header,
        );
    }
    response
}

fn encode_tokens(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenizer encode failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
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

/// Encode audio samples (i16) to WAV format.
fn encode_wav_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut buffer = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buffer);
        let mut writer = hound::WavWriter::new(cursor, spec).expect("Failed to create WAV writer");

        for &sample in samples {
            writer.write_sample(sample).expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize WAV");
    }

    buffer
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
