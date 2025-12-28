//! CosyVoice TTS Server
//!
//! High-performance text-to-speech server using CosyVoice Python backend via PyO3.

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
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
};
use tokio::signal;
use tower_http::trace::TraceLayer;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use shared::{config, ErrorResponse, HealthResponse, SynthesizeRequest};

mod tts;
use tts::TtsEngine;

// Use jemalloc for better memory allocation performance
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Application state shared across handlers.
struct AppState {
    tts: TtsEngine,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file from current directory (project root)
    // start-server.sh cd's to project root before running binary
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
                .unwrap_or_else(|_| "cosyvoice_server=info,tower_http=info".into()),
        )
        .init();

    // Initialize Prometheus metrics
    let builder = PrometheusBuilder::new();
    let handle = builder.install_recorder()?;
    info!("Prometheus metrics recorder installed");

    // Get model directory from env or use default
    let model_dir = env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| config::DEFAULT_MODEL_DIR.to_string());

    // Initialize TTS engine
    info!(model_dir = %model_dir, "Initializing CosyVoice TTS engine...");
    let tts = TtsEngine::new(&model_dir)?;

    // List available speakers
    match tts.list_speakers() {
        Ok(speakers) => info!(speakers = ?speakers, "Available speakers"),
        Err(e) => warn!(error = %e, "Could not list speakers"),
    }

    info!("CosyVoice TTS engine initialized successfully");

    let state = Arc::new(AppState { tts });

    // Build router
    let app = Router::new()
        .route("/synthesize", post(synthesize_handler))
        .route("/health", get(health_handler))
        .route("/speakers", get(speakers_handler))
        .route("/metrics", get(move || std::future::ready(handle.render())))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config::DEFAULT_PORT));
    info!("Starting server on {}", addr);

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
) -> impl IntoResponse {
    let start = Instant::now();
    counter!("tts_requests_total").increment(1);

    info!(
        text_len = request.text.len(),
        speaker = ?request.speaker,
        has_prompt = request.prompt_audio.is_some(),
        "Synthesizing speech"
    );

    // Use spawn_blocking since Python inference is CPU-bound
    let result = tokio::task::spawn_blocking(move || {
        // CosyVoice3 requires prompt audio for all synthesis modes
        let prompt_audio = match &request.prompt_audio {
            Some(path) => path.as_str(),
            None => {
                return Err(tts::TtsError::SynthesisError(
                    "prompt_audio is required for CosyVoice3 synthesis".to_string()
                ));
            }
        };

        if let Some(prompt_text) = &request.prompt_text {
            // Zero-shot voice cloning
            state.tts.synthesize_zero_shot(
                &request.text,
                prompt_audio,
                prompt_text,
                request.speed,
            )
        } else {
            // Instruct mode with default instruction
            let instruct = request.speaker.as_deref().unwrap_or("Speak naturally in English.");
            state.tts.synthesize_instruct(&request.text, instruct, prompt_audio, request.speed)
        }
    }).await;

    match result {
        Ok(Ok((samples, sample_rate))) => {
            let duration = start.elapsed();
            let audio_duration = samples.len() as f32 / sample_rate as f32;
            let rtf = duration.as_secs_f32() / audio_duration;

            histogram!("tts_synthesis_duration_seconds").record(duration.as_secs_f64());
            histogram!("tts_rtf").record(rtf as f64);
            counter!("tts_requests_success").increment(1);

            info!(
                duration_ms = duration.as_millis(),
                audio_secs = audio_duration,
                rtf = rtf,
                "Synthesis complete"
            );

            // Encode to WAV
            let wav_data = encode_wav_i16(&samples, sample_rate);

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
        Ok(Err(e)) => {
            counter!("tts_requests_error").increment(1);
            error!(error = %e, "Synthesis failed");

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: 500,
                }),
            )
                .into_response()
        }
        Err(e) => {
            counter!("tts_requests_error").increment(1);
            error!(error = %e, "Task join error");

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Internal error: {}", e),
                    code: 500,
                }),
            )
                .into_response()
        }
    }
}

/// GET /health - Health check endpoint
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let speakers = state.tts.list_speakers().ok();
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: config::VERSION.to_string(),
        speakers,
    })
}

/// GET /speakers - List available speakers
async fn speakers_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.tts.list_speakers() {
        Ok(speakers) => (StatusCode::OK, Json(speakers)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: 500,
            }),
        )
            .into_response(),
    }
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

/// Graceful shutdown signal handler.
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
