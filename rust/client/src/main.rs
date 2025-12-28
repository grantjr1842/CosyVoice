//! CosyVoice TTS Client
//!
//! CLI client for interacting with the CosyVoice TTS server.

use clap::Parser;
use reqwest::Client;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use shared::SynthesizeRequest;

/// CosyVoice TTS Client
#[derive(Parser, Debug)]
#[command(name = "cosyvoice-client")]
#[command(author, version, about = "CLI client for CosyVoice TTS server")]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Output file path (e.g., output.wav)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Speaker ID for SFT mode (e.g., "中文女", "英文女")
    #[arg(short, long)]
    speaker: Option<String>,

    /// Path to reference audio for zero-shot voice cloning
    #[arg(long)]
    prompt_audio: Option<String>,

    /// Transcription of reference audio (required for zero-shot)
    #[arg(long)]
    prompt_text: Option<String>,

    /// Speech speed (0.5 to 2.0)
    #[arg(long, default_value = "1.0")]
    speed: f32,

    /// Server URL
    #[arg(long, default_value = "http://localhost:3000")]
    server: String,

    /// List available speakers
    #[arg(long)]
    list_speakers: bool,

    /// Stream audio to stdout (pipe to aplay)
    #[arg(long)]
    stream: bool,

    /// Enable verbose output
    #[arg(short = 'V', long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::EnvFilter::new(level))
        .init();

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    // List speakers mode
    if args.list_speakers {
        let url = format!("{}/speakers", args.server);
        let response = client.get(&url).send().await?;

        if response.status().is_success() {
            let speakers: Vec<String> = response.json().await?;
            println!("Available speakers:");
            for speaker in speakers {
                println!("  - {}", speaker);
            }
        } else {
            error!("Failed to list speakers: {}", response.status());
        }
        return Ok(());
    }

    let request = SynthesizeRequest {
        text: args.text.clone(),
        prompt_audio: args.prompt_audio,
        prompt_text: args.prompt_text,
        speaker: args.speaker.clone(),
        speed: args.speed,
    };

    info!(
        text_len = args.text.len(),
        speaker = ?args.speaker,
        speed = args.speed,
        "Sending synthesis request"
    );

    let start = Instant::now();
    let url = format!("{}/synthesize", args.server);

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await?;
        error!(status = %status, body = %body, "Server returned error");
        anyhow::bail!("Server error: {} - {}", status, body);
    }

    let audio_duration: f32 = response
        .headers()
        .get("x-audio-duration")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let audio_data = response.bytes().await?;
    let elapsed = start.elapsed();

    info!(
        elapsed_ms = elapsed.as_millis(),
        audio_secs = audio_duration,
        bytes = audio_data.len(),
        "Received audio"
    );

    if args.stream {
        // Write raw audio to stdout for piping to aplay
        let mut stdout = std::io::stdout();
        stdout.write_all(&audio_data)?;
        stdout.flush()?;
    } else if let Some(output_path) = &args.output {
        std::fs::write(output_path, &audio_data)?;
        info!(path = %output_path.display(), "Saved audio file");

        println!("✓ Synthesized {} chars in {:.2}s", args.text.len(), elapsed.as_secs_f32());
        println!("  Audio duration: {:.2}s", audio_duration);
        println!("  RTF: {:.2}x", audio_duration / elapsed.as_secs_f32());
        println!("  Output: {}", output_path.display());
    } else {
        eprintln!("Warning: No output specified. Use --output or --stream");
    }

    Ok(())
}
