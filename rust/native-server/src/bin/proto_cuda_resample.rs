use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use cosyvoice_native_server::audio;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Prototype CUDA resampler using Candle conv1d", long_about = None)]
struct Args {
    #[arg(long, default_value = "asset/interstellar-tars-01-resemble-denoised.wav")]
    wav: String,

    #[arg(long, default_value_t = 16000)]
    dst_rate: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0).context("CUDA device required")?;

    let wav_path = PathBuf::from(&args.wav);
    let (samples, sr) = audio::load_wav(&wav_path)?;
    println!("Loaded wav: {} samples, sr {}", samples.len(), sr);

    let resampled = audio::resample_audio_cuda(&samples, sr, args.dst_rate, &device)?;
    println!("Resampled tensor shape: {:?}", resampled.shape());

    Ok(())
}
