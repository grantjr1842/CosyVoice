//! Test binary for verifying native TTS component weight loading.

use cosyvoice_native_server::tts::NativeTtsEngine;
use candle_core::{Device, Tensor, DType};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set ORT path if not set
    /*
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        let path = std::path::Path::new("../../.pixi/envs/default/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.19.2");
        if path.exists() {
             // Convert to absolute path because ORT might need it or CWD context matters
             if let Ok(abs_path) = std::fs::canonicalize(path) {
                 println!("Setting ORT_DYLIB_PATH to {:?}", abs_path);
                 std::env::set_var("ORT_DYLIB_PATH", abs_path);
             }
        } else {
             println!("Warning: ORT lib not found at expected path: {:?}", path);
        }
    }
    */

    println!("=== Testing Native TTS Engine End-to-End ===\n");

    let model_dir = std::env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| "../../pretrained_models/Fun-CosyVoice3-0.5B".to_string());

    println!("Model directory: {}", model_dir);
    println!("Current Directory: {:?}", std::env::current_dir().unwrap());

    // Load test artifacts
    let artifact_path_str = std::env::var("ARTIFACT_PATH")
        .unwrap_or_else(|_| "tests/test_artifacts.safetensors".to_string());
    let artifact_path = Path::new(&artifact_path_str);
    if !artifact_path.exists() {
        eprintln!("Error: {} not found. Run 'pixi run python tests/generate_test_artifacts.py' first.", artifact_path_str);
        std::process::exit(1);
    }

    println!("Candle CUDA available: {}", candle_core::utils::cuda_is_available());
    let device = if candle_core::utils::cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to create CUDA device 0: {}", e);
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    };
    println!("Selected device: {:?}", device);

    let cpu = Device::Cpu;
    let artifacts = safe_load_artifacts(artifact_path.to_str().unwrap(), &cpu)?;

    // Unpack artifacts and move to device
    let flow_feat_24k = artifacts.get("flow_feat_24k").expect("Missing flow_feat_24k").to_device(&device)?;
    let _text_ids = artifacts.get("text_ids").expect("Missing text_ids").to_dtype(DType::U32)?.to_device(&device)?;

    // New artifacts for bypassing frontend
    let speech_tokens = artifacts.get("speech_tokens").expect("Missing speech_tokens in artifacts. Did you upgrade generate_test_artifacts.py?").to_dtype(DType::U32)?.to_device(&device)?;
    let speaker_embedding = artifacts.get("speaker_embedding").expect("Missing speaker_embedding in artifacts.").to_device(&device)?;

    // Initialize Engine
    let engine = NativeTtsEngine::new(&model_dir, Some(device.clone()))?;
    println!("\n✅ Engine loaded successfully!");

    // Run synthesis
    println!("\nSynthesizing (Direct/Bypassing Frontend)...");
    let _start_total = std::time::Instant::now();

    if std::env::var("TEST_FLOW_ONLY").unwrap_or_default() == "1" {
        println!("*** Running FLOW PARITY TEST (Dummy Inputs) ***");
        let token = Tensor::zeros((1, 50), DType::U32, &device)?;
        let prompt_token = Tensor::zeros((1, 10), DType::U32, &device)?;
        let prompt_feat = Tensor::randn(0.0f32, 1.0f32, (1, 80, 10), &device)?;
        let embedding = Tensor::randn(0.0f32, 1.0f32, (1, 192), &device)?;

        println!("Running Flow inference...");
        engine.flow.inference(&token, &prompt_token, &prompt_feat, &embedding, 10, None)?;
        println!("Flow test complete. Exiting.");
        return Ok(());
    }

    // Load additional artifacts
    let flow_noise = artifacts.get("flow_noise").expect("Missing flow_noise").to_device(&device)?;

    // We skip LLM generation and use speech_tokens from artifacts to test Flow+HiFT
    println!("\nSynthesizing (Flow + HiFT from Artifact Tokens)...");
    let start_synth = std::time::Instant::now();

    // Convert speech_tokens tensor (1, 87) to Vec<u32> if needed?
    // flow.inference takes &Tensor. synthesize_from_tokens takes &Tensor.
    // So we can pass speech_tokens directly.

    // Create dummy prompt tokens/mel since we are using speech_tokens directly and Flow likely handles them?
    // Note: Flow inference uses prompt_tokens and prompt_mel for conditioning.
    // The artifacts typically correspond to a specific prompt.
    // If we don't have matching prompt artifacts, we might get mismatch?
    // But `speech_tokens` were generated conditioned on *some* prompt.
    // Ideally we pass the SAME prompt.
    // `flow_feat_24k` is target mel.
    // Artifacts likely include `prompt_speech_24k`?
    // tests/inspect_artifacts.py showed: flow_feat_24k, flow_noise, speech_tokens, speaker_embedding.
    // No `prompt_tokens`?
    // Let's assume zero prompt for now or random, but this might affect Flow output quality?
    // Wait, if `speech_tokens` were generated with a prompt, Flow expects that prompt.
    // If artifacts don't have it, we might be stuck.
    // But let's try with empty prompt (which is valid for Zero-Shot if prompt is effectively handled or if SFT).
    // Or create dummy zero prompt.

    let empty_prompt_token = Tensor::zeros((1, 0), DType::U32, &device)?;
    let empty_prompt_mel = Tensor::zeros((1, 80, 0), DType::F32, &device)?;

    let audio_samples = engine.synthesize_from_tokens(
        &speech_tokens,
        &empty_prompt_token,
        &empty_prompt_mel,
        &speaker_embedding,
        Some(&flow_noise)
    )?;

    let duration_synth = start_synth.elapsed();
    println!("\n✅ Synthesis complete!");
    println!("   Audio duration: {:.2}s", audio_samples.len() as f64 / 24000.0);
    println!("   Time taken: {:?}", duration_synth);

    // Save to WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let _ = std::fs::create_dir_all("outputs/audio");
    let mut writer = hound::WavWriter::create("outputs/audio/native_output.wav", spec)?;
    for sample in audio_samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    println!("Saved audio to outputs/audio/native_output.wav");

    // 3. Test Direct HiFT (using artifact Mel)
    println!("\nSynthesizing (Direct HiFT from Artifact Mel)...");
    // Ensure flow_feat_24k is [1, 80, T]
    // inspect_artifacts showed [1, 80, 171]. This is correct.
    let audio_samples_hift = engine.synthesize_from_mel(&flow_feat_24k)?;

    let mut wav_writer_hift = hound::WavWriter::create("outputs/audio/native_hift_output.wav", spec)?;
    for sample in audio_samples_hift {
        wav_writer_hift.write_sample(sample)?;
    }
    wav_writer_hift.finalize()?;
    println!("Saved direct output to outputs/audio/native_hift_output.wav");

    Ok(())
}

fn safe_load_artifacts(path: &str, device: &Device) -> Result<std::collections::HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    use candle_core::safetensors::load;
    let tensors = load(path, device)?;
    Ok(tensors)
}
