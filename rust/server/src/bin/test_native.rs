//! Test binary for verifying native TTS component weight loading.

use cosyvoice_rust_backend::native_tts::NativeTtsEngine;

fn main() {
    println!("=== Testing Native TTS Engine Weight Loading ===\n");

    let model_dir = std::env::var("COSYVOICE_MODEL_DIR")
        .unwrap_or_else(|_| "pretrained_models/Fun-CosyVoice3-0.5B".to_string());

    println!("Model directory: {}", model_dir);

    match NativeTtsEngine::new(&model_dir) {
        Ok(engine) => {
            println!("\n✅ All components loaded successfully!");
            println!("   - LLM: Loaded");
            println!("   - Flow: Loaded");
            println!("   - HiFT: Loaded");
            println!("   - Device: {:?}", engine.device);
            println!("   - Sample Rate: {}", engine.sample_rate);
        }
        Err(e) => {
            eprintln!("\n❌ Failed to load engine: {}", e);
            std::process::exit(1);
        }
    }
}
