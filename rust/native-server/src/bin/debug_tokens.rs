use tokenizers::Tokenizer;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("pretrained_models/Fun-CosyVoice3-0.5B/tokenizer.json")?;

    let prompt_prefix = "Please speak in English.<|endofprompt|>";
    let default_prompt_text = "Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that.";
    let full_prompt_text = format!("{}{}", prompt_prefix, default_prompt_text);

    let tts_text = "Hello! I am an AI voice assistant powered by Fun-CosyVoice3. How may I help you today?";

    // Simulate native_example.rs logic
    let prompt_encoding = tokenizer.encode(full_prompt_text.clone(), true).map_err(|e| e.to_string())?;
    let tts_encoding = tokenizer.encode(tts_text, true).map_err(|e| e.to_string())?;

    println!("Full Prompt Text: {:?}", full_prompt_text);
    println!("TTS Text: {:?}", tts_text);

    println!("\n--- Rust Tokenization ---");
    println!("Prompt Tokens (Len {}): {:?}", prompt_encoding.get_ids().len(), prompt_encoding.get_ids());
    println!("TTS Tokens (Len {}): {:?}", tts_encoding.get_ids().len(), tts_encoding.get_ids());

    let concatenated: Vec<u32> = [prompt_encoding.get_ids(), tts_encoding.get_ids()].concat();
    println!("Concatenated (Len {}): {:?}", concatenated.len(), concatenated);

    Ok(())
}
