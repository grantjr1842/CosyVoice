use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_file("pretrained_models/Fun-CosyVoice3-0.5B/tokenizer.json")
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

    let text = "<|endofprompt|>";
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("Tokenize error: {}", e))?;
    println!("Text: {}", text);
    println!("IDs: {:?}", encoding.get_ids());
    println!("Tokens: {:?}", encoding.get_tokens());

    Ok(())
}
