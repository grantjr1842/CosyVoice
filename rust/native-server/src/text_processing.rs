//! Text processing utilities to handle special tokens properly

use anyhow::Result;
use tokenizers::Tokenizer;

/// List of special tokens that should be filtered out from speech
const SPECIAL_TOKENS: &[&str] = &["<|en|>", "<|endofprompt|>", "<|im_start|>", "<|im_end|>"];

/// Clean text by removing special tokens that shouldn't be spoken
pub fn clean_special_tokens(text: &str) -> String {
    let mut cleaned = text.to_string();
    for token in SPECIAL_TOKENS {
        cleaned = cleaned.replace(token, "");
    }
    // Clean up extra whitespace
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split prompt and actual text content
pub fn split_prompt_and_text(text: &str) -> (String, String) {
    if let Some(pos) = text.find("<|endofprompt|>") {
        let prompt_end = pos + "<|endofprompt|>".len();
        let prompt = text[..prompt_end].to_string();
        let content = text[prompt_end..].trim().to_string();
        (prompt, content)
    } else {
        // No endofprompt found, treat entire text as content
        (String::new(), text.to_string())
    }
}

/// Test the special token handling
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_special_tokens() {
        assert_eq!(clean_special_tokens("<|en|>Hello world"), "Hello world");
        assert_eq!(
            clean_special_tokens("You are helpful.<|endofprompt|>Hello"),
            "You are helpful. Hello"
        );
        assert_eq!(clean_special_tokens("<|en|><|endofprompt|>Test"), "Test");
    }

    #[test]
    fn test_split_prompt_and_text() {
        let (prompt, text) = split_prompt_and_text("You are helpful.<|endofprompt|>Hello world");
        assert_eq!(prompt, "You are helpful.<|endofprompt|>");
        assert_eq!(text, "Hello world");

        let (prompt, text) = split_prompt_and_text("Just regular text");
        assert_eq!(prompt, "");
        assert_eq!(text, "Just regular text");
    }
}
