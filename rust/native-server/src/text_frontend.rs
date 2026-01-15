//! Minimal text frontend utilities for native parity with Python.

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub const PROMPT_PREFIX: &str = "Please speak in English.<|endofprompt|>";
const EN_TOKEN_MAX: usize = 80;
const EN_TOKEN_MIN: usize = 60;
const EN_MERGE_LEN: usize = 20;

/// Clean text by removing special tokens that shouldn't be spoken.
/// NOTE: This is not used by the parity path; keep it minimal.
pub fn clean_special_tokens(text: &str) -> String {
    let special_tokens = ["<|im_start|>", "<|im_end|>"];
    let mut cleaned = text.to_string();
    for token in &special_tokens {
        cleaned = cleaned.replace(token, "");
    }
    cleaned.replace("  ", " ").trim().to_string()
}

/// Split prompt and actual text content based on <|endofprompt|>
pub fn split_prompt_and_content(text: &str) -> (String, String) {
    // Find the endofprompt marker
    if let Some(pos) = text.find("<|endofprompt|>") {
        // Find the period before endofprompt to split properly
        let before_prompt = &text[..pos];
        if let Some(period_pos) = before_prompt.rfind('.') {
            // Include the period in the prompt
            let prompt = &text[..period_pos + 1];
            let content = &text[pos + "<|endofprompt|>".len()..];
            (prompt.trim().to_string(), content.trim().to_string())
        } else {
            // No period found, use everything before endofprompt
            let prompt = &text[..pos];
            let content = &text[pos + "<|endofprompt|>".len()..];
            (prompt.trim().to_string(), content.trim().to_string())
        }
    } else {
        // No endofprompt found, treat entire text as content
        (String::new(), text.to_string())
    }
}

pub fn ensure_prompt_prefix(text: &str) -> String {
    if text.contains("<|endofprompt|>") {
        text.to_string()
    } else {
        format!("{PROMPT_PREFIX}{text}")
    }
}

pub fn text_normalize_english(
    text: &str,
    tokenizer: &Tokenizer,
    split: bool,
    text_frontend: bool,
) -> Result<Vec<String>> {
    let token_len = |t: &str| token_count(tokenizer, t);
    text_normalize_english_with_counter(text, &token_len, split, text_frontend)
}

fn text_normalize_english_with_counter<F>(
    text: &str,
    token_len: &F,
    split: bool,
    text_frontend: bool,
) -> Result<Vec<String>>
where
    F: Fn(&str) -> Result<usize>,
{
    if !text_frontend || text.is_empty() {
        return Ok(vec![text.to_string()]);
    }

    if contains_special_tokens(text) {
        return Ok(vec![text.to_string()]);
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(vec![String::new()]);
    }

    let normalized = spell_out_numbers(trimmed);

    if !split {
        return Ok(vec![normalized]);
    }

    let mut texts = split_paragraph(
        &normalized,
        token_len,
        EN_TOKEN_MAX,
        EN_TOKEN_MIN,
        EN_MERGE_LEN,
        false,
    )?;

    texts.retain(|t| !is_only_punctuation(t));

    Ok(texts)
}

fn contains_special_tokens(text: &str) -> bool {
    text.contains("<|") && text.contains("|>")
}

fn is_only_punctuation(text: &str) -> bool {
    if text.is_empty() {
        return true;
    }
    // Check if string contains only punctuation/symbols/spaces
    text.chars()
        .all(|c| !c.is_alphanumeric() && !c.is_whitespace())
}

fn spell_out_numbers(text: &str) -> String {
    // First handle currency patterns ($N -> N dollars)
    let text = handle_currency(text);
    // Then handle decimal numbers (3.14 -> three point one four)
    let text = handle_decimals(&text);
    // Then spell out remaining integers
    let text = spell_integers(&text);
    // Finally add spaces around sentence-ending punctuation
    normalize_sentence_ends(&text)
}

/// Handle currency patterns: $5 -> "five dollars", $1 -> "one dollar"
fn handle_currency(text: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            // Found currency pattern, extract the number
            let start = i + 1;
            let mut end = start;
            while end < chars.len() && chars[end].is_ascii_digit() {
                end += 1;
            }
            let digits: String = chars[start..end].iter().collect();
            if let Ok(value) = digits.parse::<u64>() {
                let word = number_to_words(value);
                let unit = if value == 1 { "dollar" } else { "dollars" };
                out.push_str(&format!("{} {}", word, unit));
            } else {
                out.push('$');
                out.push_str(&digits);
            }
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Handle decimal numbers: 3.14 -> "three point one four"
fn handle_decimals(text: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check if we're at the start of a decimal number
        if chars[i].is_ascii_digit() {
            let start = i;
            // Consume integer part
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Check for decimal point followed by digits
            if i < chars.len()
                && chars[i] == '.'
                && i + 1 < chars.len()
                && chars[i + 1].is_ascii_digit()
            {
                let int_part: String = chars[start..i].iter().collect();
                i += 1; // Skip the dot
                let frac_start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let frac_part: String = chars[frac_start..i].iter().collect();
                // Convert: "3.14" -> "three point one four"
                if let Ok(int_val) = int_part.parse::<u64>() {
                    out.push_str(&number_to_words(int_val));
                    out.push_str(" point");
                    // Spell out each fractional digit individually
                    for c in frac_part.chars() {
                        if let Some(d) = c.to_digit(10) {
                            out.push(' ');
                            out.push_str(unit_word(d as u16));
                        }
                    }
                } else {
                    out.push_str(&int_part);
                    out.push('.');
                    out.push_str(&frac_part);
                }
            } else {
                // Just integers, push them back (spell_integers will handle)
                let int_part: String = chars[start..i].iter().collect();
                out.push_str(&int_part);
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Spell out standalone integers
fn spell_integers(text: &str) -> String {
    let mut out = String::new();
    let mut digits_start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            if digits_start.is_none() {
                digits_start = Some(idx);
            }
            continue;
        }

        if let Some(start) = digits_start.take() {
            let digits = &text[start..idx];
            out.push_str(&spell_number_simple(digits));
        }
        out.push(ch);
    }

    if let Some(start) = digits_start {
        let digits = &text[start..];
        out.push_str(&spell_number_simple(digits));
    }

    out
}

/// Add space before sentence-ending punctuation for better TTS prosody
fn normalize_sentence_ends(text: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        match *c {
            '.' | '!' | '?' => {
                // Add space before if previous char is not whitespace
                if i > 0 && !chars[i - 1].is_whitespace() {
                    out.push(' ');
                }
                out.push(*c);
            }
            _ => out.push(*c),
        }
    }
    out
}

fn spell_number_simple(digits: &str) -> String {
    if let Ok(value) = digits.parse::<u64>() {
        number_to_words(value)
    } else {
        digits.to_string()
    }
}

fn number_to_words(value: u64) -> String {
    if value == 0 {
        return "zero".to_string();
    }

    let scales = [
        "",
        "thousand",
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
        "sextillion",
    ];

    let mut groups: Vec<(usize, u16)> = Vec::new();
    let mut remaining = value;
    let mut scale_idx = 0;

    while remaining > 0 {
        let group = (remaining % 1000) as u16;
        if group > 0 {
            groups.push((scale_idx, group));
        }
        remaining /= 1000;
        scale_idx += 1;
    }

    groups.reverse();
    if groups.is_empty() {
        return "zero".to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    for (scale, group) in &groups {
        let mut part = words_under_1000(*group);
        let scale_label = scales.get(*scale).unwrap_or(&"");
        if !scale_label.is_empty() {
            part.push(' ');
            part.push_str(scale_label);
        }
        parts.push(part);
    }

    if parts.len() == 1 {
        return parts[0].clone();
    }

    let last_group = groups.last().map(|g| g.1).unwrap_or(0);
    let mut result = String::new();
    for (idx, part) in parts.iter().enumerate() {
        if idx == 0 {
            result.push_str(part);
            continue;
        }
        if idx == parts.len() - 1 {
            if last_group < 100 {
                result.push_str(" and ");
            } else {
                result.push_str(", ");
            }
        } else {
            result.push_str(", ");
        }
        result.push_str(part);
    }

    result
}

fn words_under_1000(value: u16) -> String {
    let hundreds = value / 100;
    let remainder = value % 100;

    if hundreds == 0 {
        return words_under_100(remainder);
    }

    if remainder == 0 {
        format!("{} hundred", unit_word(hundreds))
    } else {
        format!(
            "{} hundred and {}",
            unit_word(hundreds),
            words_under_100(remainder)
        )
    }
}

fn words_under_100(value: u16) -> String {
    match value {
        0 => "zero".to_string(),
        1..=9 => unit_word(value).to_string(),
        10..=19 => teen_word(value).to_string(),
        _ => {
            let tens = value / 10;
            let ones = value % 10;
            let tens_word = tens_word(tens);
            if ones == 0 {
                tens_word.to_string()
            } else {
                format!("{}-{}", tens_word, unit_word(ones))
            }
        }
    }
}

fn unit_word(value: u16) -> &'static str {
    match value {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        _ => "zero",
    }
}

fn teen_word(value: u16) -> &'static str {
    match value {
        10 => "ten",
        11 => "eleven",
        12 => "twelve",
        13 => "thirteen",
        14 => "fourteen",
        15 => "fifteen",
        16 => "sixteen",
        17 => "seventeen",
        18 => "eighteen",
        19 => "nineteen",
        _ => "ten",
    }
}

fn tens_word(value: u16) -> &'static str {
    match value {
        2 => "twenty",
        3 => "thirty",
        4 => "forty",
        5 => "fifty",
        6 => "sixty",
        7 => "seventy",
        8 => "eighty",
        9 => "ninety",
        _ => "",
    }
}

fn token_count(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenizer encode failed: {}", e))?;
    Ok(encoding.get_ids().len())
}

fn split_paragraph<F>(
    text: &str,
    token_len: &F,
    token_max_n: usize,
    token_min_n: usize,
    merge_len: usize,
    comma_split: bool,
) -> Result<Vec<String>>
where
    F: Fn(&str) -> Result<usize>,
{
    let mut chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Ok(Vec::new());
    }

    let mut punc = vec!['.', '?', '!', ';', ':'];
    if comma_split {
        punc.push(',');
    }

    if !punc.contains(chars.last().unwrap()) {
        chars.push('.');
    }

    let mut utts = Vec::new();
    let mut current = String::new();
    let mut has_content = false;
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        if !punc.contains(&c) {
            has_content = true;
        }
        if punc.contains(&c) {
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                if next == '"' || next == '\u{201D}' {
                    current.push(next);
                    i += 1;
                }
            }
            if has_content {
                utts.push(current.clone());
            }
            current.clear();
            has_content = false;
        }
        i += 1;
    }

    let mut final_utts = Vec::new();
    let mut cur_utt = String::new();
    for utt in utts {
        let combined = format!("{cur_utt}{utt}");
        if token_len(&combined)? > token_max_n && token_len(&cur_utt)? > token_min_n {
            if !cur_utt.is_empty() {
                final_utts.push(cur_utt);
            }
            cur_utt = String::new();
        }
        cur_utt.push_str(&utt);
    }

    if !cur_utt.is_empty() {
        if token_len(&cur_utt)? < merge_len && !final_utts.is_empty() {
            if let Some(last) = final_utts.last_mut() {
                last.push_str(&cur_utt);
            }
        } else {
            final_utts.push(cur_utt);
        }
    }

    Ok(final_utts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_english_spells_numbers() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter("I have 2 dogs.", &counter, true, true)?;
        assert_eq!(out, vec!["I have two dogs ."]);
        Ok(())
    }

    #[test]
    fn normalize_skips_special_tokens() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter(
            "Please speak.<|endofprompt|>Test.",
            &counter,
            true,
            true,
        )?;
        assert_eq!(out, vec!["Please speak.<|endofprompt|>Test."]);
        Ok(())
    }

    #[test]
    fn test_decimal_numbers() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        // New behavior: 3.14 -> "three point one four" (WeText parity)
        let out = text_normalize_english_with_counter("It is 3.14 value.", &counter, true, true)?;
        assert_eq!(out, vec!["It is three point one four value ."]);
        Ok(())
    }

    #[test]
    fn test_currency() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        // New behavior: $5 -> "five dollars" (WeText parity)
        let out = text_normalize_english_with_counter("Costs $5.", &counter, true, true)?;
        assert_eq!(out, vec!["Costs five dollars ."]);

        let out2 = text_normalize_english_with_counter("$1 price.", &counter, true, true)?;
        assert_eq!(out2, vec!["one dollar price ."]);
        Ok(())
    }

    #[test]
    fn test_punctuation_spacing() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter("Hello,world!", &counter, true, true)?;
        // Now adds space before ! only (sentence-ending punctuation)
        assert_eq!(out, vec!["Hello,world !"]);
        Ok(())
    }
}
