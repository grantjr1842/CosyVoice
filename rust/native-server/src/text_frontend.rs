//! Minimal text frontend utilities for native parity with Python.

use anyhow::{anyhow, Result};
use regex::Regex;
use tokenizers::Tokenizer;

pub const PROMPT_PREFIX: &str = "You are a helpful assistant.<|endofprompt|>";
const EN_PREFIX: &str = "<|en|>";
const EN_TOKEN_MAX: usize = 80;
const EN_TOKEN_MIN: usize = 60;
const EN_MERGE_LEN: usize = 20;

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
    if text.is_empty() {
        return Ok(vec![text.to_string()]);
    }

    let mut text_frontend = text_frontend;
    if contains_special_tokens(text) {
        text_frontend = false;
    }

    if !text_frontend {
        return Ok(vec![text.to_string()]);
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(vec![String::new()]);
    }

    // Pipeline: Numbers -> Punctuation -> Splitting
    let normalized = spell_out_numbers(trimmed);
    let normalized = normalize_punctuation(&normalized);

    // If split is false, return the normalized string (wrapped in a vec)
    // Note: The Python code splits paragraphs even if split=False if Chinese,
    // but for English with wetext it returns list.
    // Here we replicate logic: if !split, just return the whole normalized text.
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
    texts = texts
        .into_iter()
        .map(|t| format!("{EN_PREFIX}{t}"))
        .collect();
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

fn normalize_punctuation(text: &str) -> String {
    // Add spaces around punctuation similar to wetext/Python logic
    // WeText often does: "Hello!" -> "Hello !"
    // But we need to be careful with decimals (3.14) which are handled by spell_out_numbers first

    // Add space before and after specific punctuation marks: ! ? , . ; :
    // Use regex replacement.
    let re = Regex::new(r"([!?,.;:])").unwrap();
    let text = re.replace_all(text, " $1 ");

    // Clean up multiple spaces
    let re_spaces = Regex::new(r"\s+").unwrap();
    re_spaces.replace_all(&text, " ").trim().to_string()
}

fn spell_out_numbers(text: &str) -> String {
    // Strategy: find sequences that look like numbers (including decimals, currency)
    // Regex: \$?\d+(?:\.\d+)?
    // \$?       : Optional currency symbol
    // \d+       : Integer part
    // (?:\.\d+)? : Optional decimal part

    let re = Regex::new(r"(\$?)(\d+)(?:\.(\d+))?").unwrap();

    let mut new_text = String::new();
    let mut last_match_end = 0;

    for cap in re.captures_iter(text) {
        let entire_match = cap.get(0).unwrap();
        let range = entire_match.range();

        // Append text before match
        new_text.push_str(&text[last_match_end..range.start]);

        let currency = cap.get(1).map_or("", |m| m.as_str());
        let integer_part = cap.get(2).map_or("", |m| m.as_str());
        let decimal_part = cap.get(3).map_or("", |m| m.as_str());

        let spelled = spell_number_full(currency, integer_part, decimal_part);
        new_text.push_str(&spelled);

        last_match_end = range.end;
    }

    new_text.push_str(&text[last_match_end..]);
    new_text
}

fn spell_number_full(currency: &str, integer_part: &str, decimal_part: &str) -> String {
    let int_val = integer_part.parse::<u64>().unwrap_or(0);
    let mut words = number_to_words(int_val);

    if !decimal_part.is_empty() {
        words.push_str(" point");
        for c in decimal_part.chars() {
            // spell each digit
            if let Some(d) = c.to_digit(10) {
                 words.push(' ');
                 words.push_str(unit_word(d as u16));
            }
        }
    }

    if currency == "$" {
        // verify plural
        if int_val == 1 && decimal_part.is_empty() {
             words.push_str(" dollar");
        } else {
             words.push_str(" dollars");
        }
    }

    words
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
    ];

    let mut parts: Vec<String> = Vec::new();
    let mut remaining = value;
    let mut scale_idx = 0;

    while remaining > 0 {
        let group = (remaining % 1000) as u16;
        if group > 0 {
            let mut part = words_under_1000(group);
            let scale = scales.get(scale_idx).unwrap_or(&"");
            if !scale.is_empty() {
                part.push(' ');
                part.push_str(scale);
            }
            parts.push(part);
        }
        remaining /= 1000;
        scale_idx += 1;
    }

    parts.reverse();
    if parts.len() > 1 {
        let last_group = (value % 1000) as u16;
        if last_group > 0 && last_group < 100 {
            if let Some(last) = parts.pop() {
                parts.push(format!("and {last}"));
            }
        }
    }

    parts.join(" ")
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
    fn normalize_english_adds_prefix_and_spells_numbers() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter("I have 2 dogs.", &counter, true, true)?;
        assert_eq!(out, vec!["<|en|>I have two dogs ."]);
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
        let out = text_normalize_english_with_counter("It is 3.14 value.", &counter, true, true)?;
        assert_eq!(out, vec!["<|en|>It is three point one four value ."]);
        Ok(())
    }

    #[test]
    fn test_currency() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter("Costs $5.", &counter, true, true)?;
        assert_eq!(out, vec!["<|en|>Costs five dollars ."]);

        let out2 = text_normalize_english_with_counter("$1 price.", &counter, true, true)?;
        assert_eq!(out2, vec!["<|en|>one dollar price ."]);
        Ok(())
    }

    #[test]
    fn test_punctuation_spacing() -> Result<()> {
        let counter = |t: &str| Ok(t.split_whitespace().count());
        let out = text_normalize_english_with_counter("Hello,world!", &counter, true, true)?;
        assert_eq!(out, vec!["<|en|>Hello , world !"]);
        Ok(())
    }
}
