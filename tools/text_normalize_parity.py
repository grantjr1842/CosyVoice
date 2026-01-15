#!/usr/bin/env python3
import json
import sys

import inflect
from tokenizers import Tokenizer

from cosyvoice.utils.frontend_utils import (
    is_only_punctuation,
    spell_out_number,
    split_paragraph,
)


def normalize_english(text, tokenizer, split, text_frontend):
    if text_frontend is False or text == "":
        return [text] if split else [text]

    if "<|" in text and "|>" in text:
        return [text] if split else [text]

    text = text.strip()
    if text == "":
        return [""]

    # Try to use wetext first for true parity
    try:
        from wetext import Normalizer
        normalizer = Normalizer(lang="en")
        text = normalizer.normalize(text)
    except ImportError:
        # Fallback to manual simulation of WeText/Rust behavior
        parser = inflect.engine()
        # Custom logic to match Rust implementation
        text = manual_normalization(text, parser)

    if not split:
        return [text]

    tokens = split_paragraph(
        text,
        lambda t: tokenizer.encode(t, add_special_tokens=True).ids,
        "en",
        token_max_n=80,
        token_min_n=60,
        merge_len=20,
        comma_split=False,
    )

    tokens = [t for t in tokens if not is_only_punctuation(t)]
    return tokens

def manual_normalization(text, parser):
    # TODO: Implement full fallback if wetext is missing
    # For now, just use what we had but this will likely fail parity against improved Rust
    # Ideally this script should run in an env with wetext
    from cosyvoice.utils.frontend_utils import spell_out_number
    return spell_out_number(text, parser)



def main():
    payload = json.load(sys.stdin)
    text = payload["text"]
    tokenizer_path = payload["tokenizer_path"]
    split = payload.get("split", True)
    text_frontend = payload.get("text_frontend", True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    segments = normalize_english(text, tokenizer, split, text_frontend)
    lengths = [len(tokenizer.encode(s, add_special_tokens=True).ids) for s in segments]

    json.dump({"segments": segments, "token_lengths": lengths}, sys.stdout)


if __name__ == "__main__":
    main()
