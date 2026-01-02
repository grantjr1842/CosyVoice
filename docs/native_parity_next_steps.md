# Native Parity Next Steps (start here)

This file captures the next actionable steps so a new conversation can resume in order.

## Step 1 - LLM sampling and stop tokens parity

Goal: Match Python LLM sampling behavior to reduce overlong outputs.

Targets:
- Add top-p sampling (p=0.8) in addition to top-k (k=25).
- Implement stop_token_ids logic (as in `cosyvoice/llm/llm.py`).
- Mirror prompt text concat and min/max length rules used in Python.
- Align `llm.safetensors` vs `llm.rl.pt` behavior if possible.

Primary refs:
- `cosyvoice/llm/llm.py` (search for `sampling_ids`, `stop_token_ids`, `min_token_text_ratio`, `max_token_text_ratio`)
- `rust/native-server/src/cosyvoice_llm.rs`
- `pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml`

Suggested work:
- Add a top-p path to `CosyVoiceLLM::sample_top_k` or a new sampler.
- Implement EOS checks consistent with Python `stop_token_ids`.
- Add a small parity test that compares token lengths for the same text.

## Step 2 - Text frontend parity

Goal: Normalize and split text the same way as Python before tokenization.

Targets:
- Match `CosyVoiceFrontEnd.text_normalize` behavior for English.
- Support `<|en|>` prefix and paragraph splitting by token count.
- Keep the same prompt prefix behavior with `<|endofprompt|>`.

Primary refs:
- `cosyvoice/cli/frontend.py` (`text_normalize`, `split_paragraph`)
- `cosyvoice/utils/frontend_utils.py`
- `rust/native-server/src/bin/native_example.rs`

Suggested work:
- Port minimal English normalization and splitting logic first.
- Add a unit test that compares normalized outputs against Python.

## Step 3 - Audio parity verification

Goal: Quantify audio similarity between Python and native outputs.

Targets:
- Generate output pairs using the same example inputs.
- Compare durations, RMS, SNR, and alignment.
- Use the same reference prompt and texts as `example.py`.

Primary refs:
- `example.py`
- `compare_audio.py`
- `verify_audio.py`
- `output/voice_clone_*.wav` and `output/native_voice_clone_*.wav`

Suggested commands:
- `pixi run example`
- `cargo run -p cosyvoice-native-server --bin native_example --release --features cuda`
- `pixi run python compare_audio.py output/voice_clone_0_0.wav output/native_voice_clone_0_0.wav`
