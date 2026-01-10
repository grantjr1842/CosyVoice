# Parity Harness (Python vs Native)

This harness captures Python (PyO3) intermediate tensors and runs the native Rust parity checks against them. It is designed to make mismatches visible at the frontend, Flow, and HiFT stages with deterministic inputs.

## What it produces

The harness writes artifacts and logs into a date-stamped directory under `output/benchmarks/YYYY-MM-DD/` (unless you pass a custom path).

- `frontend_artifacts.safetensors`
  - `speech_tokens`: Python ONNX tokenizer output (prompt audio -> speech tokens)
  - `speaker_embedding`: Python campplus embedding
  - `speech_feat`: Python 24k mel features for Flow prompt
  - `whisper_mel`: Python Whisper log-mel (16k) for tokenizer
- `debug_artifacts.safetensors`
  - `text`, `prompt_text`: normalized text tokens
  - `llm_prompt_speech_token`, `llm_embedding`
  - `token`, `prompt_token`, `prompt_feat`, `embedding` (Flow inputs)
  - `python_flow_output`
  - `python_audio_output`, `python_hift_source`
  - `rand_noise` (Flow noise for deterministic parity)
- `debug_artifacts.wav` (Python HiFT output)
- `check_frontend.log`, `test_flow.log`, `test_hift.log`

## How to run

```bash
tools/run_parity_harness.sh
```

If you already have Python artifacts and want to rerun only the Rust checks:

```bash
tools/run_rust_parity_only.sh
```

Optional environment overrides:

```bash
MODEL_DIR=/path/to/Fun-CosyVoice3-0.5B \
PROMPT_WAV=/path/to/prompt.wav \
PROMPT_TEXT="Your prompt text" \
TTS_TEXT="Your tts text" \
tools/run_parity_harness.sh /custom/output/dir
```

## Notes

- The harness uses the same Python code paths as the PyO3 bridge (CosyVoice3 + frontend).
- The Rust checks use deterministic Flow noise from `debug_artifacts.safetensors` when available.
- The TTS text is normalized in Python and only a single segment is used (segment index 0). You can change this in `debug_scripts/generate_fresh_artifacts.py`.

## Latest results (2026-01-10)

- Frontend parity: Whisper mel length 633 with L1 avg ~1e-6, speech tokens 100% match (159/159) after aligning resampler to torchaudio.

## Related files

- `debug_scripts/dump_frontend.py`
- `debug_scripts/generate_fresh_artifacts.py`
- `rust/native-server/src/bin/check_frontend.rs`
- `rust/native-server/src/bin/test_flow.rs`
- `rust/native-server/src/bin/test_hift.rs`
- `tools/run_parity_harness.sh`
