# Developer Onboarding Guide

## Prerequisites
- **Rust Toolchain**: Stable channel (install via [rustup](https://rustup.rs/)).
- **CUDA Toolkit** (Optional): For GPU acceleration (recommended 11.8+ or 12.x).
- **Python**: For downloading models and running comparison scripts.

## Setup
### 1. Clone Repository
```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
```

### 2. Download Models
Ensure pretrained models are in `pretrained_models/Fun-CosyVoice3-0.5B/`.
Structure:
- `llm.rl.safetensors` or `llm.gguf`
- `flow.safetensors`
- `hift.safetensors`
- `campplus.onnx`
- `speech_tokenizer_v1.onnx`

### 3. Build Rust Crate
Navigate to the Rust directory:
```bash
cd rust/native-server
cargo build --release
```
To build with CUDA support (requires CUDA toolkit):
```bash
cargo build --release --features cuda
```

## Running Examples

### End-to-End Synthesis
Run `native_example` to generate speech:
```bash
cargo run --release --bin native_example
```
This uses default assets and generates `native_voice_clone_*.wav` in `output/`.

### Benchmarking
Measure Real-Time Factor (RTF):
```bash
cargo run --release --bin benchmark_tts -- --model-dir ../../pretrained_models/Fun-CosyVoice3-0.5B --prompt-wav ../../asset/interstellar-tars-01-resemble-denoised.wav
```

## Common Tasks

### Adding Features
- Modify `src/tts.rs` for high-level pipeline changes.
- Modify `src/cosyvoice_*.rs` for model-specific logic.

### Troubleshooting
- **DType Mismatches**: Ensure inputs are cast to `engine.dtype` (F16 on CUDA). Use `Tensor::to_dtype(DType::F16)?`.
- **Model Not Found**: Check paths. `native_example` expects models relative to `CARGO_MANIFEST_DIR`.

## Testing
Run parity tests:
```bash
cargo run --bin test_llm_parity
cargo run --bin test_flow_parity
# etc.
```
