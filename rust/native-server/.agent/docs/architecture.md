# CosyVoice Native Server Architecture

## Overview

High-performance Rust implementation of CosyVoice TTS (Text-to-Speech) system using the Candle ML framework.

## Technology Stack

- **Language:** Rust
- **ML Framework:** Candle (candle-core, candle-nn, candle-transformers)
- **CUDA Support:** candle-kernels, flash-attention
- **Audio Processing:** hound, rubato, rustfft, realfft
- **ONNX Runtime:** ort (for frontend)
- **Tokenization:** tokenizers (HuggingFace)
- **CLI:** clap
- **Web Server:** axum, tower-http

## Project Structure

```
rust/native-server/
├── src/
│   ├── lib.rs              # Library exports
│   ├── main.rs             # HTTP server entrypoint
│   ├── tts.rs              # Main TTS engine orchestration
│   ├── cosyvoice_llm.rs    # LLM wrapper (FP16/GGUF)
│   ├── cosyvoice_flow.rs   # Flow model wrapper
│   ├── flow.rs             # DiT transformer implementation
│   ├── hift.rs             # HiFT vocoder implementation
│   ├── qwen.rs             # Qwen2 FP16 model
│   ├── quantized_qwen.rs   # Qwen2 GGUF quantized model
│   ├── audio.rs            # Audio processing utilities
│   ├── text_frontend.rs    # Text preprocessing
│   ├── onnx_frontend.rs    # ONNX-based frontend
│   ├── utils.rs            # STFT/ISTFT utilities
│   └── bin/                # Test and benchmark binaries
├── tests/                  # Integration tests
└── Cargo.toml
```

## Key Components

### LLM (cosyvoice_llm.rs)
- Supports both FP16 SafeTensors and GGUF quantized models
- Auto-detects GGUF via `COSYVOICE_LLM_GGUF` env var
- Contains speech_embedding and llm_decoder layers

### Flow (flow.rs)
- DiT (Diffusion Transformer) implementation
- Uses rotary position embeddings (RoPE)
- Supports streaming with chunk masking
- Flash attention via candle SDPA

### HiFT Vocoder (hift.rs)
- F0 predictor for pitch estimation
- Source module for harmonic generation
- ISTFT for waveform synthesis
- Causal convolutions for streaming support

## Debugging Tips

### Enable verbose Flow debug
```bash
FLOW_DEBUG_LOG_V=1 cargo run --bin test_flow
```

### Force FP16 mode (skip GGUF)
```bash
cargo run --bin benchmark_tts -- --fp16
```

### Debug tensor artifacts
Test binaries save `.safetensors` files for parity testing with Python.

## Common Patterns

### Weight loading with WeightNorm
The codebase handles PyTorch weight normalization in multiple formats:
- `weight_g` / `weight_v` (older)
- `weight.original0` / `weight.original1` (parametrizations)
- `parametrizations.weight.original0` / `...original1`

### CUDA device selection
```rust
let device = if candle_core::utils::cuda_is_available() {
    Device::new_cuda(0)?
} else {
    Device::Cpu
};
```

### Dtype handling
- Flow model: F32 for stability (F16 causes NaN in softmax)
- HiFT: Forces F32 internally via `mel.to_dtype(DType::F32)`
- GGUF models: Dequantize weights per-layer
