# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fun-CosyVoice3 is a state-of-the-art text-to-speech (TTS) system based on large language models (LLM). It supports zero-shot voice cloning across 9 languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian) and 18+ Chinese dialects/accents.

### Architecture

The project has **two implementations** of the same TTS system:

1. **Python Implementation** (`cosyvoice/`):
   - PyTorch-based reference implementation
   - Primary development environment
   - Used for training and experimentation
   - Located in `cosyvoice/` directory

2. **Rust Native Implementation** (`rust/native-server/`):
   - High-performance production server
   - Uses Candle (Rust ML framework) and ONNX Runtime
   - Zero Python dependencies at runtime
   - Located in `rust/native-server/src/`

### Core Model Components

The TTS pipeline consists of three main stages:

1. **LLM Stage** (`cosyvoice/llm/`, `rust/native-server/src/cosyvoice_llm.rs`):
   - Qwen-based language model that converts text to semantic tokens
   - Supports streaming inference
   - RL-trained weights (`llm.rl.pt`) loaded by default for better quality

2. **Flow Stage** (`cosyvoice/flow/`, `rust/native-server/src/flow.rs`):
   - Flow matching model for acoustic feature generation
   - Converts semantic tokens to mel-spectrograms
   - Uses DiT (Diffusion Transformer) architecture

3. **HiFi-GAN Vocoder** (`cosyvoice/hifigan/`, `rust/native-server/src/hift.rs`):
   - Neural vocoder that converts mel-spectrograms to audio
   - Final stage of the pipeline

### Data Flow

```
Input Text → LLM → Semantic Tokens → Flow → Mel-Spectrogram → HiFi-GAN → Audio Waveform
          ↑          ↑                    ↑                      ↑
    Text        Flow                  Vocoder              Audio
  Frontend    Decoder               (HiFT)                Output
```

## Development Commands

### Pixi (Recommended)

All development commands should use **Pixi** for environment management:

```bash
# Install dependencies (Python, PyTorch, CUDA, etc.)
pixi install

# Download models
pixi run download-model

# Run example script
pixi run example

# Start web UI
pixi run webui

# Enter development shell
pixi shell
```

**Critical**: Pixi manages the entire Python environment. Do not manually install pip packages - everything should go through `pyproject.toml`.

### Python Development

```bash
# Run inference example
python example.py

# Start web UI (Gradio)
python webui.py --port 8000

# Start FastAPI server
cd runtime/python/fastapi
python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B

# Start gRPC server
cd runtime/python/grpc
python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

### Rust Development

```bash
# Build Rust server (from project root)
cd rust
cargo build --release

# Run native server (from project root to load .env)
./rust/target/release/cosyvoice-native-server

# Run bridge server (PyO3-based, requires Python environment)
./rust/target/release/cosyvoice-bridge-server
```

**Important**: The Rust build is configured via `rust/.cargo/config.toml` to automatically find the Pixi Python environment. You do **not** need to wrap cargo commands with pixi.

### Testing

```bash
# Test Rust server end-to-end
COSYVOICE_SERVER_URL=http://127.0.0.1:3000 \
COSYVOICE_PROMPT_AUDIO=asset/zero_shot_prompt.wav \
COSYVOICE_PROMPT_TEXT="You are a helpful assistant.<|endofprompt|>Greetings, how are you today?" \
python tests/test_rust_server_e2e.py
```

## Critical Implementation Details

### Model Version

This repository **only supports Fun-CosyVoice3-0.5B-2512**. Legacy CosyVoice v1/v2 models have been removed.

### Transformers Version Pin

**CRITICAL**: The `transformers` library is pinned to `==4.51.3` in `pyproject.toml`. Versions 4.54+ break multilingual capabilities, causing garbled audio output (especially for English). Never upgrade this version without thorough testing.

### PyO3 Integration

The Rust server builds with PyO3 to optionally use Python components. The build configuration in `rust/.cargo/config.toml` automatically points PyO3 to the correct Python environment (pixi or system). The `.env` file at the project root configures runtime paths.

### Flash Attention

Flash Attention v2 is supported in the Rust native-server on NVIDIA GPUs with Compute Capability 8.0+ (Ampere and newer). Enabled via the `cuda` feature flag, which includes `candle-transformers/flash-attn`.

Verify it's active:
```bash
cd rust/native-server
cargo tree --features cuda | grep candle-flash-attn
```

### TensorRT (Optional)

TensorRT providers are only enabled when `COSYVOICE_ORT_USE_TRT=1` is set. If your TensorRT libs are not on the default linker path, set `COSYVOICE_TRT_LIB_DIR` to the directory containing `libnvinfer.so.*`.

The `rust/start-server.sh` script will attempt to discover pixi-installed TensorRT libs automatically when the flag is enabled.

## File Structure

### Python Code

- `cosyvoice/cli/` - Main CLI interface and model loading
  - `cosyvoice.py` - Main entry point (CosyVoice3 class, AutoModel)
  - `model.py` - Model initialization and inference orchestration
  - `frontend.py` - Text frontend and tokenization
- `cosyvoice/llm/` - LLM stage implementation
- `cosyvoice/flow/` - Flow matching stage
  - `flow.py` - Flow matching model
  - `decoder.py` - Decoder architecture
- `cosyvoice/hifigan/` - HiFi-GAN vocoder
- `cosyvoice/tokenizer/` - Tokenization utilities
- `cosyvoice/utils/` - Utilities (gpu_optimizer, file_utils, etc.)

### Rust Code

- `rust/native-server/src/` - Native server implementation
  - `main.rs` - Server entry point and HTTP API
  - `tts.rs` - TTS request handling and orchestration
  - `cosyvoice_llm.rs` - LLM inference (Qwen model)
  - `flow.rs` - Flow matching model
  - `hift.rs` - HiFi-GAN vocoder
  - `audio.rs` - Audio processing utilities
  - `onnx_frontend.rs` - ONNX-based text frontend
  - `text_frontend.rs` - Text processing
  - `qwen.rs` - Qwen model implementation
  - `utils.rs` - Utilities
- `rust/bridge-server/` - PyO3-based bridge server (uses Python backend)
- `rust/shared/` - Shared types and utilities
- `rust/client/` - CLI client for testing

### Configuration

- `pyproject.toml` - Pixi configuration, dependencies, and tasks
- `rust/Cargo.toml` - Rust workspace configuration
- `rust/.cargo/config.toml` - Cargo build configuration (PyO3 Python path)
- `.env` - Environment variables for Rust server

## Inference Modes

Fun-CosyVoice3 supports three inference modes:

1. **Zero-Shot** (`inference_zero_shot`): Clone voice from short audio clip
2. **Cross-Lingual** (`inference_cross_lingual`): Synthesize text in different language than prompt
3. **Instruction** (`inference_instruct2`): Control voice style with natural language instructions

### Language Instructions

For zero-shot voice cloning, add explicit language instructions to avoid language confusion:
```python
prompt_text = "You are a helpful assistant. Please speak in English.<|endofprompt|>" + reference_text
```

## Troubleshooting

### Common Issues

1. **Garbled Audio**: Ensure `transformers==4.51.3` - newer versions break multilingual support
2. **Language Confusion**: Add explicit language instructions in prompt text
3. **Audio Quality**: Default settings use RL-trained LLM, `top_p=0.7`, and optimized vocoder for best quality

### Debugging

The repository includes extensive debugging utilities in `tests/`:
- `compare_intermediates.py` - Compare Python vs Rust intermediate outputs
- `debug_hift.py` - Debug HiFi-GAN vocoder issues
- `inspect_artifacts.py` - Inspect model artifacts

## Code Architecture Patterns

### Python-Rust Parity

The Rust implementation is designed to maintain parity with the Python implementation. When modifying model code:

1. **Python First**: Make changes in Python first and verify
2. **Rust Port**: Port changes to Rust, maintaining the same architecture
3. **Validation**: Use `tests/compare_intermediates.py` to verify parity

### Model Loading

Models are loaded from `pretrained_models/Fun-CosyVoice3-0.5B/`:
- `llm.pt` or `llm.rl.pt` - LLM weights (RL version preferred)
- `flow.pt` - Flow matching weights
- `hift.pt` - HiFi-GAN vocoder weights
- `cosyvoice3.yaml` - Model configuration

### GPU Optimization

The Python implementation includes automatic GPU optimization (`cosyvoice/utils/gpu_optimizer.py`):
- Auto-detects GPU capabilities
- Enables TF32 for Ampere+ GPUs
- Configures MatMul precision
- Suggests FP16 vs FP32 based on hardware

## Deployment

### Docker

```bash
cd runtime/python
docker build -t cosyvoice:v3.0 .
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v3.0
```

### Production

For production deployments, use the Rust native server for best performance:
- Lower memory footprint
- Faster inference
- No Python runtime dependency
- Support for streaming inference
