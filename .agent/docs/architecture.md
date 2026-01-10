# CosyVoice Architecture

## System Overview
CosyVoice is a high-fidelity, zero-shot text-to-speech (TTS) system. This repository contains both the original Python research codebase and a high-performance Rust native server for inference.

## Technology Stack
- **Languages**: Python (Research/Training/Reference), Rust (Inference/Deployment).
- **Core Libraries**:
    - **Python**: PyTorch, Hydra, Gradio, WeText.
    - **Rust**: Candle (ML framework), Axum (Web server), PyO3 (Python/Rust bridge).
- **Environment**: Pixi (Package management).
- **Deployment**: Supports vLLM (version >= 0.11.0, V1 engine) for high-performance usage.

## Project Structure
```text
project-root/
├── cosyvoice/           # Core Python source code
│   ├── llm/             # LLM (Transformer/Qwen) components
│   ├── flow/            # Flow Matching model
│   ├── hifigan/         # HiFT/HiFiGAN vocoder
│   └── cli/             # Command-line interfaces
├── rust/                # Rust Native Server
│   ├── native-server/   # Main axum-based inference server
│   │   ├── src/bin/     # Binaries (server, tools)
│   │   └── src/llm/     # Rust LLM implementations
│   └── bridge-server/   # PyO3 bridge for testing/legacy
├── pretrained_models/   # Downloaded model weights (safetensors, gguf)
└── runtime/             # Export runtimes (Triton, ONNX, etc.)
```

## Core Components
### 1. LLM (Large Language Model)
- **Role**: Generates semantic tokens from text input and speaker prompts.
- **Hierarchy**:
    - `CosyVoiceModel`: Base class for V1.
    - `CosyVoice2Model`: Inherits from base, adds support for V2 features.
    - `CosyVoice3Model`: Inherits from `CosyVoice2Model`, adds instruction following tokens.
- **Implementation**:
    - Python: `TransformerLM` / `Qwen2ABITLM`.
    - Rust: `CosyVoiceLLM` using `candle-transformers` (Qwen).
- **Parity**: Rust implementation validated against Python for token probability distribution.

### 2. Flow Matching
- **Role**: Predicts Mel-spectrogram features from LLM semantic tokens using Conditional Flow Matching (CFM).
- **Implementation**:
    - Python: `MaskedDiff` / `ConditionalFlowMatchingModel`.
    - Rust: `CosyVoiceFlow` using ODE solvers (Euler/RK4).

### 3. HiFT Vocoder
- **Role**: Converts Mel-spectrograms into time-domain audio waveforms.
- **Implementation**:
    - Source Generation: Generates F0 and sine-based source signals.
    - Neural Filter: Refines the source into high-fidelity speech.

## Key Design Patterns
### Hybrid Development & Parity
- **Pattern**: Features are prototyped in Python and ported to Rust.
- **Verification**: Dedicated parity tests (e.g., `test_llm_parity.rs`) capture intermediate tensors from Python (`debug_scripts/`) and strictly compare them against Rust outputs to ensure numerical exactness.

### Configuration
- **Python**: Uses `Hydra` (YAML) for flexible experiment configuration.
- **Rust**: Uses strongly typed structs via `serde` to parse model configurations.
