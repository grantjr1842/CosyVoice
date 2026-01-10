# Developer Onboarding Guide

## Prerequisites
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Conda/Pixi**: This project uses `pixi` for environment management.
- **GPU**: NVIDIA GPU with CUDA drivers installed (Compute Capability 7.0+ recommended).

## Setup
### 1. Clone Repository
```bash
git clone https://github.com/FunAudioLLM/CosyVoice
cd CosyVoice
```

### 2. Install Dependencies
Initialize the environment using Pixi:
```bash
pixi install
```
This installs Python, PyTorch (CUDA), Rust toolchain, and system libraries.

### 3. Download Models
Download the pretrained CosyVoice models:
```bash
pixi run download-model
```

## Running the System
### Python Web UI
Start the Gradio interface:
```bash
pixi run webui
```
Access at `http://localhost:8000`.

### Rust Native Server
Build and run the high-performance inference server:
```bash
# Build
pixi run cargo build --release --manifest-path rust/native-server/Cargo.toml

# Run
pixi run cargo run --release --manifest-path rust/native-server/Cargo.toml
```

## Development Workflow
### Testing
- **Python**: `pixi run pytest`
- **Rust**: `pixi run cargo test --manifest-path rust/native-server/Cargo.toml`
- **Parity Verification**: Run `debug_scripts/` (Python) followed by `rust/native-server/src/bin/test_*.rs` binaries to verify numerical parity.

### Common Issues
- **FlashAttention Build Error**: If running on older GPUs (e.g., T4/Pascal), ensure `flash-attn` is disabled in `Cargo.toml` or `TORCH_CUDA_ARCH_LIST` is set correctly.
- **Missing Weights**: Ensure `pretrained_models/` is populated using the download script.
