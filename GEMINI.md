# CosyVoice-1 (Fun-CosyVoice3)

## Project Overview
Fun-CosyVoice3 is an advanced text-to-speech (TTS) system based on large language models (LLM) and Flow Matching. It supports zero-shot multilingual speech synthesis, cross-lingual voice cloning, and instruction-based control.

The project is a hybrid **Python** (research, training, UI) and **Rust** (high-performance inference) codebase.

## 🛠 Environment & Setup
This project uses **Pixi** as the primary package manager. It handles Python dependencies, system libraries (CUDA, OpenSSL), and the environment for building the Rust components.

### Initial Setup
```bash
# 1. Install dependencies
pixi install

# 2. Download pretrained models
pixi run download-model
```

## 📂 Project Structure

### Core Directories
- **`cosyvoice/`**: Main Python source code.
    - `bin/`: Scripts for export and training.
    - `cli/`: CLI entry points.
    - `flow/`, `llm/`, `hifigan/`: Model components.
- **`rust/`**: High-performance Rust backend.
    - `bridge-server/`: PyO3-based bridge using Python backend.
    - `native-server/`: Pure Rust inference server (Candle/ONNX).
    - `client/`: Testing client.
- **`pretrained_models/`**: Stores downloaded model weights (e.g., `Fun-CosyVoice3-0.5B`).
- **`tests/`**: Verification scripts, specifically for Python/Rust parity.
- **`webui.py`**: Entry point for the Gradio Web UI.

## 🚀 Key Commands

### Python / General
**Note:** Always prefix commands with `pixi run` to ensure the correct environment.

- **Start Web UI:** `pixi run webui` (Runs on port 8000)
- **Run Example:** `pixi run example`
- **Export ONNX:** `pixi run export-onnx`
- **Download Models:** `pixi run download-model`

### Rust Development
The Rust components **must** be built within the Pixi environment to link correctly against system libraries (like OpenSSL).

- **Build Release:**
  ```bash
  pixi run cargo build --release
  ```

- **Run Native Server:**
  ```bash
  ./rust/target/release/cosyvoice-native-server
  ```

## ⚠️ Development Protocols

### Rust <-> Python Parity
Strict numerical parity is required between the Python reference implementation and the Rust backend.

**Verification Process:**
1. **Build Rust Lib:**
   ```bash
   pixi run cargo build --manifest-path rust/Cargo.toml
   ```
2. **Copy Shared Object:**
   ```bash
   cp rust/target/debug/libcosyvoice_rust_backend.so cosyvoice_rust_backend.so
   ```
3. **Run Verification:**
   ```bash
   pixi run python3 tests/verify_flow_rust.py
   ```
   **Pass Criteria:** L1 Error < `0.002`.

### Common Pitfalls
*   **AdaLayerNormZeroFinal:** Rust implementation order is `(scale, shift)`, Python unpacks as `(scale, shift)`. Ensure careful handling of chunking.
*   **Tensor Layouts:** Python typically uses `[Batch, Sequence, Dim]`. Rust (Candle) may prefer `[Batch, Dim, Sequence]` for Convolutions. Always verify `.transpose()` calls.
*   **Transformers Version:** `transformers==4.51.3` is pinned in `pyproject.toml`. Newer versions may break multilingual capabilities.

## 🏗 Architecture

### Python Backend
- Uses **Hydra** for configuration.
- **Torch/Candle** for inference.
- **FastAPI** / **Gradio** for serving.

### Rust Backend
- **Candle**: Used for tensor operations and inference in `native-server`.
- **Axum**: Web server framework.
- **PyO3**: Used in `bridge-server` to wrap the Python implementation for testing or compatibility.
