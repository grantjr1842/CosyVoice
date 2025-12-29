# Session Summary: Rust Backend Migration

## Accomplishments
- **Phase 1 (HiFT Vocoder)**: Successfully implemented in Rust using Candle. Includes custom ISTFT and SineGen logic.
- **Phase 2 (Flow Matching)**: Successfully implemented in Rust. Architecture mirrors `DiT` (Diffusion Transformer) with Adaptive LayerNorm.
- **CUDA Enablement**:
    - Resolved environment conflicts (GCC 13 vs NVCC 12.1) by correctly setting `PATH` to `/usr/local/cuda-13.1/bin`.
    - Added `cuda` feature to `rust/server/Cargo.toml` with optional `candle-kernels`.
    - Note: CUDA 13.1 is not supported by cudarc; CPU-only builds work.
- **PyO3 Integration & Verification**:
    - Achieved L1 numerical parity (< 1e-4) between Rust and Python Flow implementations using `tests/verify_flow_rust.py`.
    - Robustified `lib.rs` constructors to handle various weight prefixes and both `.safetensors` and `.pt` formats.
    - Fixed `TimestepEmbedding` to support both `time_mlp.0/2` (PyTorch naming) and `linear_1/2` (test naming).
- **Model Conversion**:
    - Successfully converted `flow.pt`, `hift.pt`, and `llm.pt` to `.safetensors`.
    - Handled shared weights in `llm.pt` by cloning to avoid `safetensors` serialization errors.
- **End-to-End Server**:
    - Simplified `tts.rs` to use the full CosyVoice3 Python model for proper pipeline orchestration.
    - Server starts successfully and synthesizes audio.
    - Performance after warmup: RTF ~1.27 (near real-time).

## Current State
- **Build Status**: Rust server binary (`cosyvoice-server`) builds and runs successfully.
- **Code State**:
    - `flow.rs`: Fully parity-checked with `TimestepEmbedding` supporting dual naming conventions.
    - `lib.rs`: Updated with robust weight loading and prefix mapping (uses `llm.model` for LLM weights).
    - `tts.rs`: Simplified to use CosyVoice3 Python model for proper LLM→Flow→HiFT pipeline.
- **Environment**: CUDA 13.1 + GCC 13.3 (CPU-only build due to cudarc compatibility).

## Instructions for Next Agent
- **Inference**: Use the `/synthesize` endpoint with `prompt_audio` and `prompt_text` for zero-shot cloning.
- **Benchmarking**: Run multiple requests to allow torch.compile warmup; RTF ~1.27 after warmup.
- **Verification**: If modifying `flow.rs`, always run `tests/verify_flow_rust.py`.
- **Build Command**: `pixi run cargo build --release -p cosyvoice-server` (no CUDA features until cudarc supports CUDA 13.1).
