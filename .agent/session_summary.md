# Session Summary: Advanced Optimizations Implementation

## Date: 2025-12-27

## Objective
Enable advanced optimization packages (bitsandbytes, accelerate, flash-attn) for CosyVoice3 TTS.

## Completed Tasks

### 1. Package Installation ✅
- Installed bitsandbytes 0.49.0
- Installed accelerate 1.12.0
- Installed flash-attn 2.8.3

### 2. Dependency Updates ✅
- Updated `pyproject.toml` with new pypi-dependencies
- Updated `requirements.txt` for legacy compatibility
- Relaxed transformers to `>=4.54.0` for huggingface-hub compatibility

### 3. GPU Compatibility Fixes ✅
- **FlashAttention-2**: Added compute capability check - only enabled on Ampere (8.0+) GPUs
- **SDPA Fallback**: Turing (7.x) and older automatically use PyTorch's native SDPA
- **Quantization**: Made conservative - only triggers for <4GB VRAM GPUs

### 4. Logging Improvements ✅
- Added optimization summary log at model load time
- Clear messages showing which attention implementation is active

## GitHub Artifacts
- Issue: #13
- PR: #14 (merged)
- Branch: `agent/task-13` (deleted after merge)

## Verification Results
Tested on RTX 2070 (Turing, 8GB VRAM):
- SDPA attention enabled ✅
- FP16 mode active (no quantization) ✅
- TensorRT engine loaded ✅
- Inference RTF: ~1.1-2.1 ✅

---

# Session Summary: CosyVoice3 Documentation & PyTorch Fixes

## Date: 2025-12-28

## Objective
Document CosyVoice3 requirements, configure environment via .env, and fix PyTorch deprecation warnings.

## Important: CosyVoice3 `prompt_audio` Requirement

> **CosyVoice3 (Fun-CosyVoice3-0.5B) requires `prompt_audio` for ALL synthesis requests.**

Unlike CosyVoice v1/v2, there is **no SFT (Speaker Fine-Tuning) mode** with pre-trained speaker embeddings. You must always provide a reference audio file for voice cloning. The server enforces this requirement and returns an error if `prompt_audio` is not provided.

Supported synthesis modes:
- **Zero-shot voice cloning**: Provide `prompt_audio` + `prompt_text` + `text`
- **Instruction-based synthesis**: Provide `prompt_audio` + `instruct_text` + `text`

## Completed Tasks

### 1. Environment Configuration ✅
- Created `.env` file with `LD_LIBRARY_PATH_EXTRA` configuration
- Updated `rust/start-server.sh` to source `.env` file
- Library paths are now externalized and configurable

### 2. PyTorch Deprecation Fixes ✅
- Fixed `torch.load` FutureWarning by adding `weights_only=True` parameter (3 locations in model.py)
- Fixed `torch.cuda.amp.autocast` deprecation by using new `torch.amp.autocast("cuda", ...)` API (2 locations in model.py)
- All warnings are **properly fixed**, not suppressed

### 3. Documentation ✅
- Documented `prompt_audio` requirement for CosyVoice3
- Documented `.env` configuration approach

## GitHub Artifacts
- Master Issue: #21
- Implementation Issue: #22
- Branch: `agent/task-22`

---

# Session Summary: .env Loading & Output Organization

## Date: 2025-12-28

## Objective
Load .env from Rust server (not shell script) and organize output files into dedicated directory.

## Completed Tasks

### 1. .env Loading from Rust ✅
- Added `dotenvy` crate to Rust workspace
- Server now loads `.env` at startup before any other initialization
- Removed shell-based `.env` sourcing from `start-server.sh`
- Kept `LD_LIBRARY_PATH` in shell (required before binary loads for libpython linking)

### 2. Output File Organization ✅
- Created `output/` directory with `.gitkeep` and README.md
- Updated `example.py` to save generated audio to `output/` directory
- Updated `.gitignore` to handle output directory properly

## GitHub Artifacts
- Issue: #24 (closed by PR)
- PR: #25 (merged)

---

# Session Summary: Python Upgrade & Build Fixes

## Date: 2025-12-28

## Objectives
- Upgrade to Python 3.12 to match system and resolve SRE module mismatch
- Fix "hacky" PyO3 build process (remove `pixi run cargo build` requirement)
- Document lib.rs default model directory update

## Completed Tasks

### 1. Python 3.12 Upgrade ✅
- Upgraded `pyproject.toml` to use Python 3.12
- Reinstalled pixi environment to match system Python version
- **Verification**: `python --version` now returns 3.12.x in pixi env

### 2. PyO3 Build Configuration ✅
- Created `rust/.cargo/config.toml` setting `PYO3_PYTHON` to point to pixi's Python
- This enables standard `cargo build` commands to work correctly without wrappers
- Eliminates the SRE/version mismatch error during PyTorch compilation

### 3. Model Directory Update ✅
- Updated `rust/shared/src/lib.rs` default model to `Fun-CosyVoice3-0.5B`
- This ensures the server uses the correct locally downloaded model by default

### 4. Documentation ✅
- Updated `README.md` with:
    - Python 3.12 requirement
    - Rust server build instructions (native, no pixi wrapper)
    - Note on `ttsfrd` incompatibility with Python 3.12
- Created `rust/README.md` for local development guidance
