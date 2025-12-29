# Research Log - TTS Audio and CUDA Issues

## Findings

### 1. ONNX Runtime (ORT) Hang/OOM
- **Issue**: `SessionBuilder::commit_from_memory` hangs when loading the `speech_tokenizer_v3.onnx` (969MB) on CPU.
- **Cause**: High initial memory consumption during model initialization. Without GPU acceleration, ORT on CPU may exceed available memory or hang during optimization.
- **Solution**: Re-enabling GPU providers (CUDA/TensorRT) is essential for large models. If CPU must be used, disabling optimizations or using `mimalloc` might help.

### 2. Missing `libcublasLt.so.12`
- **Issue**: `ort` fails to load the CUDA execution provider because `libcublasLt.so.12` is not found.
- **Cause**: This is a core CUDA 12 library. Pixi installs dependencies in `.pixi/envs/default`, but the system loader doesn't know about them unless `LD_LIBRARY_PATH` is correctly set.
- **Action**: Locate the library within the `.pixi` environment and ensure it's in the search path.

### 3. Candle CUDA Detection
- **Issue**: `candle_core::utils::cuda_is_available()` was returning false.
- **Cause**: Missing `cuda` feature flag during `cargo run`.
- **Solution**: Use `--features cuda`. This is already verified to work.

### 4. Audio Garble
- **Issue**: Garbled output audio.
- **Cause**: Redundant `repeat_interleave` in `cosyvoice_flow.rs` caused double upsampling of mel features.
- **Solution**: Removed the redundant call. Verified theoretically, needs final audio validation once the engine runs.

### 5. CUDA Device-Side Assert
- **Issue**: `Assertion `ids[id_i] < src_dim_size` failed` in `PreLookaheadLayer::forward`.
- **Cause**: Device mismatch between test runner and engine, or incorrect padding/upsampling logic.
- **Solution**: Ensured consistent device selection. Fixed the upsampling logic.

## Next Steps
- Finalize the implementation plan.
- Resolve ORT library paths.
- Run end-to-end verification.
