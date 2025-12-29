# Critical Development Protocols & Verification Processes

## 1. Rust <-> Python Parity Verification

**Objective**: Ensure the Rust implementation of Flow Matching achieves L1 numerical parity (< 1e-4) with the Python reference.

### Critical Workflow
Any changes to `rust/server/src/flow.rs` MUST be followed by this verification process before committing:

1.  **Build the Rust Library**:
    You must build the shared library with the correct OpenSSL environment variables provided by `pixi`.
    ```bash
    OPENSSL_DIR=$CONDA_PREFIX \
    OPENSSL_LIB_DIR=$CONDA_PREFIX/lib \
    OPENSSL_INCLUDE_DIR=$CONDA_PREFIX/include \
    PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig \
    RUST_BACKTRACE=1 \
    pixi run cargo build --manifest-path rust/Cargo.toml
    ```

2.  **Update the Shared Object**:
    Copy the debug build to the root directory for the Python script to load.
    ```bash
    cp rust/target/debug/libcosyvoice_rust_backend.so cosyvoice_rust_backend.so
    ```

3.  **Run Verification Script**:
    Execute the Python verification script that compares per-layer outputs.
    ```bash
    pixi run python3 tests/verify_flow_rust.py
    ```

4.  **Success Criteria**:
    - The script prints `L1 Error: <value>`.
    - **PASS**: L1 Error < `0.002` (approximate float tolerance).
    - **FAIL**: Any higher error indicates a logic divergence.

## 2. Common Pitfalls & Findings

### AdaLayerNormZeroFinal Variable Ordering
**Critical Finding**: Use caution when porting `AdaLayerNormZero` variants.
- **Python**: `AdaLayerNormZero_Final` unpacks modulation chunks as `scale, shift`.
- **Rust**: Standard implementation often unpacks as `shift, scale`.
- **Resolution**: The Rust implementation for `AdaLayerNormZeroFinal` MUST use:
  ```rust
  let chunks = emb.chunk(2, 1)?;
  let (scale, shift) = (&chunks[0], &chunks[1]); // ORDER MATTERS
  ```

### Tensor Layouts
- **Python**: Generally `[Batch, Sequence, Dim]`.
- **Rust (Candle)**: Often treats `[Batch, Dim, Sequence]` for Conv1d operations efficiently, but Transformer blocks expect `[Batch, Sequence, Dim]`.
- **Verification**: Always verify `.transpose(1, 2)` calls when porting code involving Convolutions or Linear layers that expect different input shapes.

## 3. Environment Dependencies
The Rust build heavily relies on the `pixi` environment.
- **Never** run `cargo build` outside of `pixi run` or without sourcing the necessary OpenSSL environment variables.
- The `cosyvoice_rust_backend.so` is required for the verification script.

## 4. Debugging Divergence
If parity checks fail:
1.  **Instrument Python**: Add `register_forward_hook` to `DiT` blocks in `tests/verify_flow_rust.py`.
2.  **Instrument Rust**: Add `eprintln!` in `flow.rs` to print mean/first-5 elements of tensors.
3.  **Pointwise Comparison**: Compare `x_norm`, `shift`, and `scale` values inside normalization layers. These are often the source of subtle math errors (like the swap bug identified above).
