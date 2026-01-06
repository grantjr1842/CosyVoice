# CosyVoice Native Server - Clippy & Code Quality Fixes

## Session Summary (2026-01-06)

This session fixed all IDE errors and clippy warnings, and cleaned up verbose debug output.

## Issues Fixed

### 1. Missing Dependency
- **File:** `Cargo.toml`
- **Fix:** Added `clap = { version = "4", features = ["derive"] }` to dependencies
- **Affected bins:** `benchmark_tts`, `test_flow`, `test_hift_parity`

### 2. Unused Imports
| File | Import Removed |
|------|----------------|
| `src/bin/test_flow.rs` | `Tensor` |
| `src/bin/test_hift_stages.rs` | `std::collections::HashMap` |
| `src/quantized_qwen.rs` | `IndexOp` |

### 3. Unused Variables
| File | Variable | Fix |
|------|----------|-----|
| `src/bin/test_cuda_ops.rs` | `t_u32_gpu`, `t_f32_gpu` | Prefixed with `_` |

### 4. Clippy Warnings
| File:Line | Issue | Fix |
|-----------|-------|-----|
| `flow.rs:403` | unnecessary cast `as f64` | Removed cast (self.scale is already f64) |
| `flow.rs:835` | too_many_arguments | Added `#[allow(clippy::too_many_arguments)]` |
| `hift.rs:135` | needless_range_loop | Converted to iterator with `.enumerate()` |
| `hift.rs:182` | too_many_arguments | Added `#[allow(clippy::too_many_arguments)]` |
| `hift.rs:636` | if_same_then_else | Replaced `if is_causal { 0 } else { 0 }` with `0` |
| `test_f0_parity.rs:44,49` | needless_borrow | Removed `&` from `&mel` |

## Debug Output Cleanup

Removed **13 verbose per-inference debug prints** from `hift.rs`:

### F0Predictor::forward
- Removed per-layer ELU activation stats
- Removed classifier output stats

### HiFTGenerator::forward
- Removed input mel shape/stats logging
- Removed F0 predictor output stats
- Removed upsampling debug
- Removed source module stats
- Removed final audio stats and issue warnings

### decode method
- Removed dead debug blocks computing `_mean_si`, `_mean_x`, `_min`, `_max`
- Cleaned up duplicate "Remove DC offset" comments

## Verification

```bash
# All pass
cargo clippy --all-targets -- -D warnings
cargo build --release
```

## Remaining eprintln! Count

| File | Count | Type |
|------|-------|------|
| `flow.rs` | 11 | RoPE debug (env-gated), sdpa fallback warning |
| `hift.rs` | 5 | Load-time conv weight stats |
| `cosyvoice_llm.rs` | 10 | Load-time model loading messages |
| `cosyvoice_flow.rs` | 21 | Load-time and debug tensor saving |

These are acceptable load-time diagnostics, not hot-path inference spam.
