Chat export — CosyVoice session
Date: 2026-01-04
Workspace: /home/grant/github/CosyVoice-1

---

Summary / Transcript (condensed)

1) High-level goal
- Enable Candle fused flash-attention (sdpa / flash-attn) in the Rust native server/LLM, verify at runtime, and squeeze GPU performance.

2) Progress so far (actions taken by the assistant during the session)
- Ported Flow attention to attempt `sdpa(q,k,v,mask,...)` and fall back to a manual matmul + softmax path when `sdpa` fails.
- Precomputed RoPE cos/sin in rotary embedding initialization to avoid repeated trig at runtime.
- Added `FLOW_ATTN_DEBUG=1` gated debug prints to trace attention shapes and decisions.
- Fixed multiple syntax fallout issues and a small HiFT logging dtype conversion bug.
- Built release binaries with CUDA features and ran the parity test `target/release/test_flow_parity`.
- Captured parity run logs to `rust/native-server/test_flow_parity.log`.
- Applied a small bug fix in `rust/native-server/src/hift.rs` to avoid shape/dtype multiplication mismatch (use `broadcast_mul` when multiplying scalar tensor with broadcastable tensor shapes).

3) Key runtime findings (from parity run)
- Attn backend selection: repeated message "[Attn WARNING] sdpa failed: no cuda implementation for metal-sdpa, falling back to manual path" — `sdpa` fused kernel not used on this machine/GPU (Turing RTX 2070 / CUDA 13.1) and fallback path used many times.

- Flow output contains many NaNs:
  [FlowParity] Rust Flow Output stats: min=inf, max=-inf, mean=0.000000, shape=[1, 80, 476] (NaN=38080, Inf=0)

- HiFT progressed further than earlier runs but then failed with a shape mismatch during f0 upsampling / multiply:
  Error: shape mismatch in mul, lhs: [1, 1, 228480], rhs: [1]

- Timing & memory (last parity run): Elapsed ~4.8s, Max RSS ~1.52 GB, exit status 1.

(Full parity log file path: `rust/native-server/test_flow_parity.log`)

4) Files changed by the assistant during troubleshooting
- `rust/native-server/src/hift.rs`
  - Fix: use `broadcast_mul` when multiplying a scalar tensor (noise_amp_scale) with a broadcastable tensor `(&ones - &uv)?` to avoid runtime shape mismatch.
  - Small snippet of change applied:

    - old: `let term2 = ((&ones - &uv)? * noise_amp_scale)?;`
    - new: `let term2 = ((&ones - &uv)?).broadcast_mul(&noise_amp_scale)?;`

  This change prevents `shape mismatch in mul` where the RHS was a scalar tensor and LHS was shape `[B,1,L]`.

5) Next recommended steps (tracked todo)
- Convert Flow output to `F32` before calling HiFT (quick unblock), or make HiFT tolerate `F16` inputs.
- Add numeric diagnostics inside `rust/native-server/src/flow.rs` (Attention::forward) to pinpoint where NaNs arise (log stats after q·k, after scaling, after softmax, after soft@v).
- Add a sdpa backend selection heuristic (avoid selecting `metal-sdpa` for CUDA device when that implementation isn't available) or map to an appropriate CUDA sdpa implementation.
- Improve manual fallback numeric stability (softmax eps, clamping, correct scaling) as needed.

6) Where to find artifacts
- Built release binaries: `rust/target/release` (e.g., `test_flow_parity`, `native_example`)
- Parity log: `rust/native-server/test_flow_parity.log`
- This exported chat: `/home/grant/github/CosyVoice-1/chat_export.md`

---

Appendix: parity run excerpt (most relevant lines)

[Attn WARNING] sdpa failed: no cuda implementation for metal-sdpa, falling back to manual path
... (repeated many times)
[FlowParity] Rust Flow Output stats: min=inf, max=-inf, mean=0.000000, shape=[1, 80, 476] (NaN=38080, Inf=0)
HiFT loaded.
Running HiFT on Reference Flow Output...
Reference Flow Output shape: [1, 80, 476]
[HiFT.forward] Starting...
input mel shape: [1, 80, 476]
[F0Predictor] input shape: [1, 80, 476]
F0 predictor output shape: [1, 1, 476]
F0 stats: min=0.052979 Hz, max=288.500000 Hz, mean=102.644226 Hz
upsampled f0 shape: [1, 1, 228480] (scale=480)
Error: shape mismatch in mul, lhs: [1, 1, 228480], rhs: [1]

(End excerpt)

---

If you want a different export format (JSON, PDF, or include complete raw logs), tell me which format and I will create it and add it to the repo (or package it for download).
