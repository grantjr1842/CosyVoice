
# Native Rust TTS Reference – CosyVoice-Oriented Design

## Purpose
This document is a **practical engineering reference** for building a **native Rust text-to-speech (TTS) system**
inspired by the CosyVoice architecture. It focuses on **runtime design, streaming semantics, model boundaries,
export pipelines, and GPU optimization**, rather than research novelty.

The goal is a **production-grade Rust TTS runtime** that can replace Python inference incrementally while
remaining faithful to CosyVoice model behavior.

---

## 1. CosyVoice Architectural Summary

CosyVoice (v2/v3) decomposes TTS into **three inference stages**:

1. **Text → Semantic Speech Tokens (LM)**
   - Autoregressive transformer
   - ~25 tokens / second
   - Encodes linguistic and prosodic intent

2. **Semantic Tokens → Mel Spectrogram (Flow Matching, FM)**
   - Converts semantics into acoustic structure
   - Injects speaker identity / reference audio
   - ~50 mel frames / second

3. **Mel → Waveform (Vocoder)**
   - Final waveform synthesis
   - 24 kHz output

Streaming is defined as **incremental text ingestion and incremental audio emission**, not post-hoc audio chunking.

---

## 2. Phase 2 — Replacing Python with ONNX Runtime (In-Process)

This is the **most common and most realistic endgame** for a Rust-native CosyVoice deployment.

### 2.1 Export Strategy (What to Export First)

Do **not** attempt to export everything at once.

**Recommended order:**
1. **Vocoder** (lowest risk, highest payoff)
2. **Flow Matching model**
3. **Language Model (LM)** (highest risk)

Why:
- Vocoders usually use convolutional or diffusion-style ops that export cleanly.
- FM models are heavier but still manageable.
- LMs frequently hit ONNX limitations around attention.

#### Practical Export Notes
- Export models with **fixed or semi-fixed shapes** when possible.
- Prefer **static batch size = 1** for streaming.
- Freeze sample rate and mel dimensions early.

---

### 2.2 ONNX Runtime Integration in Rust

Use `onnxruntime-rs` as the execution backend.

**Design principles:**
- One ORT `Environment` per process
- One `Session` per model graph
- Sessions are long-lived and reused

Example session layout:
```rust
struct OnnxBackend {
    env: Environment,
    vocoder: Session,
    fm: Session,
    lm: Option<Session>,
}
```

**Execution rules:**
- Never create sessions per request
- Pre-allocate input/output tensors
- Reuse buffers across inference calls

---

### 2.3 Attention Kernel Export Issues (LM Stage)

This is the main pain point.

Common failures:
- `ScaledDotProductAttention` not supported
- Dynamic sequence lengths rejected
- Unsupported fused ops

**Mitigation strategies:**
1. Replace attention with export-friendly variants (where possible)
2. Use `torch.onnx.export` with:
   - `opset_version >= 17`
   - explicit `attention_mask` inputs
3. Consider splitting LM into:
   - token embedding graph
   - decoder step graph (one token at a time)

If LM export becomes intractable:
- Keep LM in Python or libtorch
- Export FM + vocoder first (still removes most Python cost)

---

## 3. Phase 3 — NVIDIA Optimization (TensorRT Path)

Once ONNX inference is stable, NVIDIA optimization becomes attractive.

### 3.1 TensorRT Engines

Target components:
- **Vocoder** (largest compute cost)
- **Flow Matching model**

LM optimization is optional unless you require very low text-to-audio latency.

Workflow:
1. Export ONNX
2. Convert to TensorRT engine (offline)
3. Load engines at runtime

**Guidelines:**
- Use FP16 if quality allows
- Lock input shapes where possible
- One engine per model variant

---

### 3.2 Memory and Execution Model

To reach low latency, you must control memory explicitly.

**Key techniques:**
- Pinned (page-locked) host memory for audio buffers
- Async CUDA streams per stage
- Overlap compute:
  - LM token generation
  - FM mel synthesis
  - Vocoder waveform generation

Think in terms of a **pipeline**, not sequential inference.

---

### 3.3 Chunk Size and First-Packet Latency

CosyVoice latency is driven by:
- LM token generation speed
- FM chunk size
- Vocoder receptive field

**Tuning rules:**
- Smaller chunks → lower latency, higher overhead
- Larger chunks → better throughput, worse first audio delay

Recommended starting points:
- Semantic tokens: 2–4 tokens per step
- Mel frames: 10–20 frames per chunk
- Audio chunks: 20–40 ms

Measure **time-to-first-sample**, not just total synthesis time.

---

## 4. Audio Correctness (Non-Negotiable)

Lock these parameters early and never deviate:

- **Sample rate:** 24,000 Hz
- **Format:** PCM16LE
- **Channels:** mono
- **Chunk size:** deterministic, fixed

### Why This Matters
If any of these drift:
- Debugging becomes impossible
- Latency measurements lose meaning
- Audio artifacts become nondeterministic

**Rules:**
- No implicit resampling
- No float audio outside the vocoder
- Convert to PCM16LE immediately after vocoder output

---

## 5. Deterministic Streaming Model

### Required Properties
- Fixed-size ring buffers
- Explicit backpressure handling
- No hidden allocations in the hot path

### Reference API
```rust
struct TtsSession;

impl TtsSession {
    fn push_text(&mut self, text: &str);
    fn flush(&mut self);
    fn next_audio_chunk(&mut self) -> Option<&[i16]>;
}
```

This API remains stable regardless of backend (Python, ONNX, TensorRT).

---

## 6. Engineering Reality Check

A fully native Rust CosyVoice-style TTS stack is **achievable**, but only if done incrementally.

**Correct order of operations:**
1. Validate streaming semantics
2. Replace vocoder with ONNX
3. Replace FM with ONNX
4. Decide if LM replacement is worth the pain
5. Optimize with TensorRT only after correctness is locked

If you invert this order, you will suffer.

---

## 7. End State

When done correctly, you get:
- Deterministic, low-latency Rust TTS
- Backend-agnostic inference
- GPU-optimized execution where it matters
- Zero Python in production inference (if desired)

This is how CosyVoice becomes a **system**, not just a research repo.


---

## 8. ONNX Export Checklists (CosyVoice Components)

This section is intentionally **operational**, not theoretical.

### 8.1 Vocoder Export Checklist (Start Here)

**Why first:** Lowest graph complexity, highest runtime cost reduction.

**Checklist**
- [ ] Batch size fixed to 1
- [ ] Sample rate hardcoded to 24kHz
- [ ] Mel bins fixed (e.g. 80)
- [ ] No dynamic control flow
- [ ] Opset ≥ 17
- [ ] FP32 first, FP16 later
- [ ] Deterministic output verified against PyTorch

**Validation**
- Single mel chunk → PCM
- Multi-chunk concatenation identical to monolithic run (±1 LSB)

---

### 8.2 Flow Matching (FM) Export Checklist

**Risks:** Large tensors, memory pressure, shape drift.

**Checklist**
- [ ] Fixed mel frame chunk size (e.g. 10–20 frames)
- [ ] Speaker embedding passed explicitly
- [ ] No Python-side post-processing
- [ ] Attention masks static where possible
- [ ] Disable training-time noise schedules

**Validation**
- Token→mel alignment preserved
- Chunk boundaries do not introduce clicks or pitch drift

---

### 8.3 Language Model (LM) Export Checklist (Hard Mode)

**Reality check:** This is where most ONNX attempts fail.

**Checklist**
- [ ] Decoder-only graph
- [ ] One-token-per-step inference
- [ ] Explicit KV cache inputs/outputs
- [ ] Attention masks as tensors (no implicit causal logic)
- [ ] Avoid `scaled_dot_product_attention` if possible
- [ ] Static hidden size, head count, head dim

**Fallback Strategy**
If export fails:
- Keep LM in libtorch or Python
- Export FM + vocoder only
- You still eliminate most runtime overhead

---

## 9. onnxruntime-rs: Streaming Inference Pattern

### 9.1 Session Lifecycle

**Rules**
- One `Environment` per process
- One `Session` per model
- Sessions live for entire runtime

```rust
struct Ortx {
    vocoder: ort::Session,
    fm: ort::Session,
}
```

### 9.2 Zero-Allocation Hot Path

**Preallocate everything:**
- Input tensors
- Output tensors
- Scratch buffers

Never allocate in `next_audio_chunk()`.

### 9.3 Execution Pattern

```rust
fn run_vocoder(
    session: &Session,
    mel: &[f32],
    pcm_out: &mut [i16],
) {
    // Bind preallocated tensors
    // Run session
    // Convert float → PCM16LE
}
```

**Key rule:** Convert to PCM **immediately** after inference.

---

## 10. TensorRT Optimization Playbook (NVIDIA)

### 10.1 What to Convert to TensorRT

Convert in this order:
1. Vocoder
2. Flow Matching
3. LM (optional)

Do **not** TensorRT everything at once.

---

### 10.2 Engine Build Rules

- One engine per:
  - model
  - precision (FP32 / FP16)
  - chunk size
- Serialize engines to disk
- Never build engines at runtime

**Recommended flags**
- FP16 enabled
- Strict type constraints
- Explicit batch

---

### 10.3 CUDA Execution Model

**Required**
- One CUDA stream per stage
- Pinned host buffers for PCM
- Async memcpy

**Pipeline**
```
LM token gen ─┐
              ├─ FM mel gen ─┐
                              ├─ Vocoder ─ PCM out
Text input ───┘               └─ overlap compute
```

This overlap is where latency disappears.

---

## 11. Chunk-Size Tuning (Latency Control)

### 11.1 Why Chunk Size Matters

Latency = time to **first audible sample**, not total runtime.

### 11.2 Recommended Defaults

| Stage | Chunk Size |
|-----|-----------|
| LM | 2–4 tokens |
| FM | 10–20 mel frames |
| Audio | 20–40 ms |

Tune **only after correctness is locked**.

---

## 12. Audio Correctness Test Suite (Mandatory)

### Tests to Implement
- [ ] Bitwise PCM equality across runs
- [ ] Chunk-boundary continuity (no clicks)
- [ ] Identical output for chunked vs monolithic inference
- [ ] Fixed latency variance (<1ms jitter)

If these fail, **stop optimizing**.

---

## 13. Final Guidance

If you remember nothing else:

- Export **vocoder first**
- Lock **audio format forever**
- Streaming correctness beats raw throughput
- TensorRT is useless without determinism
- ONNX failures are normal — plan for partial success

This is how you build a Rust TTS system that survives real users.


---

## Companion Artifacts
This reference is complemented by:
- `ARCHITECTURE.md` (system design)
- `RUNBOOK.md` (operational procedures)
- `examples/vocoder_onnx_stream.rs` (minimal ORT vocoder runner)
- `scripts/build_trt_engines.sh` (TensorRT engine builder)
- `PLAN.md` + `AGENTS.md` (agent handoff spec)

---

## Deep dive: Phase 2 — Replace the Python server with ONNX Runtime in Rust

### What CosyVoice already exports today (use this as your “first win”)

CosyVoice includes an ONNX exporter for **Flow Decoder Estimator**:

- Script: `cosyvoice/bin/export_onnx.py`
- Export target: `{model_dir}/flow.decoder.estimator.fp32.onnx`
- Opset: 18
- Dynamic axis: `seq_len` on the time dimension
- Includes a built-in consistency test using ONNX Runtime (CUDA EP if available, otherwise CPU) with `torch.testing.assert_allclose`.

**Actionable Rust plan**
1. Run the exporter once per model bundle.
2. Load the generated ONNX with `onnxruntime-rs`.
3. Implement a micro-benchmark that calls the estimator repeatedly over realistic `seq_len` distributions (your streaming hop policy).

### Export order that minimizes pain

In practice, export complexity increases sharply as you move earlier in the pipeline:

1. **Flow Decoder Estimator** (already implemented; do it first).
2. **Vocoder (HiFT)**: export as `mel + cache -> pcm + cache`. This is usually the biggest runtime win.
3. **Flow Encoder / full Flow inference**: token → mel, with flow cache inputs/outputs.
4. **LLM** last: attention + KV cache + dynamic control flow + sampling. This is where exports tend to stall.

### Known trouble area: attention ops and cache shapes

CosyVoice’s transformer attention code includes explicit export-oriented logic (for example: making cache tensors “exist” even for the first chunk, and allowing split/concat on zero-shaped tensors) to simplify the exported graph.
Treat this as a signal that:
- your ONNX export CI should exercise first-chunk behavior, not just steady-state chunks
- you should lock cache tensor shapes and dtypes early

### Practical ONNX export CI (don’t skip this)

Add a CI job that does all of the following for each supported model:
- exports all target graphs
- runs `onnx.checker` validation
- runs ORT inference on fixed dummy inputs (and at least one “golden” real prompt)
- produces an artifact pack: `*.onnx`, I/O spec JSON, and a tiny test vector

This catches attention/kernel regressions early, before you spend hours debugging runtime segfaults that were really export issues.

---

## Deep dive: Phase 3 — NVIDIA optimization (TensorRT + pinned memory + async streams)

### What CosyVoice already does (and you should copy conceptually)

`cosyvoice/cli/model.py` contains a `load_trt(...)` path that:
- converts the ONNX estimator to a TensorRT engine if the `.plan` file is missing/empty
- replaces `flow.decoder.estimator` with a TensorRT execution wrapper
- defines min/opt/max shapes for TRT builds (this is essential for dynamic seq_len).

**Rust implication**
- keep one TRT engine per distinct graph + precision profile (fp16/fp32)
- build or load engines at startup (or via a lazy cache), not on the first request

### Pinned memory + async overlap (how you actually reduce first-packet latency)

The latency you care about is not “RTF” in isolation; it’s the end-to-end chain:

LLM emits tokens → Flow turns tokens into mel → Vocoder turns mel into PCM → you ship first bytes.

To shrink time-to-first-audio:
- keep recurrent caches on GPU
- use pinned host buffers for the output PCM chunks
- overlap:
  - flow compute
  - vocoder compute
  - host transfer
  - chunk framing + network write

In Rust, implement double-buffered output chunks and ensure each stage can run without blocking the others.

### Chunk-size tuning (the hidden lever)

CosyVoice2/3 uses a hop-length concept (e.g., `token_hop_len = 25`) and caching to emit consistent chunks.
Smaller chunks reduce time-to-first-audio but increase per-chunk overhead and boundary artifacts risk.
Larger chunks improve throughput but increase latency.

**Rule:** pick one chunk policy, log it, and make it reproducible. If you change it mid-flight, your bugs become folklore.

---

## Audio correctness: lock these, then build a harness

### Hard requirements
- Output sample rate: **24,000 Hz**
- Wire format: **PCM16LE**
- Deterministic chunk boundaries (token hop, mel overlap, vocoder cache, crossfade window)

### Deterministic chunking: non-negotiables
- pick fixed hop/overlap/cache sizes per model family
- ensure “first chunk” uses the same structural graph path as subsequent chunks (export + runtime)
- record and return chunk metadata (sample offsets, token offsets, cache state IDs)

### Minimal correctness harness (recommended)
For a fixed prompt wav + prompt text + tts text:
- Assert identical chunk boundaries across runs.
- Compute per-chunk hashes (after PCM16LE conversion) and compare.
- Track summary stats: RMS, peak, silence ratio, total samples.
- Keep a “golden” run recorded in-repo for regression testing.

---

## Concrete mapping to CosyVoice repo artifacts (what to mirror in Rust)

From `cosyvoice/cli/cosyvoice.py`, the expected model bundle includes:
- `cosyvoice.yaml`
- `llm.pt`, `flow.pt`, `hift.pt`
- `campplus.onnx` (speaker embedding)
- `speech_tokenizer_v1.onnx`
- `spk2info.pt`

Optional accelerations referenced by the same file and `cosyvoice/cli/model.py`:
- TorchScript zips for llm + flow encoder
- TensorRT plan for `flow.decoder.estimator` built from `flow.decoder.estimator.fp32.onnx`

Your Rust runtime should validate these paths at startup and fail fast with clear diagnostics.


