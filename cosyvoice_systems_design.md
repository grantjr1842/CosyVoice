# CosyVoice Systems Design (Painfully Thorough)

This document is a systems-design reference for **FunAudioLLM/CosyVoice** focused on *production deployment* and a path toward a **native Rust** runtime.
It emphasizes **streaming**, **ONNX Runtime**, and **NVIDIA/TensorRT** optimization, while locking audio correctness requirements:
- **24 kHz** sample rate
- **PCM16LE**
- **deterministic chunk boundaries**

---

## 1) What CosyVoice is (at the system level)

CosyVoice is a multi-stage TTS stack that can be viewed as three heavy inference components plus a frontend:

1. **Frontend / conditioning**
   - Text normalization, tokenization
   - Speaker embedding and prompt conditioning (voice cloning / style prompt)
2. **LLM (text → speech tokens)**
   - Generates discrete speech token sequences conditioned on text + prompt.
3. **Flow Matching model (speech tokens → acoustic features / mel-like)**
   - Converts token stream to a continuous acoustic representation.
4. **HiFT vocoder (acoustic features → waveform)**
   - Generates the final audio waveform.

The codebase also supports **streaming** synthesis by maintaining **per-session caches** (flow cache, vocoder cache, overlap buffers, etc.) and emitting audio chunks as tokens arrive.

---

## 2) Repository contracts that matter for native runtime

Even if you eventually discard the Python server, CosyVoice establishes several *runtime contracts* that your Rust implementation must preserve:

### 2.1 Model directory contract

CosyVoice expects a **model_dir** that contains (at minimum) a YAML config and multiple artifacts referenced by code.

From `cosyvoice/cli/cosyvoice.py` (CosyVoice class):
- Requires `{model_dir}/cosyvoice.yaml`
- Loads:
  - `{model_dir}/campplus.onnx` (speaker embedding)
  - `{model_dir}/speech_tokenizer_v1.onnx`
  - `{model_dir}/spk2info.pt`
  - `{model_dir}/llm.pt`, `{model_dir}/flow.pt`, `{model_dir}/hift.pt`
- Optional accelerations:
  - JIT zip bundles (e.g. `llm.text_encoder.*.zip`, `llm.llm.*.zip`, `flow.encoder.*.zip`)
  - TensorRT plan for `flow.decoder.estimator` (e.g. `flow.decoder.estimator.*.mygpu.plan`) + its ONNX (`flow.decoder.estimator.fp32.onnx`)

**Implication for Rust:** treat `model_dir` as a *versioned bundle* with strict invariants, and validate it at startup.

### 2.2 Streaming/state contract

`cosyvoice/cli/model.py` uses per-session dictionaries keyed by UUID to store:
- speech token stream buffer (`tts_speech_token_dict`)
- end-of-LLM signal (`llm_end_dict`)
- overlap buffers and caches:
  - `mel_overlap_dict`
  - `flow_cache_dict`
  - `hift_cache_dict`

**Implication for Rust:** implement a **Session** object with explicit lifecycle management. Avoid “global dicts”; use `DashMap<Uuid, SessionState>` or a dedicated actor/task per session.

---

## 3) Component-level design

### 3.1 Frontend

Frontend responsibilities (as used in `CosyVoice.__init__`):
- Text normalization and splitting (the code iterates `text_normalize(..., split=True)`).
- Speaker prompt ingestion:
  - Zero-shot prompt text + prompt wav processed into prompt features/tokens and stored in `spk2info`.
- Provides model inputs for:
  - SFT inference (`frontend_sft`)
  - Zero-shot / voice cloning (`frontend_zero_shot`)
  - Potentially instruction-based conditioning (model variants)

**Rust-native design goal:** keep the frontend as a separate crate/module because it is the most change-prone part (tokenizers, feature extraction, wav IO, etc.).

### 3.2 LLM (token generator)

The LLM emits discrete speech tokens. In `CosyVoiceModel.llm_job`, it:
- runs `llm.inference(...)` for non-streaming text, or
- `llm.inference_bistream(...)` for generator-based streaming text (CosyVoice2 only; vLLM path disables this).

**Key performance detail:** the code creates a dedicated CUDA stream context (`torch.cuda.Stream`) for LLM work. That’s a hint: isolate LLM execution from flow/vocoder execution to reduce tail latency.

### 3.3 Flow Matching model (token → mel-like features)

In `CosyVoiceModel.token2wav`, flow inference returns:
- `tts_mel`
- an updated `flow_cache` for streaming continuation

Then the code:
- applies **mel overlap** fade-in/out (windowed crossfade) using a Hamming window.
- appends cached mel segments for vocoder continuity.

The chunk sizing behavior differs by model generation:
- CosyVoiceModel (v1): has `token_min_hop_len`, `token_max_hop_len`, `token_overlap_len`, and computes `mel_overlap_len` using constants that historically assume 22050 Hz and hop size 256 in that specific path.
- CosyVoice2/3: use `token_hop_len = 25` and have different cache sizing; CosyVoice3 enforces `token_hop_len` as “must match training static_chunk_size”.

**Rust-native design goal:** mirror the hop/overlap policy as a first-class configuration object and log it with every session for reproducible debugging.

### 3.4 HiFT vocoder (mel → waveform)

The vocoder is invoked with `hift.inference(speech_feat=tts_mel, cache_source=...)` and returns:
- waveform chunk (`tts_speech`)
- updated source cache (`tts_source`)

The code then does **speech crossfade** between cached and new waveform segments using another Hamming window to avoid clicks/pops at chunk boundaries.

**Rust-native design goal:** treat vocoder output as `Vec<i16>` (PCM16LE) and apply crossfade in integer space carefully (or float and then clamp/convert).

---

## 4) Control plane / orchestration

### 4.1 Sessions, concurrency, and backpressure

The Python code uses:
- a per-model `lock` and multiple dicts
- thread-based background jobs (`threading`) and `uuid` keys
- explicit cleanup: delete session state, `torch.cuda.empty_cache()`, stream sync

For a Rust runtime you want:
- **One session = one state machine**
- clear separation between:
  - **token producer** (LLM)
  - **mel producer** (flow)
  - **audio producer** (vocoder)
- backpressure: a bounded channel between stages to avoid unbounded token queues.

A solid shape is:

- `SessionManager` (accept requests, allocate session IDs)
- `SessionTask` per session that owns:
  - `FrontendState`
  - `LlmState`
  - `FlowState`
  - `VocoderState`
  - `ChunkPolicy` (hop/overlap, target latency)
- stage pipelines connected by `tokio::sync::mpsc` bounded queues

### 4.2 Determinism and debug-ability

For “debugging that isn’t interpretive dance”, enforce:
- deterministic split policy (same text chunking every run)
- deterministic token hop length and overlap lengths
- deterministic RNG usage (seeded) when sampling is used
- persistent logs per chunk: (chunk_idx, token_range, mel_range, pcm_range, wall time, GPU time)

---

## 5) Phase 2: Replace the server with ONNX Runtime in Rust

### 5.1 What’s already exportable today

The repo already contains a concrete ONNX export script for the **flow decoder estimator**:

- `cosyvoice/bin/export_onnx.py` exports `flow.decoder.estimator` to `flow.decoder.estimator.fp32.onnx` (opset 18, dynamic axis on seq_len) and then validates ORT vs PyTorch with `assert_allclose`.  
- It selects ORT providers: CUDA if available, otherwise CPU.

This is the “most common endgame” starting point: get the flow decoder estimator running in-process.

### 5.2 Incremental export plan (recommended)

Export order that reduces risk and keeps you shipping:

1. **Export flow decoder estimator** (already implemented).
2. Export **vocoder** (HiFT) to ONNX (or export a subgraph that maps `mel -> audio` with cache inputs/outputs).
3. Export **flow encoder** and the rest of flow inference graph (token → mel).
4. Export **LLM** last (highest risk: attention variants, KV cache, dynamic shapes, sampling).

You can already see effort in the transformer attention implementation to simplify ONNX export. The attention module includes explicit notes about feeding zero-shaped cache tensors to force a consistent control path during export (avoiding dynamic branches).

### 5.3 Rust integration architecture with onnxruntime-rs

Target architecture:

- `onnx` crate module containing:
  - `OrtEnv` singleton (global)
  - `OrtSession` wrappers per model graph
  - typed tensor adapters (`ndarray` or direct `OrtValue` APIs)
- one session per model graph per device, with:
  - CPU session for frontend artifacts (if any)
  - CUDA EP sessions for heavy graphs

**Key requirements to get right early:**
- input/output tensor layouts (NCHW vs NTC etc.)
- int32 vs int64 token types (CosyVoice flow expects int32 tokens)
- dynamic shapes: preserve `seq_len` as dynamic, but constrain min/opt/max for TRT later

### 5.4 Expected ONNX export pain points (and how to de-risk)

**Known problematic area:** attention kernels / custom ops / dynamic control flow.

Practical de-risking:
- build a small ONNX “export CI” that exports every graph and runs:
  - `onnx.checker.check_model`
  - ORT inference sanity checks on fixed dummy inputs
- keep a “golden” set of prompt/text inputs and assert:
  - token sequences equal
  - mel statistics within tolerance
  - waveform hash within tolerance (or perceptual metrics)

---

## 6) Phase 3: NVIDIA optimization (TensorRT, pinned memory, async streams)

CosyVoice already has a TensorRT path for the **flow decoder estimator**:
- `CosyVoiceModel.load_trt(...)` converts an ONNX model to TRT and loads a `.plan` engine, replacing `self.flow.decoder.estimator` with a TensorRT wrapper.
- It defines TRT min/opt/max shapes and input names.

### 6.1 TensorRT endgame

You will likely end up with:
- TRT engine for **flow decoder estimator** (already supported conceptually)
- TRT engine for **vocoder** (often the biggest wall time per second of audio)
- possibly TRT engine for **flow encoder** / other flow subgraphs
- LLM may be on:
  - TRT-LLM, or
  - a separate inference stack (vLLM-like), or
  - remain in PyTorch if export remains painful

### 6.2 Pinned memory and async transfers

To reduce first-packet latency:
- keep all recurrent caches on GPU
- use pinned host buffers for the final PCM chunk
- overlap:
  - flow inference (GPU)
  - vocoder inference (GPU)
  - host transfer (DMA)
  - audio encoding / streaming write (CPU)

In Rust, that means:
- explicit CUDA stream handling (or using ORT CUDA EP’s internal streams carefully)
- a double-buffering scheme for audio output

### 6.3 Chunk-size tuning (latency vs throughput)

CosyVoice’s effective latency is the sum of:
- LLM time to emit enough tokens for first hop
- flow time for hop tokens
- vocoder time for hop mel frames
- crossfade and output overhead

Smaller hops reduce time-to-first-audio but increase overhead per second; larger hops improve throughput but increase latency.

The Python code bakes in hop sizes (e.g., `token_hop_len = 25` for CosyVoice2/3). In Rust, make hop size and overlap explicit config knobs but keep defaults aligned with training.

---

## 7) Audio correctness requirements

Lock these and enforce them in tests:

### 7.1 Sample rate and PCM format
- **24,000 Hz** output sample rate
- **PCM16LE** on the wire
- all internal wav IO must normalize to float32 [-1, 1] before model and convert back at output

### 7.2 Deterministic chunk boundaries
- Choose one and only one chunking policy per model family:
  - token hop length
  - mel overlap length
  - vocoder cache length
  - speech crossfade length
- Make chunk boundaries observable: include offsets in logs and return metadata with each audio chunk.

### 7.3 Validation / golden tests
- Provide a deterministic end-to-end test:
  - fixed prompt wav
  - fixed prompt text
  - fixed tts text
  - fixed chunk policy
- Output:
  - total PCM length
  - per-chunk PCM hashes
  - optional perceptual similarity score for regression tolerance

---

## 8) Proposed Rust-native implementation layout

Recommended crate split:

- `cosyvoice-core`
  - session manager, chunk policy, state machine, tracing/logging
- `cosyvoice-frontend`
  - text normalization, tokenizer bindings, wav IO, prompt feature extraction
- `cosyvoice-ort`
  - onnxruntime-rs wrappers, graph I/O adapters, device/provider selection
- `cosyvoice-trt` (optional)
  - TensorRT engine building/loading, bindings, stream mgmt
- `cosyvoice-cli`
  - smoke tests, benchmark harness, export helpers
- `cosyvoice-server` (optional)
  - gRPC/WebSocket streaming server wrapper (thin)

---

## 9) Deployment patterns

### 9.1 In-process (native library)
- your app links to `cosyvoice-core` and runs sessions in-process
- best latency, simplest integration with audio pipelines

### 9.2 Sidecar service (still native)
- deploy as a local gRPC service (Rust)
- isolate model memory from app, easier hot reload

### 9.3 Hybrid migration path
- keep Python frontend and LLM, but move flow+vocoder to Rust+ORT/TRT first
- then migrate LLM once you have stable audio outputs

---

## 10) Checklist: “done” means this is true

- [ ] Model bundle validated at startup; errors are explicit and actionable.
- [ ] Streaming output is deterministic: same input yields same chunk boundaries and same hashes (within tolerance if FP16/TRT).
- [ ] ONNX Runtime path exists for at least flow decoder estimator and vocoder.
- [ ] TensorRT path exists for the heaviest graphs with pinned-memory output.
- [ ] Benchmarks quantify first-packet latency, RTF, and GPU utilization.
- [ ] Regression suite catches export breakage (attention ops, dynamic shape bugs) early.

---

## Appendix: Key code references (source paths)

- `cosyvoice/cli/cosyvoice.py`: model bundle loading and high-level inference entrypoints.
- `cosyvoice/cli/model.py`: session caching, streaming logic, TRT integration.
- `cosyvoice/bin/export_onnx.py`: ONNX export (flow decoder estimator) and ORT validation.
- `cosyvoice/transformer/attention.py`: attention export notes and cache-handling simplification for ONNX.
