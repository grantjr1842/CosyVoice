# Hayne's Manual: Python TTS Server Implementation (CosyVoice-1)

**Models Covered:** Fun-CosyVoice3-0.5B-2512
**Version:** 3.0.0
**Complexity Level:** Advanced

---

## 1. Introduction

This manual provides a deep dive into the Python TTS server implementation for the CosyVoice project. It is designed for developers who need to understand, modify, or debug the serving layer. The implementation is split into two primary interfaces: a **FastAPI** based REST server and a **gRPC** server, both powered by the same underlying `CosyVoice3` engine.

---

## 2. System Architecture

The following diagram illustrates the high-level data flow from a client request to the audio output.

```mermaid
graph TD
    Client([Client])
    subgraph "Server Layer"
        FastAPI[FastAPI Server]
        GRPC[gRPC Server]
    end
    subgraph "Application Layer"
        CV3_Class[CosyVoice3 Class]
        Frontend[CosyVoiceFrontEnd]
    end
    subgraph "Model Layer"
        LLM[LLM (Speech Generation)]
        Flow[Flow Matching]
        HiFT[HiFT Vocoder]
        CampPlus[CampPlus (Speaker Emb)]
    end

    Client -- HTTP Request --> FastAPI
    Client -- gRPC Call --> GRPC
    FastAPI -- Calls --> CV3_Class
    GRPC -- Calls --> CV3_Class
    CV3_Class -- Text Norm & Tokenization --> Frontend
    Frontend -- Tokens & Embeddings --> CV3_Class
    CV3_Class -- Inference --> LLM
    LLM --> Flow
    Flow --> HiFT
    HiFT -- Audio Waveform --> CV3_Class
    CV3_Class -- Yields Audio --> FastAPI
    CV3_Class -- Yields Audio --> GRPC
    FastAPI -- StreamingResponse --> Client
    GRPC -- Stream Response --> Client
```

---

## 3. Server Implementations

The project provides two entry points for the server, located in `runtime/python`.

### 3.1 FastAPI Server (`runtime/python/fastapi/server.py`)

This server exposes a RESTful API and is built using `fastapi` and `uvicorn`.

*   **Port:** Defaults to `50000` (configurable via `--port`).
*   **Concurrency:** Handled by `uvicorn`'s async worker implementation.
*   **Response Type:** `StreamingResponse` (yields chunks of raw PCM audio bytes).

#### Endpoints

| Endpoint | Method | Description | Key Parameters |
| :--- | :--- | :--- | :--- |
| `/inference_zero_shot` | GET/POST | Zero-shot voice cloning | `tts_text`, `prompt_text`, `prompt_wav` |
| `/inference_cross_lingual` | GET/POST | Cross-lingual synthesis | `tts_text`, `prompt_wav` |
| `/inference_instruct2` | GET/POST | Instruction-controlled synthesis | `tts_text`, `instruct_text`, `prompt_wav` |
| `/health` | GET | Health check | None |

> [!NOTE]
> The `inference_zero_shot` endpoint automatically appends the prefix `"You are a helpful assistant.<|endofprompt|>"` to the prompt text to guide the model.

### 3.2 gRPC Server (`runtime/python/grpc/server.py`)

This server uses `grpcio` and is defined by the `cosyvoice.proto` specification.

*   **Port:** Defaults to `50000` (configurable via `--port`).
*   **Concurrency:** Multithreaded, defaults to 4 workers (`--max_conc`).
*   **Service Name:** `CosyVoice`
*   **RPC Method:** `Inference` (Bi-directional streaming not explicitly shown, but response is a generator).

#### Request Handling
The `Inference` method checks `request.HasField()` to determine the type of inference:

1.  **`zero_shot_request`**: Corresponds to `inference_zero_shot`.
2.  **`cross_lingual_request`**: Corresponds to `inference_cross_lingual`.
3.  **`instruct_request`**: Corresponds to `inference_instruct2`.

Each request block converts the input audio bytes to a float32 torch tensor (normalized by $2^{15}$) before passing it to the engine.

### 3.3 Rust-Python Hybrid Server (`rust/server`)

For high-performance production deployments, the project provides a hybrid server written in Rust (`axum`) that embeds the Python engine via `PyO3`. This combines the safety and concurrency of Rust with the model flexibility of Python.

*   **Binary:** `cosyvoice-server`
*   **Source:** `rust/server/src/main.rs`
*   **Architecture:**
    *   **Axum:** Handles HTTP requests/responses (async).
    *   **Tokio:** Manages the async runtime.
    *   **PyO3:** Bridges Rust and Python. The Python interpreter is embedded within the Rust process.
    *   **Threading:** Python inference is CPU/GPU-bound and holds the GIL. Request handlers use `tokio::task::spawn_blocking` to offload inference to a thread pool, preventing blocking of the async event loop.

#### Key Components

*   **`TtsEngine` (`rust/server/src/tts.rs`)**:
    *   A thread-safe wrapper (`Arc<Mutex<PyObject>>`) around the Python `CosyVoice3` instance.
    *   Manages the GIL for every call.
    *   Methods: `synthesize_zero_shot`, `synthesize_instruct`.
    *   Converts Rust types to Python types (e.g., `Vec<f32>` -> `numpy` -> `i16`).

*   **Smart Environment Setup**:
    *   The `ensure_library_path` function in `main.rs` dynamically sets `LD_LIBRARY_PATH` and re-executes the binary if needed. This allows the compiled Rust binary to run directly without a shell wrapper, automatically finding the correct Python shared libraries within the `pixi` environment.

---

## 4. Under the Hood: The Engine

The core logic resides in the `CosyVoice3` class within `cosyvoice/cli/cosyvoice.py`.

### 4.1 Initialization (`__init__`)

When `AutoModel` is called:
1.  **Model Download:** It checks if the model exists locally. If not, it uses `modelscope.snapshot_download`.
2.  **Config Loading:** Reads `cosyvoice3.yaml`.
3.  **Frontend Setup:** Initializes `CosyVoiceFrontEnd` with tokenizer, feature extractor, and ONNX models (`campplus.onnx`, `speech_tokenizer_v3.onnx`).
4.  **Hardware Optimization:**
    *   Detects if `torch.cuda.is_available()`.
    *   Configures MatMul precision (Ampere+ GPUs).
    *   Can load **TensorRT** engines (`flow.decoder.estimator.*.plan`) if `load_trt=True` (or auto-detected).
    *   Can load into **FP16** mode.
5.  **Model Loading:**
    *   Loads LLM (supports RL-optimized `llm.rl.pt` if available).
    *   Loads Flow and HiFT models.

### 4.2 The Inference Pipeline (`inference_zero_shot`)

The process for a zero-shot request is as follows:

1.  **Text Normalization:**
    *   `prompt_text` is normalized.
    *   `tts_text` is normalized and **split** into chunks for streaming.
2.  **Frontend Processing (`frontend.frontend_zero_shot`)**:
    *   Extracts speaker embeddings from `prompt_wav` using Campplus.
    *   Extracts speech tokens using `speech_tokenizer_v3`.
    *   Prepares batch inputs for the model.
3.  **Model Inference (`model.tts`)**:
    *   Iterates through text chunks.
    *   **LLM:** Generates speech tokens from text tokens + prompt tokens.
    *   **Flow:** Refines the mel-spectrogram/features.
    *   **HiFT:** Converts features to waveform.
4.  **Yielding Audio**:
    *   Audio is yielded as a generator to the server wrapper.

### 4.3 Deep Dive: Model Internals

#### Flow Matching (`cosyvoice/flow`)
The Flow module is responsible for predicting Mel spectrogram features from speech tokens and speaker embeddings. It uses a **Conditional Flow Matching (CFM)** approach.

*   **Class:** `MaskedDiffWithXvec` or `CausalMaskedDiffWithXvec` (for streaming).
*   **Mechanism:**
    *   **Inputs:** Speech tokens (from LLM), Speaker Embedding (X-Vector), Prompt Mel Features.
    *   **Process:** It solves an Ordinary Differential Equation (ODE) to transform a prior distribution into the target Mel spectrogram.
    *   **Length Regulator:** Aligns the simplified speech tokens with the target time domain.
*   **Streaming:** The `CausalMaskedDiffWithXvec` uses a `pre_lookahead_len` to manage context windowing, allowing generation to proceed chunk-by-chunk without full future context.

#### HiFT Vocoder (`cosyvoice/hifigan`)
HiFT (HiFi-GAN + ISTFT) converts the Mel spectrograms generated by the Flow module into the final audio waveform.

*   **Class:** `HiFTGenerator` or `CausalHiFTGenerator`.
*   **Source-Filter Architecture:**
    1.  **F0 Predictor:** Estimates the fundamental frequency (pitch) from the Mel spectrogram.
    2.  **Source Module (`SourceModuleHnNSF`)**: Generates a raw excitation signal using a combination of **Sine Waves** (harmonic component based on F0) and **Gaussian Noise** (unvoiced component). This is critical for stabilizing pitch.
    3.  **Neural Filter:** A HiFi-GAN style generator that takes the Source signal and the Mel spectrogram as input to shape the final waveform.
*   **ISTFT Optimization:** Unlike standard HiFi-GAN which upsamples purely via ConvTranspose, HiFT uses **Inverse Short-Time Fourier Transform (ISTFT)** layers (`_istft`). This improves phase interaction and computational efficiency.
*   **Streaming:** Only `CausalHiFTGenerator` supports streaming. It manages a cache of the source signal (`cache_source`) to ensure phase continuity across chunk boundaries, avoiding "clicking" artifacts.

---

## 5. Operational Procedures

### Start Up Commands

**FastAPI:**
```bash
export PYTHONPATH=.
python3 runtime/python/fastapi/server.py --port 50000 --model_dir <path_to_model>
```

**gRPC:**
```bash
export PYTHONPATH=.
python3 runtime/python/grpc/server.py --port 50000 --model_dir <path_to_model> --max_conc 4
```

### Environment Variables
*   `CUDA_VISIBLE_DEVICES`: Controls which GPU is used.

---

## 6. Troubleshooting & Maintenance

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| **OOM Error** | GPU memory insufficient for FP32 | Use `--fp16` or check usage. |
| **"tensorrt module not found"** | TensorRT not installed but ONNX model present | Install tensorrt or ignore warning. |
| **Slow Inference** | Not using TensorRT or FP16 | Enable `load_trt=True` inside code or ensure ONNX models are present. |
| **Import Errors** | PYTHONPATH not set | Run `export PYTHONPATH=.` from repo root. |

> [!TIP]
> **Performance Tuning:** For production, always ensure **TensorRT** is active (`load_trt=True`) and **FP16** is enabled if your GPU supports it. The `AutoModel` function attempts to auto-enable TensorRT if it finds the ONNX file.

---

## 7. Standard Benchmarks

To ensure the server meets performance requirements, use the provided benchmark scripts.

### 7.1 Performance Metrics (`benchmark_performance.py`)

This script benchmarks the end-to-end TTS pipeline (Text -> Audio).

**Usage:**
```bash
python3 benchmark_performance.py
```

**Metrics Reported:**
*   **FTL (First Token Latency):** Time to first audio chunk. Critical for conversational agents.
*   **RTF (Real-Time Factor):** `Generation Time / Audio Duration`. Lower is better (target < 0.1).
*   **Throughput:** `Audio Duration / Generation Time`. Higher is better.

### 7.2 Text Normalization (`benchmark_wetext.py`)

This script isolates the text normalization step (`wetext`), which can be a bottleneck on CPU.

**Usage:**
```bash
python3 benchmark_wetext.py
```

**Key Observation:**
*   Initialization time (loading Regex models).
*   Latency difference between English (faster) and Chinese (slower due to TN complexity).

---

## 8. Migrating to Native Rust: A Parity Guide

Achieving 100% bitwise parity when rewriting the engine in native Rust requires strict attention to detail. Below is the definitive checklist for migration.

### 8.1 Tensor Operations
*   **Library:** Use `candle-core` or `ndarray` (with `ort` for ONNX).
*   **Dtypes:** Python `float` is `f64`, but PyTorch defaults to `float32`. Rust must use `f32` exclusively for weights and inference buffers.
*   **Layout:** PyTorch `Conv1d` expects `(Batch, Channels, Time)`. Many Rust crates (or raw pointers) might default to `(Batch, Time, Channels)`. Ensure explicit transposes match the Python traces.
*   **Broadcasting:** Verify that `broadcast_add` logic in Rust handles dimensions exactly like NumPy/PyTorch (right-aligned).

### 8.2 Randomness (The "Noise" Problem)
The Flow Matching decoder adds random noise to the ODE solver.
*   **Issue:** `torch.randn` with a specific seed produces a specific sequence. Rust's `rand::StdRng` will **not** match this sequence even with the same integer seed.
*   **Solution for Parity:**
    1.  **Injection:** Modify the Python code to accept pre-generated noise as an input tensor.
    2.  **Export:** Save this noise tensor to a file.
    3.  **Import:** Load this exact noise tensor in Rust directly.
    *   *Only after "Injection" parity is proven should you switch to a native Rust PRNG.*

### 8.3 Weight Loading (`safetensors`)
*   **Mapping:** The state dict keys in `pytorch_model.bin` or `model.safetensors` might use dot notation (`llm.layers.0`). Rust structs often assume a hierarchy.
*   **Renaming:** You may need a logic layer to map `model.layers.0.feed_forward.w1.weight` -> `layers[0].feed_forward.w1`.
*   **Shapes:** Transpose linear layer weights if your Rust matrix multiplication library expects `(In, Out)` but Torch stored `(Out, In)`.

### 8.4 Signal Processing (STFT/ISTFT)
The HiFT vocoder relies heavily on STFT.
*   **Padding:** Torch's `stft(center=True)` pads the signal by `n_fft // 2` on both sides with reflection/zeros. Rust FFT libraries (like `rustfft`) do **not** do this automatically. You must implement the padding manually.
*   **Window Function:** Ensure the Hann window is generated using the exact same formula: `0.5 * (1 - cos(2 * pi * n / (N - 1)))`. Minor float precision differences here lead to audible artifacts (ticks/pops).

### 8.5 Text Processing (`wetext`)
*   **Regex Engine:** Python's `re` module supports look-behinds/look-aheads that Rust's `regex` crate **does not**.
*   **Solution:** Use the `fancy-regex` crate in Rust for compatibility, or rewrite the regex patterns to be strictly linear if performance is paramount.

---

## 9. Extensive Parity Audit Report

An internal audit of the native Rust implementation (`rust/server/src`) against the Python reference has identified the following critical areas for achieving 100% parity.

### 9.1 Parity Hazards: Randomness (Critical)
Both the Flow Matcher and HiFT Vocoder inherently use random noise. The current Rust implementation generates this noise internally, which makes bitwise comparison with Python impossible.

*   **Flow Checker (`flow.rs`):**
    *   **Code:** `mu.randn_like(0.0, 1.0)?` inside `ConditionalCFM::forward`.
    *   **Risk:** Rust's RNG (even seeded) != PyTorch's RNG.
    *   **Fix:** Modify `ConditionalCFM::forward` to accept an optional `noise: Option<&Tensor>` argument. In Python, generate the noise tensor, save it, and pass it to Rust during debug runs.

*   **HiFT SineGen (`hift.rs`):**
    *   **Code:** `Tensor::rand` (Phase) and `Tensor::randn_like` (Noise) inside `SineGen::forward`.
    *   **Risk:** `SineGen` output will diverge immediately.
    *   **Fix:** Similar to Flow, `SineGen` must accept external noise tensors for parity testing.

### 9.2 Parity Hazards: Signal Processing
*   **ISTFT Phase Logic (`hift.rs`):**
    *   **Observation:** Rust calculates `phase = phase_in.sin()`.
    *   **Python Reference:** `phase = torch.sin(x)`. Then `_istft` calculates `real = mag * torch.cos(phase)`.
    *   **Discrepancy:** This implies the underlying math is `real = mag * cos(sin(x))`. Verify if `candle-core`'s STFT module implementation matches this exact behavior or expects raw angles.
*   **STFT Padding:**
    *   **Observation:** PyTorch `stft(center=True)` automatically pads the input.
    *   **Risk:** Rust FFT libraries require manual padding. Ensure `reflection` padding is used to match PyTorch's default behavior, not zero-padding.

### 9.3 Parity Hazards: Tensor Layouts
*   **RoPE (Rotary Embeddings):**
    *   **Observation:** Rust implements `[-x2, x1, -x4, x3]` rotation on interleaved frequencies `[f1, f1, f2, f2]`.
    *   **Risk:** `x_transformers` and other libraries often use `cat([x, x], dim=-1)` which results in `[f1, f2, f1, f2]`.
    *   **Verification:** Dump the `cos` and `sin` tensors from Python's RoPE layer and compare them element-wise with Rust's generated tables.

*   **Causal Padding:**
    *   **Observation:** `F0Predictor` uses asymmetric padding (3 Right for Layer 0, 2 Left for others).
    *   **Status:** This appears correct for the specific kernel sizes (4 and 3) to maintain causality, but requires strict verification against the Python `padding=0` + manual pad logic.
