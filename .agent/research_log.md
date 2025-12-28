# Research Log: Rust TTS Server & Client

## Date: 2025-12-27

## Request
Create a native Rust text-to-speech server and client with maximum performance.

---

## Previous Research (Preserved)

### Advanced Optimizations (2025-12-27)
| Package | Version |
|---------|---------|
| bitsandbytes | 0.49.0 |
| accelerate | 1.12.0 |
| flash-attn | 2.8.3 |

---

## Native Rust TTS Options Researched

### 1. sherpa-rs (RECOMMENDED) ⭐
- **Crate**: `sherpa-rs` → bindings to `sherpa-onnx`
- **TTS Model**: **Kokoro TTS** (82M params, quantized ~40MB)
- **Inference**: ONNX Runtime via native C++ bindings
- **Features**: Supports TTS, STT, speaker embedding, diarization
- **Performance**: Near real-time on CPU, significantly faster with GPU
- **Platforms**: Windows, Linux, macOS, Android, iOS

### 2. piper-rs
- Pure Rust Piper TTS implementation
- Uses `ort` crate for ONNX Runtime
- G2P via `mini-bart-g2p` (no espeak-ng dependency)
- ~120ms generation for 1.9s audio on Intel i5

### 3. rten (Rust Tensor Engine)
- Pure Rust ONNX inference engine
- CPU-only with SIMD (AVX2, AVX-512, NEON, WASM SIMD)
- Actively working toward ONNX Runtime parity
- Can run Piper models in pure Rust

### 4. tts crate
- High-level cross-platform TTS interface
- Uses OS native backends (SAPI, Speech Dispatcher, AppKit)
- Not ML-based, simpler but less flexible

---

## Architecture Decision

**Selected**: `sherpa-rs` with Kokoro TTS model

**Rationale**:
1. **Best performance**: Leverages ONNX Runtime's optimized C++ backend
2. **Production-ready**: Active development, well-documented API
3. **Model quality**: Kokoro TTS has excellent voice quality
4. **Streaming support**: WebSocket streaming available
5. **Multi-platform**: Single codebase works across OS targets

---

## Performance Optimizations Identified

### Server-Side
| Optimization | Implementation |
|--------------|----------------|
| Memory allocator | `tikv-jemallocator` for reduced fragmentation |
| Async runtime | `tokio` multi-threaded runtime |
| HTTP framework | `axum` (tower-based, zero-copy) |
| Connection pooling | Keep-alive enabled by default |
| Request batching | Optional concurrent synthesis |
| Audio encoding | Stream raw PCM, client-side WAV encoding |

### Client-Side
| Optimization | Implementation |
|--------------|----------------|
| Async HTTP | `reqwest` with connection reuse |
| Stream decoding | Process audio chunks as received |
| Zero-copy I/O | `bytes` crate for buffer management |

---

## Model Download URLs

Kokoro TTS ONNX models available from sherpa-onnx releases:
- Base: `kokoro-v1-multi.onnx` (~330MB)
- Quantized int8: `kokoro-v1-multi.int8.onnx` (~85MB)
- Voices: Multiple included (American English, British English)

---

## Dependencies Summary

```toml
# Server
axum = "0.8"
tokio = { version = "1", features = ["full"] }
sherpa-rs = "0.6"
tracing = "0.1"
tracing-subscriber = "0.3"
metrics = "0.24"
metrics-exporter-prometheus = "0.16"
tikv-jemallocator = "0.6"

# Client
reqwest = { version = "0.12", features = ["stream"] }
clap = { version = "4", features = ["derive"] }
hound = "3.5"  # WAV encoding
```
