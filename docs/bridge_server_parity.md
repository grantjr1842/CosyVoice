# Bridge Server Parity Reference (Rust/PyO3)

This file inventories every import and the concrete default/runtime values used by the original Rust/PyO3 bridge server. It is intended as a parity checklist for the native server.

## rust/bridge-server/src/main.rs

### Imports (exact)
- `axum::{extract::State, http::{header, StatusCode}, response::IntoResponse, routing::{get, post}, Json, Router}`
- `metrics::{counter, histogram}`
- `metrics_exporter_prometheus::PrometheusBuilder`
- `std::{env, net::SocketAddr, sync::Arc, time::Instant}`
- `tokio::signal`
- `tower_http::trace::TraceLayer`
- `tracing::{info, error, warn}`
- `tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt}`
- `shared::{config, ErrorResponse, HealthResponse, SynthesizeRequest}`
- `mod tts; use tts::TtsEngine`

### Globals and constants
- `#[global_allocator] GLOBAL`: `tikv_jemallocator::Jemalloc`

### Environment variables and defaults
- `COSYVOICE_MODEL_DIR`: if unset, falls back to `shared::config::DEFAULT_MODEL_DIR` (`pretrained_models/Fun-CosyVoice3-0.5B`)
- `.env` loading: `dotenvy::from_filename(".env")` (warnings only)
- `LD_LIBRARY_PATH_EXTRA`: default `.pixi/envs/default/lib` (used in `ensure_library_path`)
- `LD_LIBRARY_PATH`: updated to include `LD_LIBRARY_PATH_EXTRA` if missing (used in `ensure_library_path`)
- `CONDA_SHLVL`: used to detect pixi env (`in_pixi = CONDA_SHLVL != "0"`)
- `_COSYVOICE_REEXEC`: sentinel to avoid infinite re-exec loops

### Runtime values and defaults
- `model_dir`: `env::var("COSYVOICE_MODEL_DIR")` or `DEFAULT_MODEL_DIR`
- `addr`: `0.0.0.0:config::DEFAULT_PORT` (`DEFAULT_PORT = 3000`)
- `Router` routes: `POST /synthesize`, `GET /health`, `GET /speakers`, `GET /metrics`
- `TraceLayer::new_for_http()` enabled
- Prometheus metrics recorder installed via `PrometheusBuilder::new().install_recorder()`

### Synthesis handler behavior
- `prompt_audio` is **required** for CosyVoice3 (error message: `"prompt_audio is required for CosyVoice3 synthesis"`)
- If `request.prompt_text` is present → `tts.synthesize_zero_shot(text, prompt_audio, prompt_text, speed)`
- Else → `tts.synthesize_instruct(text, instruct, prompt_audio, speed)`
- `instruct` default: `request.speaker` or `"Speak naturally in English."`
- WAV encoding: `hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: Int }`
- Response headers: `Content-Type: audio/wav` and `X-Audio-Duration` (string seconds)

### Metrics emitted
- Counters: `tts_requests_total`, `tts_requests_success`, `tts_requests_error`
- Histograms: `tts_synthesis_duration_seconds`, `tts_rtf`

### ensure_library_path() behavior
- If `LD_LIBRARY_PATH` doesn’t include `LD_LIBRARY_PATH_EXTRA` or not in pixi, re-executes via `pixi run <exe>` and sets `_COSYVOICE_REEXEC=1`

## rust/bridge-server/src/tts.rs

### Imports (exact)
- `pyo3::prelude::*`
- `pyo3::types::PyList`
- `std::sync::{Arc, Mutex}`
- `thiserror::Error`

### Python module wiring (exact)
- `sys.path` insertion: `"/home/grant/github/CosyVoice-1"`
- Module: `cosyvoice.cli.cosyvoice`
- Class: `CosyVoice3`
- Constructor: `CosyVoice3(model_dir)`
- Sample rate: `model.sample_rate` (read from Python model)

### Synthesis methods (exact)
- `synthesize_zero_shot(text, prompt_audio_path, prompt_text, _speed)`
  - Calls: `model.inference_zero_shot(text, prompt_text, prompt_audio_path)`
  - Consumes generator; concatenates `output["tts_speech"]`
  - Converts float samples to `i16` with `s * 32767.0` and clamp
- `synthesize_instruct(text, instruct_text, prompt_audio_path, _speed)`
  - Calls: `model.inference_instruct2(text, instruct_text, prompt_audio_path)`
  - Same conversion to `i16`
- `list_speakers()`: calls `model.list_available_spks()`

## Native-vs-Bridge Parity Notes (current)

### Direct mismatches observed
- Output length: native example generates much longer clips (23.20s, 19.20s) vs Python example (6.52s, 6.72s).
- Native LLM sampling is **top-k only**; Python uses `ras_sampling` with `top_p=0.8`, `top_k=25`, `win_size=10`, `tau_r=0.1`.
- Native LLM stop criteria only checks `token_id >= sampling_vocab_size`; Python uses `stop_token_ids` and additional logic.
- Native uses `llm.safetensors`; Python prefers `llm.rl.pt` when available.
- Native uses random SOS/task-id embeddings (not loaded from weights); Python uses trained embeddings.
- Native DC-removes audio (mean subtraction in `synthesize_from_tokens`); Python returns raw float waveforms.

### Pre/post-processing differences
- Resampling: native uses rubato windowed-sinc; Python uses `torchaudio.transforms.Resample`.
- Prompt mel: native uses Rust `mel_spectrogram` implementation; Python uses `cosyvoice.compat.matcha_compat.mel_spectrogram`.
- Prompt speech tokens: native uses Whisper log-mel + ONNX; Python uses Whisper log-mel + ONNX (should match closely).
- Speaker embedding: native implements Kaldi fbank + mean normalization; Python uses `torchaudio.compliance.kaldi.fbank` + mean normalization.
- Text normalization: native uses a minimal English prefix (`<|en|>`), Python uses wetext/ttsfrd normalization and paragraph splitting.

## Example Output Comparison (reference)

Python example outputs:
- `output/voice_clone_0_0.wav`: 24kHz mono, 6.52s
- `output/voice_clone_1_0.wav`: 24kHz mono, 6.72s

Native example outputs:
- `output/native_voice_clone_0_0.wav`: 24kHz mono, 23.20s
- `output/native_voice_clone_1_0.wav`: 24kHz mono, 19.20s
