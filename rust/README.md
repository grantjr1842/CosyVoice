# CosyVoice Rust Components

High-performance Rust implementation of the CosyVoice TTS server and client.

## Components

- **bridge-server**: The PyO3-based bridge server that uses the Python backend.
- **native-server**: The 100% native Rust server (Candle + ONNX) with no Python dependencies.
- **client**: CLI client for testing the server.
- **shared**: Shared type definitions and utils.

## Build

```bash
# Build all components
cargo build --release
```

## Run servers

### Bridge Server
The bridge server requires the Python environment (PyO3).

```bash
cd ..
./rust/target/release/cosyvoice-bridge-server
```

### Native Server
The native server is standalone and does not require Python.

```bash
cd ..
./rust/target/release/cosyvoice-native-server
```

## Voice cloning smoke test

With either server running, use the example script to send a zero-shot request:

```bash
COSYVOICE_SERVER_URL=http://127.0.0.1:3000 \
COSYVOICE_PROMPT_AUDIO=asset/zero_shot_prompt.wav \
COSYVOICE_PROMPT_TEXT="You are a helpful assistant.<|endofprompt|>Greetings, how are you today?" \
python tests/test_rust_server_e2e.py
```

## Documentation

See the main [README](../README.md) for full project documentation.
