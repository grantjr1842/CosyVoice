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

## Documentation

See the main [README](../README.md) for full project documentation.
