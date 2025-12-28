# CosyVoice Rust Components

High-performance Rust implementation of the CosyVoice TTS server and client.

## Components

- **server**: The main TTS server (Axum + PyO3 + Torch)
- **client**: CLI client for testing the server
- **shared**: Shared type definitions and utils

## Build

```bash
# Build all components
cargo build --release
```

> **Note**: The build configuration in `.cargo/config.toml` automatically points PyO3 to the correct Python environment (pixi or system). You do not need to wrap the build command with `pixi run`.

## Run Server

Run from the project root (parent directory) to ensure `.env` is loaded correctly:

```bash
cd ..
./rust/target/release/cosyvoice-server
```

## Documentation

See the main [README](../README.md) for full project documentation.
