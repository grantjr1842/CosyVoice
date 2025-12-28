#!/usr/bin/env bash

# CosyVoice Rust Server Startup Script
#
# This script runs the Rust server within pixi's Python environment.
# The server auto-configures LD_LIBRARY_PATH from .env via re-exec pattern.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Starting CosyVoice Rust server..."
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
echo "  Model dir: $COSYVOICE_MODEL_DIR"

# Use pixi run to ensure correct Python environment
# The server will auto-configure LD_LIBRARY_PATH from .env
exec pixi run "$SCRIPT_DIR/target/release/cosyvoice-server" "$@"
