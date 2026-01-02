#!/usr/bin/env bash

# CosyVoice Rust Server Startup Script
#
# This script runs the Rust server within pixi's Python environment.
# The server auto-configures LD_LIBRARY_PATH from .env via re-exec pattern.
#
# Usage: ./start-server.sh [bridge|native]

SERVER_TYPE="${1:-bridge}"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"

if [ "$SERVER_TYPE" = "bridge" ]; then
    echo "Starting CosyVoice BRIDGE Server..."
    echo "  Model dir: $COSYVOICE_MODEL_DIR"
    # Bridge server needs pixi/python environment
    exec pixi run "$SCRIPT_DIR/target/release/cosyvoice-bridge-server" "${@:2}"
elif [ "$SERVER_TYPE" = "native" ]; then
    echo "Starting CosyVoice NATIVE Server..."
    echo "  Model dir: $COSYVOICE_MODEL_DIR"
    # Native server *might* need pixi libraries if linking against system libs provided by pixi
    # but theoretically should be more standalone.
    # For safety, we keep pixi run here too, especially for ORT/CUDA libs.
    exec pixi run "$SCRIPT_DIR/target/release/cosyvoice-native-server" "${@:2}"
else
    echo "Unknown server type: $SERVER_TYPE"
    echo "Usage: ./start-server.sh [bridge|native]"
    exit 1
fi
