#!/usr/bin/env bash

# CosyVoice Rust Server Startup Script
#
# This script sets up the LD_LIBRARY_PATH for libpython before launching the
# Rust server. The .env file is loaded by the Rust server itself via dotenvy.
#
# NOTE: LD_LIBRARY_PATH must be set here (not in Rust) because the dynamic
# linker needs it BEFORE the Rust binary loads to find libpython.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Set up library path for libpython (required by PyO3)
# This MUST be done in shell before the binary starts
export LD_LIBRARY_PATH="$PROJECT_ROOT/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"

echo "Starting CosyVoice Rust server..."
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
echo "  Model dir: $COSYVOICE_MODEL_DIR"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Use pixi run to ensure correct Python environment
exec pixi run "$SCRIPT_DIR/target/release/cosyvoice-server" "$@"
