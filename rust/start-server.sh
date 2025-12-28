#!/usr/bin/env bash

# CosyVoice Rust Server Startup Script
#
# This script sets up the correct Python environment for the Rust server
# which uses PyO3 to bridge to CosyVoice Python backend.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load environment configuration from .env if present
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    echo "Loaded environment from .env"
fi

# Set up library path for libpython (using .env value or default)
LD_LIBRARY_PATH_EXTRA="${LD_LIBRARY_PATH_EXTRA:-.pixi/envs/default/lib}"
export LD_LIBRARY_PATH="$PROJECT_ROOT/$LD_LIBRARY_PATH_EXTRA:${LD_LIBRARY_PATH:-}"

echo "Starting CosyVoice Rust server..."
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
echo "  Model dir: $COSYVOICE_MODEL_DIR"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Use pixi run to ensure correct Python environment
exec pixi run "$SCRIPT_DIR/target/release/cosyvoice-server" "$@"
