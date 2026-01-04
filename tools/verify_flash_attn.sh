#!/bin/bash

# Verify that candle-flash-attn is present in the dependency tree for native-server

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NATIVE_SERVER_DIR="$PROJECT_ROOT/rust/native-server"

echo "Verifying Flash Attention configuration in $NATIVE_SERVER_DIR..."

cd "$NATIVE_SERVER_DIR"

# Check if candle-flash-attn appears in the dependency tree when cuda feature is enabled
if cargo tree --features cuda --invert candle-flash-attn | grep -q "candle-flash-attn"; then
    echo "✓ candle-flash-attn is correctly linked in dependency tree."
    exit 0
else
    echo "✗ candle-flash-attn NOT found in dependency tree."
    echo "Please ensure 'candle-transformers/flash-attn' is enabled in Cargo.toml dependencies or features."
    exit 1
fi
