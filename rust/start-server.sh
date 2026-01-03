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

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/.env"
    set +a
fi

if [ -n "${LD_LIBRARY_PATH_EXTRA:-}" ]; then
    # Ensure the server uses pixi's libpython and other env libs by default.
    LD_LIBRARY_PATH_EXTRA_ABS="$PROJECT_ROOT/${LD_LIBRARY_PATH_EXTRA}"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$LD_LIBRARY_PATH_EXTRA_ABS:"*) ;;
        *) export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_EXTRA_ABS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"

if [ "${COSYVOICE_ORT_USE_TRT:-0}" = "1" ]; then
    TRT_LIB_DIR="${COSYVOICE_TRT_LIB_DIR:-}"
    if [ -z "$TRT_LIB_DIR" ]; then
        TRT_LIB_DIR="$(pixi run python - <<'PY' 2>/dev/null || true
import os
try:
    import tensorrt_libs
    print(os.path.dirname(tensorrt_libs.__file__))
except Exception:
    pass
PY
)"
    fi
    if [ -n "$TRT_LIB_DIR" ] && [ -d "$TRT_LIB_DIR" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$TRT_LIB_DIR:"*) ;;
            *) export LD_LIBRARY_PATH="$TRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    else
        echo "Warning: COSYVOICE_ORT_USE_TRT=1 but TensorRT libs not found."
    fi
fi

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
