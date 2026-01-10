#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date +%F)"
OUT_DIR="${1:-$ROOT/output/benchmarks/$DATE}"

MODEL_DIR="${MODEL_DIR:-$ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
PROMPT_WAV="${PROMPT_WAV:-$ROOT/asset/interstellar-tars-01-resemble-denoised.wav}"
FRONTEND_ARTIFACTS="${FRONTEND_ARTIFACTS:-$OUT_DIR/frontend_artifacts.safetensors}"
FULL_ARTIFACTS="${FULL_ARTIFACTS:-$OUT_DIR/debug_artifacts.safetensors}"

pushd "$ROOT/rust" >/dev/null

echo "==> Rust frontend parity"
cargo run -p cosyvoice-native-server --bin check_frontend -- \
  --model-dir "$MODEL_DIR" \
  --artifacts-path "$FRONTEND_ARTIFACTS" \
  --prompt-wav "$PROMPT_WAV" | tee "$OUT_DIR/check_frontend.log"

echo "==> Rust flow parity"
cargo run -p cosyvoice-native-server --bin test_flow -- \
  --model-dir "$MODEL_DIR" \
  --artifacts-path "$FULL_ARTIFACTS" | tee "$OUT_DIR/test_flow.log"

echo "==> Rust HiFT parity"
cargo run -p cosyvoice-native-server --bin test_hift -- \
  --model-dir "$MODEL_DIR" \
  --artifacts-path "$FULL_ARTIFACTS" | tee "$OUT_DIR/test_hift.log"

popd >/dev/null

echo "Rust parity logs saved to: $OUT_DIR"
