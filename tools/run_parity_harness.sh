#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="$(date +%F)"
OUT_DIR="${1:-$ROOT/output/benchmarks/$DATE}"

MODEL_DIR="${MODEL_DIR:-$ROOT/pretrained_models/Fun-CosyVoice3-0.5B}"
PROMPT_WAV="${PROMPT_WAV:-$ROOT/asset/interstellar-tars-01-resemble-denoised.wav}"

if [[ -z "${PROMPT_TEXT-}" ]]; then
  PROMPT_TEXT="Eight months to Mars. Counter-orbital slingshot around 14 months to Saturn. Nothing's changed on that."
fi

if [[ -z "${TTS_TEXT-}" ]]; then
  TTS_TEXT="Hello! This is a test for parity verification."
fi

mkdir -p "$OUT_DIR"

FRONTEND_ARTIFACTS="${FRONTEND_ARTIFACTS:-$OUT_DIR/frontend_artifacts.safetensors}"
FULL_ARTIFACTS="${FULL_ARTIFACTS:-$OUT_DIR/debug_artifacts.safetensors}"

if [[ "${SKIP_PY_FRONTEND:-0}" != "1" ]]; then
  echo "==> Generating Python frontend artifacts"
  PYTHONPATH="$ROOT" pixi run python debug_scripts/dump_frontend.py \
    --model-dir "$MODEL_DIR" \
    --prompt-wav "$PROMPT_WAV" \
    --output "$FRONTEND_ARTIFACTS"
else
  echo "==> Skipping Python frontend artifacts (SKIP_PY_FRONTEND=1)"
fi

if [[ "${SKIP_PY_FULL:-0}" != "1" ]]; then
  echo "==> Generating Python full artifacts (LLM -> Flow -> HiFT)"
  PYTHONPATH="$ROOT" pixi run python debug_scripts/generate_fresh_artifacts.py \
    --model-dir "$MODEL_DIR" \
    --prompt-wav "$PROMPT_WAV" \
    --prompt-text "$PROMPT_TEXT" \
    --tts-text "$TTS_TEXT" \
    --output-dir "$OUT_DIR"
else
  echo "==> Skipping Python full artifacts (SKIP_PY_FULL=1)"
fi

pushd "$ROOT/rust" >/dev/null

if [[ "${SKIP_RUST_FRONTEND:-0}" != "1" ]]; then
  echo "==> Rust frontend parity"
  cargo run -p cosyvoice-native-server --bin check_frontend -- \
    --model-dir "$MODEL_DIR" \
    --artifacts-path "$FRONTEND_ARTIFACTS" \
    --prompt-wav "$PROMPT_WAV" | tee "$OUT_DIR/check_frontend.log"
else
  echo "==> Skipping Rust frontend parity (SKIP_RUST_FRONTEND=1)"
fi

if [[ "${SKIP_RUST_FLOW:-0}" != "1" ]]; then
  echo "==> Rust flow parity"
  cargo run -p cosyvoice-native-server --bin test_flow -- \
    --model-dir "$MODEL_DIR" \
    --artifacts-path "$FULL_ARTIFACTS" | tee "$OUT_DIR/test_flow.log"
else
  echo "==> Skipping Rust flow parity (SKIP_RUST_FLOW=1)"
fi

if [[ "${SKIP_RUST_HIFT:-0}" != "1" ]]; then
  echo "==> Rust HiFT parity"
  cargo run -p cosyvoice-native-server --bin test_hift -- \
    --model-dir "$MODEL_DIR" \
    --artifacts-path "$FULL_ARTIFACTS" | tee "$OUT_DIR/test_hift.log"
else
  echo "==> Skipping Rust HiFT parity (SKIP_RUST_HIFT=1)"
fi

popd >/dev/null

echo "Parity artifacts and logs saved to: $OUT_DIR"
